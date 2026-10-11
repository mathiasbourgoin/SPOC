(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * C-family codegen must agree with the interpreter on two things the C
 * dialects get wrong when the IR is printed naively:
 *
 *   C-8  a [for] upper bound is evaluated ONCE, before the first iteration.
 *        A C [for (i = a; i <= bound; i++)] re-evaluates [bound] every
 *        iteration, which differs as soon as the body mutates what the bound
 *        reads.
 *   C-9  [thread_idx_x] and friends are SIGNED int32. CUDA/HIP [threadIdx.x]
 *        is unsigned, so [thread_idx_x - 1l < 0l] was always false and
 *        [(thread_idx_x - 5l) / 2l] was a huge positive number.
 *
 * Each kernel is run on the interpreter and, when present, on HIP, and both
 * are compared with an expected value computed by hand in OCaml. Self-skips
 * the HIP part off ROCm.
 ******************************************************************************)

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_hip.Hip_plugin.register ()

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

(* [to] bound read from a mutable variable the body decrements: 4 iterations
   when the bound is read once (j = 0..3), 2 when it is re-read. *)
let upto_mutating_bound =
  [%kernel
    fun (out : int32 vector) ->
      let open Sarek_stdlib.Std in
      let m = mut 3l in
      let count = mut 0l in
      for _j = 0 to m do
        m := m - 1l ;
        count := count + 1l
      done ;
      out.(thread_idx_x) <- count]

(* [downto] with a bound the body increments: 4 iterations (j = 3..0) when read
   once, 2 when re-read. *)
let downto_mutating_bound =
  [%kernel
    fun (out : int32 vector) ->
      let open Sarek_stdlib.Std in
      let m = mut 0l in
      let count = mut 0l in
      for _j = 3 downto m do
        m := m + 1l ;
        count := count + 1l
      done ;
      out.(thread_idx_x) <- count]

(* The bound is an array element the body overwrites. Only thread 0 runs it,
   so the threads do not race on [bnd]; the others keep the sentinel. *)
let array_elem_bound =
  [%kernel
    fun (out : int32 vector) (bnd : int32 vector) ->
      let open Sarek_stdlib.Std in
      let count = mut 0l in
      if thread_idx_x = 0l then begin
        for _j = 0 to bnd.(0) do
          bnd.(0) <- 0l ;
          count := count + 1l
        done ;
        out.(0) <- count
      end]

(* The start has an effect the bound observes: it must run BEFORE the bound is
   read. Start = old value of c.(0) (0) and c.(0) becomes 2, so the bound is 2
   and the loop runs j = 0..2: 3 iterations. Reading the bound first gives
   bound 0 and 1 iteration. Only thread 0 runs it. *)
let effectful_start =
  [%kernel
    fun (out : int32 vector) (c : int32 vector) ->
      let open Sarek_stdlib.Std in
      let count = mut 0l in
      if thread_idx_x = 0l then begin
        for _j = atomic_add_global_int32 c 0 2l to c.(0) do
          count := count + 1l
        done ;
        out.(0) <- count
      end]

(* Block indices are signed too: block 0 gives (0 - 3) / 2 = -1. Two blocks. *)
let signed_block_idx =
  [%kernel
    fun (res : int32 vector) ->
      let open Sarek_stdlib.Std in
      res.(global_thread_id) <- (block_idx_x - 3l) / 2l]

(* A pure bound written with the kernel's own arithmetic: unchanged shape. *)
let stable_bound =
  [%kernel
    fun (out : int32 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let count = mut 0l in
      for _j = 0 to n - 1l do
        count := count + 1l
      done ;
      out.(thread_idx_x) <- count]

(* The thread intrinsic is used INLINE (a [let tid = thread_idx_x] would
   convert to signed and hide the bug). *)
let signed_thread_idx =
  [%kernel
    fun (neg : int32 vector) (quot : int32 vector) ->
      let open Sarek_stdlib.Std in
      neg.(thread_idx_x) <- (if thread_idx_x - 1l < 0l then 1l else 0l) ;
      quot.(thread_idx_x) <- (thread_idx_x - 5l) / 2l]

(* Derived indices: [global_thread_id] is also an int32. *)
let signed_global_idx =
  [%kernel
    fun (neg : int32 vector) (rem : int32 vector) ->
      let open Sarek_stdlib.Std in
      neg.(global_thread_id) <- (if global_thread_id - 1l < 0l then 1l else 0l) ;
      rem.(global_thread_id) <- (global_thread_id - 5l) / 2l]

let block = 64

let make_out () =
  let v = Vector.create Vector.int32 block in
  for i = 0 to block - 1 do
    Vector.set v i (-999l)
  done ;
  v

let make_zero () =
  let v = Vector.create Vector.int32 1 in
  Vector.set v 0 0l ;
  v

let make_bnd () =
  let v = Vector.create Vector.int32 1 in
  Vector.set v 0 3l ;
  v

type target = Interp | Hip of Device.t

let launch ?(grid = 1) target ir args =
  match target with
  | Interp ->
      Execute.run_interpreter_vectors
        ~ir
        ~args
        ~block:(Execute.dims1d block)
        ~grid:(Execute.dims1d grid)
        ~parallel:false
  | Hip dev ->
      Execute.run_vectors
        ~device:dev
        ~ir
        ~args
        ~block:(Execute.dims1d block)
        ~grid:(Execute.dims1d grid)
        () ;
      Transfer.flush dev

let name_of = function Interp -> "interp" | Hip _ -> "hip"

let check_vec label target got expected =
  let bad = ref 0 and first = ref None in
  Array.iteri
    (fun i g ->
      if g <> expected i then begin
        incr bad ;
        if !first = None then first := Some (i, g, expected i)
      end)
    got ;
  match !first with
  | None -> true
  | Some (i, g, e) ->
      Printf.printf
        "    FAIL %s on %s: %d mismatches, first at [%d]: got %ld expected %ld\n"
        label
        (name_of target)
        !bad
        i
        g
        e ;
      false

let run_for_bounds target =
  let one_at label kernel extra expected =
    let out = make_out () in
    launch target (ir_of kernel) (Execute.Vec out :: extra ()) ;
    check_vec label target (Vector.to_array out) expected
  in
  let one label kernel extra expected =
    one_at label kernel extra (fun _ -> expected)
  in
  let a = one "upto mutating bound" upto_mutating_bound (fun () -> []) 4l in
  let b = one "downto mutating bound" downto_mutating_bound (fun () -> []) 4l in
  let c =
    one_at
      "array element bound"
      array_elem_bound
      (fun () -> [Execute.Vec (make_bnd ())])
      (fun i -> if i = 0 then 4l else -999l)
  in
  let d = one "stable bound" stable_bound (fun () -> [Execute.Int 5]) 5l in
  let e =
    one_at
      "effectful start before bound"
      effectful_start
      (fun () -> [Execute.Vec (make_zero ())])
      (fun i -> if i = 0 then 3l else -999l)
  in
  a && b && c && d && e

let run_signed target =
  let neg = make_out () and quot = make_out () in
  launch target (ir_of signed_thread_idx) [Vec neg; Vec quot] ;
  let ok1 =
    check_vec "thread_idx_x - 1 < 0" target (Vector.to_array neg) (fun i ->
        if i - 1 < 0 then 1l else 0l)
  in
  let ok2 =
    check_vec "(thread_idx_x - 5) / 2" target (Vector.to_array quot) (fun i ->
        Int32.of_int ((i - 5) / 2))
  in
  let neg = make_out () and rem = make_out () in
  launch target (ir_of signed_global_idx) [Vec neg; Vec rem] ;
  let ok3 =
    check_vec "global_thread_id - 1 < 0" target (Vector.to_array neg) (fun i ->
        if i - 1 < 0 then 1l else 0l)
  in
  let ok4 =
    check_vec
      "(global_thread_id - 5) / 2"
      target
      (Vector.to_array rem)
      (fun i -> Int32.of_int ((i - 5) / 2))
  in
  let blocks = 2 in
  let res = Vector.create Vector.int32 (blocks * block) in
  launch ~grid:blocks target (ir_of signed_block_idx) [Vec res] ;
  let ok5 =
    check_vec "(block_idx_x - 3) / 2" target (Vector.to_array res) (fun i ->
        Int32.of_int (((i / block) - 3) / 2))
  in
  ok1 && ok2 && ok3 && ok4 && ok5

let guarded label f =
  try f ()
  with e ->
    Printf.printf "    %s EXN: %s\n" label (Printexc.to_string e) ;
    false

let run_target target =
  let label = name_of target in
  let bounds = guarded (label ^ " bounds") (fun () -> run_for_bounds target) in
  let signed = guarded (label ^ " signed") (fun () -> run_signed target) in
  Printf.printf
    "    %-6s for-bound once (C-8)             : %s\n"
    label
    (if bounds then "OK" else "FAIL") ;
  Printf.printf
    "    %-6s signed thread intrinsics (C-9)   : %s\n"
    label
    (if signed then "OK" else "FAIL") ;
  bounds && signed

let () =
  print_endline "test_hip_cfamily_semantics (C-8 for bound, C-9 signed ids)" ;
  let interp_ok = run_target Interp in
  let devices = Device.init () in
  let hip =
    Array.to_list devices |> List.filter (fun d -> d.Device.framework = "HIP")
  in
  let hip_ok =
    match hip with
    | [] ->
        print_endline "    HIP    : [SKIP] no HIP device" ;
        true
    | dev :: _ ->
        Printf.printf "    HIP device: %s\n" dev.Device.name ;
        run_target (Hip dev)
  in
  if interp_ok && hip_ok then print_endline "test_hip_cfamily_semantics PASSED"
  else (
    print_endline "test_hip_cfamily_semantics FAILED" ;
    exit 1)
