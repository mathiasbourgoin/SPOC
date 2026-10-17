(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The kernel shape the shuffle was asked for (docs/plans/warp-shuffle-intake-
 * 2026-10-17.md, AC-3): an 8-lane xor-tree sum of float32 values, one column
 * per group of 8 consecutive lanes, masks 4, 2, 1.
 *
 * The reference is the same tree through shared memory and block barriers (the
 * code a shuffle replaces). Both add the same operands in the same order, so
 * the results are EXACTLY equal: the test compares bits, not a tolerance. Both
 * are also compared with the host sum of the 8 values in the tree's order.
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () =
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init ()

let block_threads = 64

let shuffle_kernel =
  [%kernel
    fun (inp : float32 vector) (out : float32 vector) ->
      let open Std in
      let g = (block_idx_x * 64l) + thread_idx_x in
      let s0 = inp.(g) in
      let s1 = s0 +. warp_shuffle_xor s0 4l in
      let s2 = s1 +. warp_shuffle_xor s1 2l in
      let s3 = s2 +. warp_shuffle_xor s2 1l in
      out.(g) <- s3]

let shared_kernel =
  [%kernel
    fun (inp : float32 vector) (out : float32 vector) ->
      let open Std in
      let%shared (sh : float32) = 64l in
      let t = thread_idx_x in
      let g = (block_idx_x * 64l) + t in
      sh.(t) <- inp.(g) ;
      block_barrier () ;
      let a = sh.(t) +. sh.(t lxor 4l) in
      block_barrier () ;
      sh.(t) <- a ;
      block_barrier () ;
      let b = sh.(t) +. sh.(t lxor 2l) in
      block_barrier () ;
      sh.(t) <- b ;
      block_barrier () ;
      let c = sh.(t) +. sh.(t lxor 1l) in
      out.(g) <- c]

let ir_of (_, kirc) =
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let blocks = 4

let n = blocks * block_threads

(* Values with different exponents so the addition order is visible in the
   bits: a reordered tree would not be equal. *)
let value i = (float_of_int (i * 37 mod 101) *. 0.0625) +. 1e-3

let run dev ir =
  let inp = Vector.create Vector.float32 n in
  let out = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set inp i (value i) ;
    Vector.set out i (-1.0)
  done ;
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec inp; Execute.Vec out]
    ~block:(Execute.dims1d block_threads)
    ~grid:(Execute.dims1d blocks)
    () ;
  Transfer.flush dev ;
  Array.init n (fun i -> Vector.get out i)

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* The host tree: lane t adds t xor 4, then xor 2, then xor 1, in float32. *)
let host_reference () =
  let vals = Array.init n (fun i -> f32 (value i)) in
  let step mask a = Array.mapi (fun i v -> f32 (v +. a.(i lxor mask))) a in
  step 1 (step 2 (step 4 vals))

let bits x = Int32.bits_of_float x

let failed = ref false

let host_ran = ref false

let report dev label outcome =
  Printf.printf
    "  %-11s %-34s %-8s %s\n%!"
    dev.Device.framework
    dev.Device.name
    label
    outcome

let compare_bits dev label expected got =
  let bad = ref 0 in
  Array.iteri (fun i e -> if bits e <> bits got.(i) then incr bad) expected ;
  if !bad > 0 then failed := true ;
  report
    dev
    label
    (if !bad = 0 then "PASS" else Printf.sprintf "FAIL (%d)" !bad)

let is_host dev =
  dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"

let run_device dev =
  let reference = host_reference () in
  let attempt label ir expected =
    match run dev ir with
    | got ->
        if is_host dev then host_ran := true ;
        compare_bits dev label expected got ;
        Some got
    | exception e ->
        let msg = Printexc.to_string e in
        let refused =
          let needle = "warp-shuffle unavailable" in
          let n = String.length needle and m = String.length msg in
          let rec go i =
            i + n <= m && (String.sub msg i n = needle || go (i + 1))
          in
          go 0
        in
        let regression =
          (not refused)
          && (is_host dev || Test_helpers.Backend_failure.verdict e = `Fail)
        in
        if regression then failed := true ;
        report
          dev
          label
          (if refused then "REFUSED (capability model)"
           else if regression then "FAIL (" ^ msg ^ ")"
           else "SKIP (" ^ msg ^ ")") ;
        None
  in
  let shuffled = attempt "shuffle" (ir_of shuffle_kernel) reference in
  let shared = attempt "shared" (ir_of shared_kernel) reference in
  match (shuffled, shared) with
  | Some a, Some b -> compare_bits dev "equal" a b
  | _ -> ()

let () =
  Test_helpers.Backend_failure.self_check ~test_name:"test_warp_shuffle_reduce" ;
  print_endline "=== 8-lane xor-tree sum: shuffle vs shared memory ===" ;
  Array.iter run_device (Device.init ()) ;
  if not !host_ran then begin
    print_endline
      "test_warp_shuffle_reduce: FAILED - no native/interpreter device ran" ;
    exit 1
  end ;
  if !failed then begin
    print_endline "test_warp_shuffle_reduce: FAILED" ;
    exit 1
  end ;
  print_endline "test_warp_shuffle_reduce: PASSED"
