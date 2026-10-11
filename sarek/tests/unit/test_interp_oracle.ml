(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The interpreter is the oracle every backend is checked against, so three
    ways it used to be wrong are pinned here, each against an exact expected
    value:

    - C-10 / R-62: an in-place cooperative-matrix accumulation [D = A*B + D] ran
      once per invocation and accumulated once per lane;
    - R-63: two kernel arguments bound to the SAME vector got two private
      copies, so a write through one was invisible to a read through the other;
    - C-18: the BSP deadlock detector could never fire. *)

open Sarek_ir_types
open Alcotest
module Interp = Sarek.Sarek_ir_interp
module Execute = Sarek.Execute
module Vector = Spoc_core.Vector
module Interp_error = Sarek_interp.Interp_error

let i32 n = EConst (CInt32 (Int32.of_int n))

let var id name ty =
  {var_name = name; var_id = id; var_type = ty; var_mutable = false}

let tid = EIntrinsic ([], "thread_idx_x", [])

let vec_param v elt = DParam (v, Some {arr_elttype = elt; arr_memspace = Global})

(** {1 C-10 / R-62: in-place cooperative-matrix accumulation} *)

let mnk = 16

let tile = mnk * mnk

let cfg =
  {
    Sarek_coopmat.cfg_shape = {Sarek_coopmat.m = mnk; n = mnk; k = mnk};
    cfg_a = Sarek_coopmat.Uint8;
    cfg_b = Sarek_coopmat.Uint8;
    cfg_c = Sarek_coopmat.Sint32;
    cfg_result = Sarek_coopmat.Sint32;
    cfg_saturating = false;
    cfg_scope = Sarek_coopmat.Subgroup;
  }

let frag_a, frag_b, frag_c, _ = Sarek_coopmat.fragments_of_config cfg

(** [d = a * b + d] on one 16x16x16 tile PER SUBGROUP (tile
    [thread_idx_x / 32]), the accumulator loaded from and stored to the SAME
    array [d]. Each subgroup owns its own tile, so the kernel is race-free on a
    device. *)
let in_place_ir ~subgroups_per_block =
  let a = var 0 "a" (TVec TUint8)
  and b = var 1 "b" (TVec TUint8)
  and d = var 2 "d" (TVec TInt32) in
  let subgroup =
    EBinop
      ( Add,
        EBinop (Mul, EIntrinsic ([], "block_idx_x", []), i32 subgroups_per_block),
        EBinop (Div, tid, i32 32) )
  in
  let at = EBinop (Mul, subgroup, i32 tile) in
  let cm op = SCoopmat op in
  let load name frag src =
    cm (CM_load {dst = name; frag; src; index = at; stride = i32 mnk})
  in
  {
    default_kernel with
    kern_name = "interp_oracle_coopmat_in_place";
    kern_params = [vec_param a TUint8; vec_param b TUint8; vec_param d TInt32];
    kern_body =
      SSeq
        [
          cm (CM_decl {name = "fa"; frag = frag_a});
          cm (CM_decl {name = "fb"; frag = frag_b});
          cm (CM_decl {name = "fc"; frag = frag_c});
          load "fa" frag_a "a";
          load "fb" frag_b "b";
          load "fc" frag_c "d";
          cm (CM_muladd {dst = "fc"; a = "fa"; b = "fb"; c = "fc"; cfg});
          cm
            (CM_store
               {
                 src = "fc";
                 frag = frag_c;
                 dst = "d";
                 index = at;
                 stride = i32 mnk;
               });
        ];
  }

let int_array n x = Array.make n (Interp.VInt32 x)

let d_values arr =
  Array.map
    (function Interp.VInt32 n -> Int32.to_int n | _ -> Alcotest.fail "d")
    arr

let check_all_equal name expected arr =
  check (array int) name (Array.make (Array.length arr) expected) (d_values arr)

let with_parallel_mode mode f =
  let saved = !Interp.parallel_mode in
  Interp.parallel_mode := mode ;
  Fun.protect ~finally:(fun () -> Interp.parallel_mode := saved) f

(* A = B = 1, so one tile adds exactly 16 to each element of D. Every subgroup
   of every block owns one tile. *)
let test_in_place ?(grid = 1) ?(parallel = true) ~block ~d0 () =
  let subgroups_per_block = max 1 ((block + 31) / 32) in
  let n = grid * subgroups_per_block * tile in
  let d = int_array n (Int32.of_int d0) in
  with_parallel_mode parallel (fun () ->
      Interp.run_kernel
        (in_place_ir ~subgroups_per_block)
        ~block:(block, 1, 1)
        ~grid:(grid, 1, 1)
        [
          ("a", Interp.ArgArray (int_array n 1l));
          ("b", Interp.ArgArray (int_array n 1l));
          ("d", Interp.ArgArray d);
        ]) ;
  check_all_equal
    (Printf.sprintf
       "block %d, grid %d, parallel %b, d0 %d: d = 16 + d0"
       block
       grid
       parallel
       d0)
    (16 + d0)
    d

(** [Sarek_gemm.Coopmat.ir] with [c] and [d] bound to the same array. *)
let test_gemm_c_is_d () =
  Sarek_coopmat.accept_relaxed_accumulation
    ~reason:"interpreter oracle test: exact small integers" ;
  let m = 64 and n = 32 and k = 16 in
  let f x = Interp.VFloat32 x in
  let cd = Array.make (m * n) (f 3.0) in
  Interp.run_kernel
    (Sarek_gemm.Coopmat.ir ())
    ~block:(64, 1, 1)
    ~grid:(1, 1, 1)
    [
      ("a", Interp.ArgArray (Array.make (m * k) (f 1.0)));
      ("b", Interp.ArgArray (Array.make (k * n) (f 1.0)));
      ("c", Interp.ArgArray cd);
      ("d", Interp.ArgArray cd);
      ("m", Interp.ArgScalar (Interp.VInt32 (Int32.of_int m)));
      ("n", Interp.ArgScalar (Interp.VInt32 (Int32.of_int n)));
      ("k", Interp.ArgScalar (Interp.VInt32 (Int32.of_int k)));
    ] ;
  let got =
    Array.map
      (function Interp.VFloat32 x -> x | _ -> Alcotest.fail "gemm d")
      cd
  in
  check (array (float 0.0)) "gemm c == d: 3 + 16" (Array.make (m * n) 19.0) got

(** {1 R-63: aliased vector arguments} *)

(** [x.(i) <- y.(i) + 7], with the written parameter first ([alias_ir]) or last
    ([alias_ir_written_last]): the entry points wrote their private copies back
    in different orders, so the aliased result depended on which parameter was
    written. *)
let alias_kernel ~name ~written_first =
  let x = var 0 "x" (TVec TInt32) and y = var 1 "y" (TVec TInt32) in
  let params = [vec_param x TInt32; vec_param y TInt32] in
  {
    default_kernel with
    kern_name = name;
    kern_params = (if written_first then params else List.rev params);
    kern_body =
      SAssign (LArrayElem ("x", tid), EBinop (Add, EArrayRead ("y", tid), i32 7));
  }

let alias_ir = alias_kernel ~name:"interp_oracle_alias" ~written_first:true

let alias_ir_written_last =
  alias_kernel ~name:"interp_oracle_alias_last" ~written_first:false

let ones () =
  let v = Vector.create Vector.int32 4 in
  for i = 0 to 3 do
    Vector.set v i 1l
  done ;
  v

let read4 v = List.init 4 (Vector.get v)

let test_alias_run_kernel_with_args ir () =
  let v = ones () in
  Interp.run_kernel_with_args
    ir
    ~block:(4, 1, 1)
    ~grid:(1, 1, 1)
    [Spoc_core.Kernel_arg.Vec v; Spoc_core.Kernel_arg.Vec v] ;
  check (list int32) "x == y: y + 7 = 8" [8l; 8l; 8l; 8l] (read4 v)

let test_alias_run_interpreter_vectors ir () =
  let v = ones () in
  Execute.run_interpreter_vectors
    ~ir
    ~args:[Execute.Vec v; Execute.Vec v]
    ~block:(Execute.dims1d 4)
    ~grid:(Execute.dims1d 1)
    ~parallel:false ;
  check (list int32) "x == y: y + 7 = 8" [8l; 8l; 8l; 8l] (read4 v)

let interpreter_device =
  lazy
    (Sarek_interpreter.Interpreter_plugin.init () ;
     match
       Array.find_opt
         (fun d -> d.Spoc_core.Device.framework = "Interpreter")
         (Spoc_core.Device.init ())
     with
     | Some d -> d
     | None -> Alcotest.fail "no Interpreter device")

(* The launch path of an Interpreter device: Execute.run_vectors, through the
   interpreter plugin and run_kernel_with_exec_args. *)
let test_alias_run_vectors_interpreter ir () =
  let v = ones () in
  Execute.run_vectors
    ~device:(Lazy.force interpreter_device)
    ~ir
    ~args:[Execute.Vec v; Execute.Vec v]
    ~block:(Execute.dims1d 4)
    ~grid:(Execute.dims1d 1)
    () ;
  check (list int32) "x == y: y + 7 = 8" [8l; 8l; 8l; 8l] (read4 v)

(* Distinct vectors must stay distinct: dedup is by identity, not by contents. *)
let test_distinct_vectors_stay_distinct () =
  let x = ones () and y = ones () in
  Execute.run_interpreter_vectors
    ~ir:alias_ir
    ~args:[Execute.Vec x; Execute.Vec y]
    ~block:(Execute.dims1d 4)
    ~grid:(Execute.dims1d 1)
    ~parallel:false ;
  check (list int32) "x written" [8l; 8l; 8l; 8l] (read4 x) ;
  check (list int32) "y untouched" [1l; 1l; 1l; 1l] (read4 y)

(** {1 C-18: the BSP deadlock detector} *)

let barrier_ir name body =
  {default_kernel with kern_name = name; kern_params = []; kern_body = body}

(* Three successive barriers on four threads: every thread re-suspends each
   round, which is progress, not a deadlock. *)
let test_successive_barriers_pass () =
  Interp.run_kernel
    (barrier_ir "interp_oracle_barriers" (SSeq [SBarrier; SBarrier; SBarrier]))
    ~block:(4, 1, 1)
    ~grid:(1, 1, 1)
    []

(* Threads 0 and 1 wait at a barrier threads 2 and 3 never reach. *)
let test_divergent_barrier_deadlocks () =
  let ir =
    barrier_ir
      "interp_oracle_divergent"
      (SIf (EBinop (Lt, tid, i32 2), SBarrier, None))
  in
  match Interp.run_kernel ir ~block:(4, 1, 1) ~grid:(1, 1, 1) [] with
  | () -> Alcotest.fail "a divergent barrier must be reported as a deadlock"
  | exception Interp_error.Interpreter_error (Interp_error.BSP_deadlock _) -> ()

(* Warp 0 reaches a warp barrier that warp 1 never does: legal, not a block-wide
   deadlock. *)
let test_warp_barrier_not_a_deadlock () =
  Interp.run_kernel
    (barrier_ir
       "interp_oracle_warp"
       (SIf (EBinop (Lt, tid, i32 32), SWarpBarrier, None)))
    ~block:(64, 1, 1)
    ~grid:(1, 1, 1)
    []

let () =
  run
    "Interp_oracle"
    [
      ( "coopmat_in_place",
        [
          test_case "block 1" `Quick (test_in_place ~block:1 ~d0:0);
          test_case "block 32" `Quick (test_in_place ~block:32 ~d0:0);
          test_case "block 64" `Quick (test_in_place ~block:64 ~d0:0);
          test_case "block 64, d0 = 5" `Quick (test_in_place ~block:64 ~d0:5);
          test_case
            "block 48 (partial subgroup)"
            `Quick
            (test_in_place ~block:48 ~d0:0);
          test_case
            "block 33 (partial subgroup)"
            `Quick
            (test_in_place ~block:33 ~d0:0);
          test_case
            "grid of 3 blocks of 64"
            `Quick
            (test_in_place ~grid:3 ~block:64 ~d0:0);
          test_case
            "sequential, block 64"
            `Quick
            (test_in_place ~parallel:false ~block:64 ~d0:0);
          test_case
            "sequential, grid of 3 blocks of 48"
            `Quick
            (test_in_place ~parallel:false ~grid:3 ~block:48 ~d0:5);
          test_case "Sarek_gemm.Coopmat c == d" `Quick test_gemm_c_is_d;
        ] );
      ( "aliased_arguments",
        [
          test_case
            "run_kernel_with_args, written first"
            `Quick
            (test_alias_run_kernel_with_args alias_ir);
          test_case
            "run_kernel_with_args, written last"
            `Quick
            (test_alias_run_kernel_with_args alias_ir_written_last);
          test_case
            "run_interpreter_vectors, written first"
            `Quick
            (test_alias_run_interpreter_vectors alias_ir);
          test_case
            "run_interpreter_vectors, written last"
            `Quick
            (test_alias_run_interpreter_vectors alias_ir_written_last);
          test_case
            "run_vectors on Interpreter, written first"
            `Quick
            (test_alias_run_vectors_interpreter alias_ir);
          test_case
            "run_vectors on Interpreter, written last"
            `Quick
            (test_alias_run_vectors_interpreter alias_ir_written_last);
          test_case
            "distinct vectors stay distinct"
            `Quick
            test_distinct_vectors_stay_distinct;
        ] );
      ( "bsp_deadlock",
        [
          test_case "successive barriers" `Quick test_successive_barriers_pass;
          test_case
            "warp barrier is not a deadlock"
            `Quick
            test_warp_barrier_not_a_deadlock;
          test_case "divergent barrier" `Quick test_divergent_barrier_deadlocks;
        ] );
    ]
