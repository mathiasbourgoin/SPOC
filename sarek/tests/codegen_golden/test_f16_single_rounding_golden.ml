(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_f16_policy.accept_single_rounding (specs/f16-narrowing-barrier.md §4).
 *
 * The opt-in is process-wide and one-way, so this file is its own executable
 * and the cases run in order: before any acceptance, after a [Kernel] scope,
 * after [All_kernels].
 ******************************************************************************)

open Sarek_ir_types

let var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let vec name elt =
  DParam (var name (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})

let tid = var "tid" TInt32

(* out.(tid) <- f16(widen h.(tid) * 1.1): the product is inexact, so without
   the opt-in this narrowing keeps its barrier. *)
let kernel name =
  {
    default_kernel with
    kern_name = name;
    kern_params = [vec "out" TFloat16; vec "h" TFloat16];
    kern_body =
      SLet
        ( tid,
          EIntrinsic ([], "global_thread_id", []),
          SAssign
            ( LArrayElem ("out", EVar tid),
              ECast
                ( TFloat16,
                  EBinop
                    ( Mul,
                      ECast (TFloat32, EArrayRead ("h", EVar tid)),
                      EConst (CFloat32 1.1) ) ) ) );
  }

let gen k = Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:[] k

let contains needle hay =
  let nl = String.length needle and hl = String.length hay in
  let rec go i = i + nl <= hl && (String.sub hay i nl = needle || go (i + 1)) in
  go 0

let barrier_call = "__float2half(sarek_f32_barrier("

let comment = "Sarek_f16_policy.accept_single_rounding covers this kernel"

let check_barrier what name expected =
  let src = gen (kernel name) in
  Alcotest.(check bool)
    (what ^ ": barrier")
    expected
    (contains barrier_call src) ;
  Alcotest.(check bool)
    (what ^ ": opt-in comment")
    (not expected)
    (contains comment src)

let test_default () =
  check_barrier "no opt-in" "optin_a" true ;
  Alcotest.(check (option string))
    "no reason"
    None
    (Sarek_f16_policy.single_rounding_reason ~kernel:"optin_a")

let test_empty_reason () =
  Alcotest.check_raises
    "blank reason"
    (Invalid_argument
       "Sarek_f16_policy.accept_single_rounding: the reason must not be empty \
        (it is the record that the deviation from the interpreter was \
        understood)")
    (fun () -> Sarek_f16_policy.accept_single_rounding All_kernels ~reason:" ")

let test_kernel_scope () =
  Sarek_f16_policy.accept_single_rounding (Kernel "optin_a") ~reason:"test a" ;
  Sarek_f16_policy.accept_single_rounding (Kernel "optin_a") ~reason:"second" ;
  check_barrier "accepted kernel" "optin_a" false ;
  check_barrier "other kernel" "optin_b" true ;
  Alcotest.(check (option string))
    "first reason kept"
    (Some "test a")
    (Sarek_f16_policy.single_rounding_reason ~kernel:"optin_a")

let test_all_kernels () =
  Sarek_f16_policy.accept_single_rounding All_kernels ~reason:"test all" ;
  check_barrier "any kernel" "optin_b" false ;
  Alcotest.(check (option string))
    "kernel scope takes precedence"
    (Some "test a")
    (Sarek_f16_policy.single_rounding_reason ~kernel:"optin_a") ;
  Alcotest.(check (option string))
    "all-kernels reason"
    (Some "test all")
    (Sarek_f16_policy.single_rounding_reason ~kernel:"optin_b")

let () =
  let case name f = Alcotest.test_case name `Quick f in
  Alcotest.run
    "f16_single_rounding"
    [
      ( "opt-in (in order)",
        [
          case "default keeps the barrier" test_default;
          case "blank reason refused" test_empty_reason;
          case "kernel scope" test_kernel_scope;
          case "all kernels" test_all_kernels;
        ] );
    ]
