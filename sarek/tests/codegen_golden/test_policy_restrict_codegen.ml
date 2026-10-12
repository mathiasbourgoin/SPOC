(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Process restrictions of the code generators (specs/optimisation-policy.md,
 * M2): [Sarek_policy.restrict Narrowing_proof] keeps every f32 -> f16 barrier
 * even where one rounding is proved to equal two, and
 * [Sarek_policy.restrict Auto_unroll] stops the automatic unroll hints.
 *
 * A restriction is process-wide and one-way, so this is its own executable and
 * the cases run in order: first the defaults, then the restrictions.
 ******************************************************************************)

open Sarek_ir_types

let var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let vec name elt =
  DParam (var name (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})

let tid = var "tid" TInt32

let i32 n = EConst (CInt32 (Int32.of_int n))

(* out.(tid) <- f16(widen h.(tid) + widen h.(tid)): the sum of two binary16
   values, for which one rounding provably equals two. *)
let provable_kernel =
  {
    default_kernel with
    kern_name = "restrict_provable";
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
                    ( Add,
                      ECast (TFloat32, EArrayRead ("h", EVar tid)),
                      ECast (TFloat32, EArrayRead ("h", EVar tid)) ) ) ) );
  }

(* A constant-trip loop: the unroll rule hints it. *)
let loop_kernel =
  {
    default_kernel with
    kern_name = "restrict_loop";
    kern_params = [vec "out" TInt32];
    kern_body =
      SFor
        ( var "i" TInt32,
          i32 0,
          i32 3,
          Upto,
          SAssign (LArrayElem ("out", EVar (var "i" TInt32)), i32 1) );
  }

let gen k = Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:[] k

let contains needle hay =
  let nl = String.length needle and hl = String.length hay in
  let rec go i = i + nl <= hl && (String.sub hay i nl = needle || go (i + 1)) in
  go 0

let barrier_call = "__float2half(sarek_f32_barrier("

let test_defaults () =
  Alcotest.(check bool)
    "a proved narrowing has no barrier"
    false
    (contains barrier_call (gen provable_kernel)) ;
  Alcotest.(check bool)
    "a constant-trip loop is hinted"
    true
    (contains "#pragma unroll" (gen loop_kernel))

let test_narrowing_restricted () =
  Sarek_policy.restrict Sarek_policy.Narrowing_proof ;
  Alcotest.(check bool)
    "the proof is not used: the barrier stays"
    true
    (contains barrier_call (gen provable_kernel))

let test_narrowing_restriction_leaves_unroll () =
  Alcotest.(check bool)
    "unroll is another optimisation"
    true
    (contains "#pragma unroll" (gen loop_kernel))

let test_unroll_restricted () =
  Sarek_policy.restrict Sarek_policy.Auto_unroll ;
  Alcotest.(check bool)
    "no unroll hint"
    false
    (contains "#pragma unroll" (gen loop_kernel)) ;
  Alcotest.(check bool)
    "the annotate pass is the identity"
    true
    (Sarek_codegen.Sarek_ir_unroll.annotate loop_kernel = loop_kernel)

let () =
  let case name f = Alcotest.test_case name `Quick f in
  Alcotest.run
    "policy_restrict_codegen"
    [
      ( "restrictions (in order)",
        [
          case "defaults" test_defaults;
          case "narrowing proof restricted" test_narrowing_restricted;
          case
            "unroll unaffected by the narrowing restriction"
            test_narrowing_restriction_leaves_unroll;
          case "unroll restricted" test_unroll_restricted;
        ] );
    ]
