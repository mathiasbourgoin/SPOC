(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The budget of a fused kernel is the meet over its members
 * (specs/optimisation-policy.md, M2), read by the narrowing decision of the
 * CUDA-C generator: a horizontally or epilogue-fused kernel whose members all
 * hold a Kernel-scoped Single_rounding grant is generated without the f32 ->
 * f16 barrier; one member lacking it keeps the barriers. CPU only: the
 * generated source is inspected, nothing is compiled.
 ******************************************************************************)

open Sarek_ir_types
module FH = Sarek_execute.Fusion_horizontal
module FE = Sarek_execute.Fusion_epilogue

let var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let vec name elt =
  DParam (var name (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})

let tid = var "tid" TInt32

(* out.(tid) <- f16(widen h.(tid) * 1.1): an inexact product, so the narrowing
   keeps its barrier unless single rounding is granted. *)
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

let single = Sarek_policy.Deviation Sarek_policy.Single_rounding

let grant_kernel name =
  match Sarek_policy.grant (Sarek_policy.Kernel name) single ~reason:"test" with
  | Ok () -> ()
  | Error _ -> Alcotest.fail "grant refused"

(* The fusion passes register the members of the kernel they build. *)
let horizontal name members =
  let fused = FH.fuse ~name (List.map kernel members) in
  Sarek_policy.register_fused ~name ~members ;
  fused

let test_horizontal () =
  grant_kernel "pf_a" ;
  grant_kernel "pf_b" ;
  Alcotest.(check bool)
    "all members granted: no barrier"
    false
    (contains barrier_call (gen (horizontal "pf_h_ab" ["pf_a"; "pf_b"]))) ;
  Alcotest.(check bool)
    "one member without the grant: barriers"
    true
    (contains barrier_call (gen (horizontal "pf_h_ac" ["pf_a"; "pf_c"]))) ;
  Alcotest.(check bool)
    "the fused name's own grant is not a member's"
    true
    (grant_kernel "pf_h_ac" ;
     contains barrier_call (gen (horizontal "pf_h_ac" ["pf_a"; "pf_c"])))

let test_nested () =
  let fused_ab = horizontal "pf_n_ab" ["pf_a"; "pf_b"] in
  ignore fused_ab ;
  Sarek_policy.register_fused ~name:"pf_n_top" ~members:["pf_n_ab"; "pf_a"] ;
  Alcotest.(check bool)
    "nested fused kernel, all granted"
    false
    (contains barrier_call (gen {(kernel "pf_x") with kern_name = "pf_n_top"})) ;
  Sarek_policy.register_fused ~name:"pf_n_bad" ~members:["pf_n_ab"; "pf_c"] ;
  Alcotest.(check bool)
    "nested fused kernel, one member lacks it"
    true
    (contains barrier_call (gen {(kernel "pf_x") with kern_name = "pf_n_bad"}))

let test_late_grant () =
  let src = gen (horizontal "pf_l_dc" ["pf_d"; "pf_c"]) in
  Alcotest.(check bool) "before" true (contains barrier_call src) ;
  grant_kernel "pf_d" ;
  grant_kernel "pf_c" ;
  Alcotest.(check bool)
    "after the members are granted"
    false
    (contains barrier_call (gen (horizontal "pf_l_dc" ["pf_d"; "pf_c"])))

let () =
  let case name f = Alcotest.test_case name `Quick f in
  Alcotest.run
    "policy_fused"
    [
      ( "meet over members (in order)",
        [
          case "horizontal" test_horizontal;
          case "nested" test_nested;
          case "late grant" test_late_grant;
        ] );
    ]
