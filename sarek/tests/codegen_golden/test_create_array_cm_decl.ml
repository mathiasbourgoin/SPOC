(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Package B / PR B (C-12, C-14): codegen assertions. COMPILE-ONLY: no device
 * is created, only the pure source generators run (glslang / naga validation
 * of the same text is recorded in the PR log, not run here).
 *
 *   1. WGSL lowers a constant-size [create_array n Local] to a function-scope
 *      [var a : array<T, N>;] (C-14, FR-B3). It used to raise an internal
 *      "local dynamic arrays unsupported" error.
 *   2. A declared cooperative-matrix fragment is ZERO on every backend that
 *      emits cooperative matrices (C-12, FR-B4): CUDA [fill_fragment(.., 0)],
 *      HIP [= {}], GLSL [coopmat<..>(0)]. Before the fix GLSL emitted a bare
 *      declaration with undefined contents.
 *   3. The backends that do not emit cooperative matrices (OpenCL, Metal,
 *      WGSL, PTX) refuse the declaration rather than emit it uninitialised.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen
module C = Sarek_coopmat_types

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

let check_contains name ~needle out =
  if not (contains ~needle out) then
    Alcotest.failf "%s: expected %S in the output:\n%s" name needle out

let check_absent name ~needle out =
  if contains ~needle out then
    Alcotest.failf "%s: unexpected %S in the output:\n%s" name needle out

(* ------------------------------------------------------------------ *)
(* 1. WGSL constant-size Local array                                  *)
(* ------------------------------------------------------------------ *)

let local_array_kernel () =
  let out = make_var "out" (TVec TInt32) in
  let arr = make_var "scratch" (TArray (TInt32, Local)) in
  {
    default_kernel with
    kern_name = "local_array_probe";
    kern_params = [DParam (out, None)];
    kern_body =
      SLet
        ( arr,
          EArrayCreate (TInt32, EConst (CInt32 16l), Local),
          SSeq
            [
              SAssign
                (LArrayElem ("scratch", EConst (CInt32 0l)), EConst (CInt32 7l));
              SAssign
                ( LArrayElem ("out", EConst (CInt32 0l)),
                  EArrayRead ("scratch", EConst (CInt32 0l)) );
            ] );
  }

let test_wgsl_local_array () =
  let src =
    try Sarek_ir_wgsl.generate_with_types ~types:[] (local_array_kernel ())
    with e ->
      Alcotest.failf
        "WGSL refused a constant-size Local array: %s"
        (Printexc.to_string e)
  in
  check_contains "WGSL local" ~needle:"var scratch : array<i32, 16>;" src ;
  check_absent "WGSL local" ~needle:"var<workgroup> scratch" src

(* ------------------------------------------------------------------ *)
(* 2. CM_decl is zero on every emitting backend                       *)
(* ------------------------------------------------------------------ *)

let frag =
  {
    C.frag_use = C.Accumulator;
    frag_shape = {C.m = 16; n = 16; k = 16};
    frag_component = C.Sint32;
    frag_scope = C.Subgroup;
  }

let decl_kernel () =
  let out = make_var "out" (TVec TInt32) in
  {
    default_kernel with
    kern_name = "cm_decl_probe";
    kern_params = [DParam (out, None)];
    kern_body = SCoopmat (CM_decl {name = "facc"; frag});
  }

let test_cm_decl_zero () =
  let k = decl_kernel () in
  let cuda = Sarek_ir_cuda.generate_with_types ~types:[] k in
  check_contains "CUDA" ~needle:"fill_fragment(facc, 0)" cuda ;
  check_contains "HIP" ~needle:"facc = {};" cuda ;
  let glsl = Sarek_ir_glsl.generate_with_types ~types:[] k in
  check_contains "GLSL" ~needle:"facc = coopmat<" glsl ;
  check_contains "GLSL" ~needle:">(0);" glsl

(* ------------------------------------------------------------------ *)
(* 3. Backends without cooperative matrices refuse the declaration    *)
(* ------------------------------------------------------------------ *)

let expect_refused name gen =
  match gen (decl_kernel ()) with
  | (_ : string) ->
      Alcotest.failf "%s emitted a CM_decl instead of refusing it" name
  | exception Sarek_backend_error.Backend_error.Backend_error _ -> ()
  | exception Sarek_ir_ptx_types.Ptx_codegen_error _ -> ()

let test_refusals () =
  expect_refused "OpenCL" (Sarek_ir_opencl.generate_with_types ~types:[]) ;
  expect_refused "Metal" (Sarek_ir_metal.generate_with_types ~types:[]) ;
  expect_refused "WGSL" (Sarek_ir_wgsl.generate_with_types ~types:[]) ;
  expect_refused "PTX" (fun k -> Sarek_ir_ptx.generate_with_types ~types:[] k)

let () =
  Alcotest.run
    "create-array-cm-decl"
    [
      ( "package-B-C12-C14",
        [
          Alcotest.test_case
            "WGSL constant Local array"
            `Quick
            test_wgsl_local_array;
          Alcotest.test_case
            "CM_decl zero on CUDA/HIP/GLSL"
            `Quick
            test_cm_decl_zero;
          Alcotest.test_case "CM_decl refused elsewhere" `Quick test_refusals;
        ] );
    ]
