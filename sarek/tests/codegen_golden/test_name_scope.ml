(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Which binder a reference reaches after Sarek_ir_names renamed (roster review
 * of the renaming pass).
 *
 * A [SLet] has no braces on the C family, so a binder lives to the end of its
 * statement list; the pass therefore renames a later binder that has the same
 * name as an earlier one. A reference to the EARLIER binder (same name, other
 * [var_id]) that comes after it must keep the earlier spelling: matching by
 * name alone redirected it to the renamed later binder, which turned a kernel
 * the compiler rejected into one that compiles and reads the wrong variable.
 * The PPX produces the shape for any module-level constant that shares its
 * name with a kernel parameter.
 *
 * Also: a [let] in a helper that rebinds a helper parameter, on every target.
 ******************************************************************************)

open Sarek_ir_types
module Codegen = Sarek_codegen

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  m = 0 || go 0

let var name id ty =
  {var_name = name; var_id = id; var_type = ty; var_mutable = false}

let f x = EConst (CFloat32 x)

let c_family =
  [
    ("CUDA", Codegen.Sarek_ir_cuda.generate);
    ("OpenCL", Codegen.Sarek_ir_opencl.generate);
    ("Metal", Codegen.Sarek_ir_metal.generate);
  ]

let all_backends =
  c_family
  @ [
      ("GLSL", fun k -> Codegen.Sarek_ir_glsl.generate k);
      ("WGSL", fun k -> Codegen.Sarek_ir_wgsl.generate k);
    ]

let vec_param name id =
  DParam
    ( var name id (TVec TFloat32),
      Some {arr_elttype = TFloat32; arr_memspace = Global} )

let out_zero value = SAssign (LArrayElem ("a", EConst (CInt32 0l)), value)

(* a.(0) <- scale, where the parameter [scale] (id 2) comes after a constant
   also called [scale] (id 1) bound by an earlier sibling. *)
let module_constant_kernel =
  let constant = var "scale" 1 TFloat32 and param = var "scale" 2 TFloat32 in
  {
    default_kernel with
    kern_name = "module_constant";
    kern_params = [vec_param "a" 0; DParam (param, None)];
    kern_body =
      SSeq [SSeq [SLet (constant, f 2.0, SEmpty)]; out_zero (EVar param)];
  }

let test_later_sibling_keeps_outer_binder () =
  List.iter
    (fun (name, gen) ->
      let src = gen module_constant_kernel in
      Alcotest.(check bool)
        (name ^ ": the constant is renamed")
        true
        (contains src "scale_1 = 2.0f") ;
      Alcotest.(check bool)
        (name ^ ": the later reference still reads the parameter")
        false
        (contains src "= scale_1;") ;
      Alcotest.(check bool)
        (name ^ ": it reads the parameter by its own name")
        true
        (contains src "] = scale;"))
    c_family

(* The binder that is renamed is the later one only when the earlier one is
   still visible; the reverse order must read each binder once. *)
let test_each_reference_follows_its_binder () =
  let first = var "t" 1 TFloat32 and second = var "t" 2 TFloat32 in
  let k =
    {
      default_kernel with
      kern_name = "two_t";
      kern_params = [vec_param "a" 0];
      kern_body =
        SLet
          ( first,
            f 1.0,
            SSeq
              [
                SLet (second, f 2.0, out_zero (EVar second));
                out_zero (EVar first);
              ] );
    }
  in
  List.iter
    (fun (name, gen) ->
      let src = gen k in
      Alcotest.(check bool)
        (name ^ ": the second t is renamed")
        true
        (contains src "t_1 = 2.0f") ;
      Alcotest.(check bool)
        (name ^ ": the last assignment reads the first t")
        true
        (contains src "] = t;"))
    c_family

(* An array reference carries a name and no identity: it reaches the array of
   that name, never a scalar [let] of the same name that was renamed. *)
let test_array_reference_ignores_scalar () =
  let scalar = var "a" 1 TFloat32 in
  let k =
    {
      default_kernel with
      kern_name = "array_vs_scalar";
      kern_params = [vec_param "a" 0];
      kern_body = SSeq [SSeq [SLet (scalar, f 2.0, SEmpty)]; out_zero (f 1.0)];
    }
  in
  List.iter
    (fun (name, gen) ->
      let src = gen k in
      Alcotest.(check bool)
        (name ^ ": the scalar is renamed")
        true
        (contains src "a_1 = 2.0f") ;
      Alcotest.(check bool)
        (name ^ ": the store reaches the array")
        true
        (contains src "a[0] = ") ;
      Alcotest.(check bool)
        (name ^ ": the store does not reach the scalar")
        false
        (contains src "a_1[0]"))
    c_family

(* A helper that rebinds its own parameter: a redeclaration on every target. *)
let helper_rebinds_parameter =
  let x_param = var "x" 30 TFloat32 and x_local = var "x" 31 TFloat32 in
  let helper =
    {
      hf_name = "bump";
      hf_params = [x_param];
      hf_ret_type = TFloat32;
      hf_body =
        SLet (x_local, EBinop (Add, EVar x_param, f 1.0), SReturn (EVar x_local));
    }
  in
  {
    default_kernel with
    kern_name = "helper_rebind";
    kern_params = [vec_param "a" 0];
    kern_funcs = [helper];
    kern_body = out_zero (EApp (EVar (var "bump" 0 TFloat32), [f 1.0]));
  }

let test_helper_parameter_rebound () =
  List.iter
    (fun (name, gen) ->
      let src = gen helper_rebinds_parameter in
      Alcotest.(check bool)
        (name ^ ": the helper local is renamed")
        true
        (contains src "x_1"))
    all_backends

(* The renamed sources are accepted by the real compilers when present. *)
let test_compilers_accept () =
  let kernels = [module_constant_kernel; helper_rebinds_parameter] in
  if Opencl_gate.Opencl_clang.available () then
    List.iter
      (fun k ->
        match
          Opencl_gate.Opencl_clang.run_clang
            (Codegen.Sarek_ir_opencl.generate k)
        with
        | Ok () -> ()
        | Error e -> Alcotest.failf "%s rejected by clang:\n%s" k.kern_name e)
      kernels ;
  if Metal_gate.Metal_compile.available () then
    List.iter
      (fun k ->
        match
          Metal_gate.Metal_compile.run_metal (Codegen.Sarek_ir_metal.generate k)
        with
        | Ok () -> ()
        | Error e -> Alcotest.failf "%s rejected by Metal:\n%s" k.kern_name e)
      kernels

let () =
  Alcotest.run
    "name scope"
    [
      ( "scope",
        [
          Alcotest.test_case
            "a later sibling keeps the outer binder"
            `Quick
            test_later_sibling_keeps_outer_binder;
          Alcotest.test_case
            "each reference follows its binder"
            `Quick
            test_each_reference_follows_its_binder;
          Alcotest.test_case
            "an array reference ignores a scalar of its name"
            `Quick
            test_array_reference_ignores_scalar;
          Alcotest.test_case
            "helper parameter rebound, every backend"
            `Quick
            test_helper_parameter_rebound;
          Alcotest.test_case
            "real compilers accept the renamed sources"
            `Quick
            test_compilers_accept;
        ] );
    ]
