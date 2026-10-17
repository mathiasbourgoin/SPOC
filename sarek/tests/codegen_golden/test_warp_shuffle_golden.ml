(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Golden source of the warp operations on every source backend
 * (docs/plans/warp-shuffle-intake-2026-10-17.md, AC-5).
 *
 * One kernel calls each of the four shuffles on int32, float32 and (where the
 * backend has float64) float64, plus warp_size and lane_id. [dune runtest]
 * compares the generated source of each backend with a checked-in
 * [warp_shuffle_<backend>.expected]. To regenerate after an intended change:
 *   dune exec sarek/tests/codegen_golden/test_warp_shuffle_golden.exe -- cuda \
 *     > sarek/tests/codegen_golden/warp_shuffle_cuda.expected
 * (likewise for the other backend names), and review the diff.
 ******************************************************************************)

open Sarek_ir_types
module Codegen = Sarek_codegen

let var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let param v elt = DParam (v, Some {arr_elttype = elt; arr_memspace = Global})

(* out.(i) <- shuffle in.(0) lane_id, for each operation and value type. *)
let kernel ~f64 =
  let ivec = var "src_i" (TVec TInt32) and fvec = var "src_f" (TVec TFloat32) in
  let dvec = var "src_d" (TVec TFloat64) in
  let oi = var "out_i" (TVec TInt32) and of_ = var "out_f" (TVec TFloat32) in
  let od = var "out_d" (TVec TFloat64) in
  let read name = EArrayRead (name, EConst (CInt32 0l)) in
  let lane = EIntrinsic ([], "lane_id", []) in
  let store name e = SAssign (LArrayElem (name, EConst (CInt32 0l)), e) in
  let call op ty name =
    store
      (match ty with
      | Sarek_warp.I32 -> "out_i"
      | F32 -> "out_f"
      | F64 -> "out_d")
      (EIntrinsic ([], Sarek_warp.ir_name op ty, [read name; lane]))
  in
  let body =
    List.concat_map
      (fun op ->
        [call op Sarek_warp.I32 "src_i"; call op Sarek_warp.F32 "src_f"]
        @ if f64 then [call op Sarek_warp.F64 "src_d"] else [])
      Sarek_warp.all_ops
    @ [store "out_i" (EIntrinsic ([], "warp_size", []))]
  in
  {
    default_kernel with
    kern_name = "warp_golden";
    kern_params =
      ([
         param ivec TInt32;
         param fvec TFloat32;
         param oi TInt32;
         param of_ TFloat32;
       ]
      @ if f64 then [param dvec TFloat64; param od TFloat64] else []);
    kern_body = SSeq body;
  }

let () =
  let backend = if Array.length Sys.argv > 1 then Sys.argv.(1) else "" in
  let source =
    match backend with
    | "cuda" -> Codegen.Sarek_ir_cuda.generate (kernel ~f64:true)
    | "opencl" -> Codegen.Sarek_ir_opencl.generate (kernel ~f64:true)
    | "glsl" -> Codegen.Sarek_ir_glsl.generate (kernel ~f64:true)
    | "metal" -> Codegen.Sarek_ir_metal.generate (kernel ~f64:false)
    | "wgsl" -> Codegen.Sarek_ir_wgsl.generate (kernel ~f64:false)
    | "ptx" -> Codegen.Sarek_ir_ptx.generate (kernel ~f64:true)
    | other ->
        prerr_endline ("unknown backend: " ^ other) ;
        exit 2
  in
  print_string source
