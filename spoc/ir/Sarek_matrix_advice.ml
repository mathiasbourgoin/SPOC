(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Which matrix-unit configuration a recognised GEMM nest would use, on which
   backend, under which accuracy regime. The table is the one of
   specs/matrix-pattern-advice.md; the backend facts are those of
   specs/coopmat-hip-wmma.md. *)

open Sarek_ir_types
module P = Sarek_matrix_pattern
module C = Sarek_coopmat_types

type element_case = F16_inputs | F32_inputs | Int32_inputs | No_case

type lowering = Direct | Operand_rounding | Byte_planes of int

type advice = {
  case : element_case;
  config : C.config option;
  lowering : lowering option;
  regime : C.accuracy_regime option;
  accuracy : string;
  backends : (string * string) list;
}

let shape = {C.m = 16; n = 16; k = 16}

let config ~a ~acc =
  {
    C.cfg_shape = shape;
    cfg_a = a;
    cfg_b = a;
    cfg_c = acc;
    cfg_result = acc;
    cfg_saturating = false;
    cfg_scope = C.Subgroup;
  }

let element_case (g : P.gemm) =
  match (g.a.elt, g.b.elt, g.acc.var_type, g.c.elt) with
  | TFloat16, TFloat16, TFloat32, (TFloat32 | TFloat16) -> F16_inputs
  | TFloat32, TFloat32, TFloat32, TFloat32 -> F32_inputs
  | TInt32, TInt32, TInt32, TInt32 -> Int32_inputs
  | _ -> No_case

let others = ("Metal, OpenCL, WGSL, PTX, Native", "no matrix-unit lowering")

let float_backends =
  [
    ("HIP gfx11 (wave32)", "executes");
    ("CUDA sm_70+", "compile-only, launch refused (no device probe)");
    ("Vulkan", "none (the GLSL emitter lowers integer configurations only)");
    others;
  ]

let integer_backends =
  [
    ("HIP gfx11 (wave32)", "executes");
    ("CUDA sm_72+", "compile-only, launch refused (no device probe)");
    ("Vulkan", "when the device advertises this configuration");
    others;
  ]

(* Byte-plane products for a 32-bit result: the pairs (i, j) with i + j <= 3. *)
let byte_plane_products = 10

let advise (g : P.gemm) =
  let f16 = config ~a:C.Float16 ~acc:C.Float32 in
  match element_case g with
  | F16_inputs ->
      {
        case = F16_inputs;
        config = Some f16;
        lowering = Some Direct;
        regime = Some C.Relaxed_bounded;
        accuracy =
          "relaxed: the matrix unit chooses the order of the additions, so \
           each output is within gamma_K * sum|a*b| of the exact value but not \
           bit-identical (docs/design/f16-relaxed-accuracy.md section 5); a \
           lowering needs Sarek_coopmat.accept_relaxed_accumulation ~reason \
           before launch";
        backends = float_backends;
      }
  | F32_inputs ->
      {
        case = F32_inputs;
        config = Some f16;
        lowering = Some Operand_rounding;
        regime = Some C.Relaxed_bounded;
        accuracy =
          "relaxed beyond the accumulation order: no matrix unit takes float32 \
           operands, so they would be rounded to float16 (relative error up to \
           2^-11 each); no existing opt-in covers that deviation";
        backends = float_backends;
      }
  | Int32_inputs ->
      {
        case = Int32_inputs;
        config = Some (config ~a:C.Uint8 ~acc:C.Sint32);
        lowering = Some (Byte_planes byte_plane_products);
        regime = Some C.Strict;
        accuracy =
          "exact: int32 arithmetic is modulo 2^32, and so is the sum of the \
           byte-plane products, so the result is bit-identical; no opt-in";
        backends = integer_backends;
      }
  | No_case ->
      {
        case = No_case;
        config = None;
        lowering = None;
        regime = None;
        accuracy = "no matrix-unit configuration for these element types";
        backends = [];
      }

let has_matrix_unit adv = adv.config <> None

(* ---- Rendering --------------------------------------------------------- *)

let strip_prefix s =
  let prefix = "Sarek_stdlib.Gpu." in
  let plen = String.length prefix in
  let buf = Buffer.create (String.length s) in
  let i = ref 0 in
  while !i < String.length s do
    if !i + plen <= String.length s && String.sub s !i plen = prefix then
      i := !i + plen
    else (
      Buffer.add_char buf s.[!i] ;
      incr i)
  done ;
  Buffer.contents buf

let expr_text e = strip_prefix (Format.asprintf "%a" Sarek_ir_pp.pp_expr e)

let operand_text name (o : P.operand) =
  let ld =
    match o.layout with
    | P.Row_major -> Printf.sprintf ", ld %s" (expr_text o.row_stride)
    | P.Col_major -> Printf.sprintf ", ld %s" (expr_text o.col_stride)
    | P.Strided ->
        Printf.sprintf
          ", strides %s x %s"
          (expr_text o.row_stride)
          (expr_text o.col_stride)
  in
  Printf.sprintf
    "%s = %s[%s] (%s%s)"
    name
    o.array
    (expr_text o.index)
    (P.layout_name o.layout)
    ld

let lowering_text = function
  | Direct -> "direct"
  | Operand_rounding -> "operands rounded to float16"
  | Byte_planes n ->
      Printf.sprintf
        "on the byte planes of each int32 operand, %d products per tile"
        n

let render ?kernel_name (g : P.gemm) adv =
  let name =
    match kernel_name with
    | Some n -> Printf.sprintf "kernel '%s'" n
    | None -> "this kernel"
  in
  let lines =
    [
      Printf.sprintf
        "Sarek: %s contains a matrix product (GEMM) that matrix units could \
         compute."
        name;
      Printf.sprintf
        "  C[m,n] = sum over %s = %s..%s of A[m,%s] * B[%s,n]"
        g.k_var.var_name
        (expr_text g.k_lo)
        (expr_text g.k_hi)
        g.k_var.var_name
        g.k_var.var_name;
      "  " ^ operand_text "A" g.a;
      "  " ^ operand_text "B" g.b;
      "  " ^ operand_text "C" g.c;
      Printf.sprintf
        "  m = %s, n = %s%s"
        (expr_text g.m_index)
        (expr_text g.n_index)
        (match g.batch with
        | [] -> ""
        | bs -> ", batched over " ^ String.concat ", " (List.map expr_text bs));
    ]
    @ (match (adv.config, adv.lowering) with
      | Some cfg, Some low ->
          [
            Printf.sprintf
              "  configuration: %s (%s)"
              (C.config_name cfg)
              (lowering_text low);
          ]
      | _ -> [])
    @ List.map (fun (b, s) -> Printf.sprintf "  %s: %s" b s) adv.backends
    @ [
        "  accuracy: " ^ adv.accuracy;
        "  No code was changed. Silence with [@alert \"-sarek_matrix\"] on the \
         kernel expression or its binding.";
      ]
  in
  String.concat "\n" lines
