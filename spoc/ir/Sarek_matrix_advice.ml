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

type accuracy_class = Exact | Relaxed_accumulation | Relaxed_operands

type advice = {
  case : element_case;
  config : C.config option;
  lowering : lowering option;
  accuracy_class : accuracy_class option;
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

let others = ("OpenCL, WGSL, PTX, Native", "no matrix-unit lowering")

(* Metal's simdgroup_matrix has 8x8x8 half/float fragments and no integer
   element: the float16 operands are rewritten onto 8x8x8 fragments, the
   integer case cannot be (specs/matrix-pattern-rewrite.md, "Metal"). *)
let metal_float =
  ( "Metal (Apple GPU)",
    "float16 operands, or float32 rounded to float16, onto 8x8x8 fragments \
     (explicit request, or the automatic opt-in); the bfloat16 lowerings: none"
  )

let metal_integer = ("Metal", "none (MSL has no integer simdgroup_matrix)")

let float_backends =
  [
    ("HIP gfx11 (wave32)", "executes");
    ("CUDA/C sm_70+", "none (the rewrite lowers the integer case only)");
    ("Vulkan", "none (the GLSL emitter lowers integer configurations only)");
    metal_float;
    others;
  ]

let integer_backends =
  [
    ("HIP gfx11 (wave32)", "executes");
    ( "CUDA/C sm_72+",
      "explicit request only; compiled (NVRTC), never run; ZLUDA lacks wmma" );
    ("Vulkan", "executes (RADV, 64-wide subgroups)");
    metal_integer;
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
        accuracy_class = Some Relaxed_accumulation;
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
        accuracy_class = Some Relaxed_operands;
        accuracy =
          "relaxed beyond the accumulation order: no matrix unit takes float32 \
           operands, so they are rounded (float16: relative error up to 2^-11 \
           each; bfloat16; or a three-term bfloat16 split, near float32); a \
           rewrite needs Sarek_coopmat.accept_operand_rounding ~precision and \
           accept_relaxed_accumulation";
        backends = float_backends;
      }
  | Int32_inputs ->
      {
        case = Int32_inputs;
        config = Some (config ~a:C.Uint8 ~acc:C.Sint32);
        lowering = Some (Byte_planes byte_plane_products);
        accuracy_class = Some Exact;
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
        accuracy_class = None;
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

let elt_name = function
  | TFloat16 -> "float16"
  | TFloat32 -> "float32"
  | TFloat64 -> "float64"
  | TInt32 -> "int32"
  | TInt64 -> "int64"
  | _ -> "other"

let is_zero = function
  | EConst (CInt32 0l | CInt64 0L) -> true
  | EConst (CFloat32 f | CFloat64 f) -> f = 0.0
  | _ -> false

(* What was found: the product, the operands and the indices. *)
let pattern_lines (g : P.gemm) =
  let kv = g.k_var.var_name in
  let init = if is_zero g.init then "" else expr_text g.init ^ " + " in
  [
    Printf.sprintf
      "  C[m,n] = %ssum over %s = %s..%s of A[m,%s] * B[%s,n]"
      init
      kv
      (expr_text g.k_lo)
      (expr_text g.k_hi)
      kv
      kv;
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
    Printf.sprintf
      "  elements: %s x %s -> %s accumulator"
      (elt_name g.a.elt)
      (elt_name g.b.elt)
      (elt_name g.acc.var_type);
  ]

(* What Execute does with the nest at launch (specs/matrix-pattern-dynamic.md). *)
let rewrite_note = function
  | Int32_inputs ->
      "Execute.run_vectors rewrites it onto matrix units by default where they \
       exist (HIP gfx11, Vulkan; CUDA/C on request only); \
       SAREK_MATRIX_REWRITE=off disables it."
  | F16_inputs | F32_inputs ->
      "Execute.run_vectors rewrites it onto matrix units only after the opt-in \
       above AND Sarek_coopmat.accept_automatic_float_rewrite (and \
       SAREK_MATRIX_REWRITE not off or exact); Sarek_matrix.run needs the \
       opt-in above only."
  | No_case -> "Not rewritten."

let render ?kernel_name (g : P.gemm) adv =
  let name =
    match kernel_name with
    | Some n -> Printf.sprintf "kernel '%s'" n
    | None -> "this kernel"
  in
  let lines =
    Printf.sprintf
      "Sarek: %s contains a matrix product (GEMM) that matrix units could \
       compute."
      name
    :: pattern_lines g
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
        "  " ^ rewrite_note adv.case;
        "  Silence with [@alert \"-sarek_matrix\"] on the kernel expression or \
         its binding.";
      ]
  in
  String.concat "\n" lines
