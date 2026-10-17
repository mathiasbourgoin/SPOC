(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Source-text lowering of the warp operations, shared by the source backends
    (CUDA C / HIP, OpenCL C, GLSL, MSL, WGSL).

    The semantics are those of {!Sarek_warp}: every lane of the warp
    participates, and the result for lane [i] is the table of
    docs/plans/warp-shuffle-intake-2026-10-17.md section 4.3. The vendors differ
    in what they leave undefined (an out-of-range delta, the high bits of a lane
    operand), so no backend passes the Sarek operand to a relative shuffle
    unguarded:

    - CUDA C / HIP clamp an out-of-range [delta] / [mask] to 0 (which every
      native relative shuffle maps to "own value") and use the native
      [__shfl_down] / [__shfl_up] / [__shfl_xor] / [__shfl];
    - OpenCL C, GLSL, MSL and WGSL compute the source lane explicitly from the
      lane id and the warp width, and call the INDEX shuffle with it. Only the
      base shuffle extension is then needed (no [_relative] / [_shuffle_up]
      extension, no SHUFFLE_RELATIVE feature bit), and the out-of-range cases
      are defined by construction.

    A shuffle is emitted as a call of a small helper function, defined once per
    (operation, value type) the kernel uses, so an operand expression is
    evaluated exactly once. The value type is part of the IR name
    ({!Sarek_warp.ir_name}), so no backend infers expression types. *)

type dialect = Cuda | Opencl | Glsl | Metal | Wgsl

let value_ctype dialect ty =
  match (dialect, ty) with
  | (Cuda | Opencl | Metal | Glsl), Sarek_warp.I32 -> "int"
  | (Cuda | Opencl | Metal | Glsl), Sarek_warp.F32 -> "float"
  | (Cuda | Opencl | Metal | Glsl), Sarek_warp.F64 -> "double"
  | Wgsl, Sarek_warp.I32 -> "i32"
  | Wgsl, Sarek_warp.F32 -> "f32"
  | Wgsl, Sarek_warp.F64 -> "f64"

(** The name of the helper a shuffle call is emitted as. *)
let helper_name op ty = "sarek_" ^ Sarek_warp.ir_name op ty

let shuffles_of_calls calls =
  List.filter_map
    (function Sarek_warp.Shuffle (op, ty) -> Some (op, ty) | _ -> None)
    calls

let uses_shuffle calls = shuffles_of_calls calls <> []

(** {1 The source-lane expression}

    Written once over a tiny syntax record, instantiated for C-like targets,
    GLSL and WGSL. [x] is the signed lane operand, [lane] and [w] the lane id
    and the warp width, both unsigned. *)

type syntax = {
  to_u32 : string -> string;
  one : string;  (** the literal 1 as an unsigned *)
  select : cond:string -> then_:string -> else_:string -> string;
}

let c_syntax =
  {
    to_u32 = (fun e -> "(uint)(" ^ e ^ ")");
    one = "1u";
    select =
      (fun ~cond ~then_ ~else_ ->
        Printf.sprintf "((%s) ? (%s) : (%s))" cond then_ else_);
  }

let cuda_syntax =
  {c_syntax with to_u32 = (fun e -> "(unsigned int)(" ^ e ^ ")")}

let glsl_syntax = {c_syntax with to_u32 = (fun e -> "uint(" ^ e ^ ")")}

let wgsl_syntax =
  {
    to_u32 = (fun e -> "u32(" ^ e ^ ")");
    one = "1u";
    select =
      (fun ~cond ~then_ ~else_ ->
        Printf.sprintf "select(%s, %s, %s)" else_ then_ cond);
  }

(** The unsigned source lane for operation [op], operand expression [x]. *)
let source_lane_expr syn op ~x ~lane ~w =
  let ux = syn.to_u32 x in
  match op with
  | Sarek_warp.Idx -> Printf.sprintf "(%s & (%s - %s))" ux w syn.one
  | Down ->
      syn.select
        ~cond:(Printf.sprintf "%s < %s - %s" ux w lane)
        ~then_:(Printf.sprintf "%s + %s" lane ux)
        ~else_:lane
  | Up ->
      syn.select
        ~cond:(Printf.sprintf "%s <= %s" ux lane)
        ~then_:(Printf.sprintf "%s - %s" lane ux)
        ~else_:lane
  | Xor ->
      syn.select
        ~cond:(Printf.sprintf "%s < %s && (%s ^ %s) < %s" ux w lane ux w)
        ~then_:(Printf.sprintf "%s ^ %s" lane ux)
        ~else_:lane

(** {1 Calls}

    The spelling of a warp operation at its call site. *)

let metal_lane = "__sarek_lane"

let metal_width = "__sarek_w"

let wgsl_lane = "sarek_lane"

let wgsl_width = "sarek_w"

(** [warp_size] and [lane_id], as int32 expressions. *)
let constant_expr dialect call =
  match (dialect, call) with
  | Cuda, Sarek_warp.Warp_size -> "((int)warpSize)"
  | Cuda, Lane_id ->
      "((int)((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * \
       threadIdx.z)) % (unsigned int)warpSize))"
  | Opencl, Warp_size -> "((int)get_sub_group_size())"
  | Opencl, Lane_id -> "((int)get_sub_group_local_id())"
  | Glsl, Warp_size -> "int(gl_SubgroupSize)"
  | Glsl, Lane_id -> "int(gl_SubgroupInvocationID)"
  | Metal, Warp_size -> Printf.sprintf "((int)%s)" metal_width
  | Metal, Lane_id -> Printf.sprintf "((int)%s)" metal_lane
  | Wgsl, Warp_size -> Printf.sprintf "i32(%s)" wgsl_width
  | Wgsl, Lane_id -> Printf.sprintf "i32(%s)" wgsl_lane
  | _, Shuffle _ -> invalid_arg "Sarek_ir_warp_text.constant_expr: a shuffle"

(** Emit [warp_size], [lane_id] or a shuffle call. [args] are the already
    rendered operand strings; [bad_arity] raises the backend's located arity
    error. *)
let emit_call dialect ~bad_arity buf call (args : string list) =
  match (call, args) with
  | (Sarek_warp.Warp_size | Lane_id), [] ->
      Buffer.add_string buf (constant_expr dialect call)
  | Shuffle (op, ty), [value; operand] ->
      let extra =
        match dialect with Metal -> [metal_lane; metal_width] | _ -> []
      in
      Printf.bprintf
        buf
        "%s(%s)"
        (helper_name op ty)
        (String.concat ", " ([value; operand] @ extra))
  | Shuffle (op, ty), _ ->
      bad_arity (Sarek_warp.ir_name op ty) 2 (List.length args)
  | (Warp_size | Lane_id), _ ->
      bad_arity
        (match call with Warp_size -> "warp_size" | _ -> "lane_id")
        0
        (List.length args)

(** {1 Helper definitions} *)

let cuda_helper op ty =
  let t = value_ctype Cuda ty in
  let base =
    match op with
    | Sarek_warp.Idx -> "shfl"
    | Down -> "shfl_down"
    | Up -> "shfl_up"
    | Xor -> "shfl_xor"
  in
  (* Idx masks the operand to the warp; the relative shuffles clamp an operand
     outside [0, warpSize) to 0, which every one of them maps to the own value
     (a delta or xor mask of 0), exactly the table of Sarek_warp. *)
  let operand =
    match op with
    | Sarek_warp.Idx -> "(x & (warpSize - 1))"
    | Down | Up | Xor -> "(((unsigned int)x < (unsigned int)warpSize) ? x : 0)"
  in
  Printf.sprintf
    "__device__ __forceinline__ %s %s(%s v, int x) {\n\
     #if defined(__HIP_PLATFORM_AMD__)\n\
    \  return __%s(v, %s, warpSize);\n\
     #else\n\
    \  return __%s_sync(0xffffffffu, v, %s, 32);\n\
     #endif\n\
     }\n"
    t
    (helper_name op ty)
    t
    base
    operand
    base
    operand

(* The index shuffle of one value, in each target that computes the source lane
   itself. *)
let index_shuffle dialect ty ~src =
  match (dialect, ty) with
  | Opencl, _ -> Printf.sprintf "sub_group_shuffle(v, %s)" src
  | Metal, _ -> Printf.sprintf "simd_shuffle(v, (ushort)(%s))" src
  | Wgsl, _ -> Printf.sprintf "subgroupShuffle(v, %s)" src
  | Glsl, (Sarek_warp.I32 | F32) -> Printf.sprintf "subgroupShuffle(v, %s)" src
  | Glsl, F64 -> assert false
  | Cuda, _ -> assert false

let c_helper dialect op ty =
  let t = value_ctype dialect ty in
  let lane, w, extra_params, pre =
    match dialect with
    | Metal -> ("lane", "w", ", uint lane, uint w", "")
    | Opencl ->
        ( "lane",
          "w",
          "",
          "  uint w = get_sub_group_size();\n\
          \  uint lane = get_sub_group_local_id();\n" )
    | _ -> assert false
  in
  let src = source_lane_expr c_syntax op ~x:"x" ~lane ~w in
  Printf.sprintf
    "%sinline %s %s(%s v, int x%s) {\n%s  return %s;\n}\n"
    (if dialect = Metal then "static " else "")
    t
    (helper_name op ty)
    t
    extra_params
    pre
    (index_shuffle dialect ty ~src)

let glsl_helper op ty =
  let src =
    source_lane_expr
      glsl_syntax
      op
      ~x:"x"
      ~lane:"gl_SubgroupInvocationID"
      ~w:"gl_SubgroupSize"
  in
  let name = helper_name op ty in
  match ty with
  | Sarek_warp.F64 ->
      (* 64-bit subgroup operations need an extended-types feature the base
         shuffle extension does not give: shuffle the two 32-bit halves. *)
      Printf.sprintf
        "double %s(double v, int x) {\n\
        \  uint src = %s;\n\
        \  uvec2 p = unpackDouble2x32(v);\n\
        \  return packDouble2x32(uvec2(subgroupShuffle(p.x, src), \
         subgroupShuffle(p.y, src)));\n\
         }\n"
        name
        src
  | I32 | F32 ->
      Printf.sprintf
        "%s %s(%s v, int x) {\n  uint src = %s;\n  return %s;\n}\n"
        (value_ctype Glsl ty)
        name
        (value_ctype Glsl ty)
        src
        (index_shuffle Glsl ty ~src:"src")

let wgsl_helper op ty =
  let t = value_ctype Wgsl ty in
  let src =
    source_lane_expr wgsl_syntax op ~x:"x" ~lane:wgsl_lane ~w:wgsl_width
  in
  Printf.sprintf
    "fn %s(v : %s, x : i32) -> %s {\n  return %s;\n}\n"
    (helper_name op ty)
    t
    t
    (index_shuffle Wgsl ty ~src)

let helper_source dialect op ty =
  match dialect with
  | Cuda -> cuda_helper op ty
  | Opencl | Metal -> c_helper dialect op ty
  | Glsl -> glsl_helper op ty
  | Wgsl -> wgsl_helper op ty

(** {1 Preambles} *)

(** The helper definitions for the shuffles in [calls]. *)
let helpers dialect calls =
  String.concat
    "\n"
    (List.map
       (fun (op, ty) -> helper_source dialect op ty)
       (shuffles_of_calls calls))

(** OpenCL C extension pragmas for [calls]: sub-group built-ins, and the shuffle
    extension when a shuffle is used. Empty when the kernel uses no warp
    operation. *)
let opencl_pragmas calls =
  if calls = [] then ""
  else
    "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
    ^
    if uses_shuffle calls then
      "#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable\n"
    else ""

(** GLSL [#extension] lines for [calls] (placed in the header, before any
    declaration). *)
let glsl_extensions calls =
  if calls = [] then ""
  else
    "#extension GL_KHR_shader_subgroup_basic : require\n"
    ^
    if uses_shuffle calls then
      "#extension GL_KHR_shader_subgroup_shuffle : require\n"
    else ""

(** The WGSL directive and module-scope state: [enable subgroups;] must come
    first in the module; the lane and width are module-scope private variables
    that [main] assigns from its subgroup built-ins, so helper functions can
    read them. *)
let wgsl_enable calls = if calls = [] then "" else "enable subgroups;\n"

let wgsl_state calls =
  if calls = [] then ""
  else
    Printf.sprintf
      "var<private> %s : u32;\nvar<private> %s : u32;\n"
      wgsl_lane
      wgsl_width

(** Extra [main] parameters and prologue for WGSL. *)
let wgsl_main_params calls =
  if calls = [] then ""
  else
    ",\n\
    \  @builtin(subgroup_invocation_id) sarek_sg_lane : u32,\n\
    \  @builtin(subgroup_size) sarek_sg_size : u32"

let wgsl_main_prologue calls =
  if calls = [] then ""
  else
    Printf.sprintf
      "  %s = sarek_sg_lane;\n  %s = sarek_sg_size;\n"
      wgsl_lane
      wgsl_width

(** Extra Metal kernel parameters (the lane and the width are attributes of the
    kernel entry, not readable from a helper, so they are passed on). *)
let metal_params calls =
  if calls = [] then ""
  else
    Printf.sprintf
      ",\n\
      \  uint %s [[thread_index_in_simdgroup]],\n\
      \  uint %s [[threads_per_simdgroup]]"
      metal_lane
      metal_width
