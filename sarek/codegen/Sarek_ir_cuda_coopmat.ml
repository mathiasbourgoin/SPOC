(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative matrix ([SCoopmat]) for the CUDA-C generator, which HIP shares.

    One generated source serves both toolchains, so every construct here is
    emitted twice under the platform guard the generator already uses for its
    f32 barrier: the [__HIP_PLATFORM_AMD__] arm is RDNA3 (gfx11) WMMA through
    the clang builtins, the other arm is [nvcuda::wmma]. See
    specs/coopmat-hip-wmma.md for the configurations, the per-lane layout and
    the reasons for both choices.

    {b What is validated where.} The AMD arm is executed on an RX 7900 XTX
    (gfx1100) against the interpreter: bit-identical for the integer
    configurations, within the derived bound for f16. The NVIDIA arm is
    compile-checked only (NVRTC, and ptxas/nvdisasm when present): no NVIDIA GPU
    was available, and the CUDA backend does not probe [coopmat], so the launch
    gate refuses it before this code is ever compiled for a device. *)

open Sarek_ir_types
module C = Sarek_coopmat_types

(* The CUDA generator's error module, so a refusal here is the same exception,
   tagged the same way, as every other CUDA/HIP codegen refusal. *)
module Codegen_error = Sarek_backend_error.Backend_error.Make (struct
  let name = "CUDA"
end)

let unsupported what why =
  Codegen_error.raise_error (Codegen_error.unsupported_construct what why)

(** {1 What the lowering accepts} *)

let shape_16 = {C.m = 16; n = 16; k = 16}

let is_8bit = function C.Uint8 | C.Sint8 -> true | _ -> false

let is_int32 = function C.Uint32 | C.Sint32 -> true | _ -> false

(** Refuse, at code generation, a fragment this lowering cannot represent. Total
    over the fragment type: every case either names its representation or raises
    with the reason. *)
let check_fragment (f : C.fragment) =
  if f.C.frag_shape <> shape_16 then
    unsupported
      "cooperative-matrix shape"
      (Printf.sprintf
         "CUDA/HIP: only 16x16x16 is lowered (RDNA3 WMMA and nvcuda::wmma \
          m16n16k16); got %s"
         (C.shape_name f.C.frag_shape)) ;
  if f.C.frag_scope <> C.Subgroup then
    unsupported
      "cooperative-matrix scope"
      (Printf.sprintf
         "CUDA/HIP: only subgroup (wavefront / warp) scope exists; got %s"
         (C.scope_name f.C.frag_scope)) ;
  match (f.C.frag_use, f.C.frag_component) with
  | (C.Matrix_a | C.Matrix_b), (C.Float16 | C.Uint8 | C.Sint8) -> ()
  | C.Accumulator, (C.Float32 | C.Uint32 | C.Sint32) -> ()
  | use, comp ->
      unsupported
        "cooperative-matrix fragment"
        (Printf.sprintf
           "CUDA/HIP: no %s fragment of %s (operands are f16, u8 or s8; \
            accumulators are f32, s32 or u32)"
           (C.use_name use)
           (C.component_name comp))

(** The configurations the lowering emits a multiply-add for. *)
let check_config (cfg : C.config) =
  let a, b, c, r = C.fragments_of_config cfg in
  List.iter check_fragment [a; b; c; r] ;
  if cfg.C.cfg_saturating then
    unsupported
      "saturating cooperative-matrix accumulation"
      "CUDA/HIP: the saturating variant computes a different function and the \
       interpreter, the oracle it would be checked against, refuses it" ;
  let f16 = cfg.C.cfg_a = C.Float16 && cfg.C.cfg_b = C.Float16 in
  let int8 = is_8bit cfg.C.cfg_a && is_8bit cfg.C.cfg_b in
  let ok =
    (f16 && cfg.C.cfg_c = C.Float32 && cfg.C.cfg_result = C.Float32)
    || (int8 && is_int32 cfg.C.cfg_c && cfg.C.cfg_c = cfg.C.cfg_result)
  in
  if not ok then
    unsupported
      "cooperative-matrix configuration"
      (Printf.sprintf
         "CUDA/HIP: %s is not lowered (f16*f16+f32->f32 and \
          8-bit*8-bit+i32->i32 with the same 32-bit accumulator type are)"
         (C.config_name cfg))

(** The kernel buffer element type a fragment may be loaded from or stored to.
    Integer accumulators of either signedness live in an [int32] buffer, and
    both 8-bit operand types in a [uint8] buffer: the signedness is the
    fragment's, applied by the instruction, exactly as the interpreter narrows
    the loaded byte. *)
let buffer_elttype (f : C.fragment) =
  match f.C.frag_component with
  | C.Float16 -> TFloat16
  | C.Float32 -> TFloat32
  | C.Uint8 | C.Sint8 -> TUint8
  | C.Uint32 | C.Sint32 -> TInt32

let elttype_name = function
  | TFloat16 -> "float16"
  | TFloat32 -> "float32"
  | TUint8 -> "uint8"
  | TInt32 -> "int32"
  | TInt64 -> "int64"
  | TFloat64 -> "float64"
  | _ -> "a non-numeric type"

(** [buffers] maps a parameter or shared-array name to its element type, when
    the generator knows it. An unknown name is left to the C++ compiler, which
    rejects a mismatch too: each helper below takes exactly one pointer type. *)
let check_buffer ~buffers ~buf (f : C.fragment) =
  match List.assoc_opt buf buffers with
  | None -> ()
  | Some elt ->
      let want = buffer_elttype f in
      if elt <> want then
        unsupported
          "cooperative-matrix buffer"
          (Printf.sprintf
             "a %s %s fragment needs a %s buffer, but %s holds %s"
             (C.component_name f.C.frag_component)
             (C.use_name f.C.frag_use)
             (elttype_name want)
             buf
             (elttype_name elt))

(** Element types of the kernel's array parameters and shared arrays. *)
let kernel_buffers (k : kernel) =
  let of_param = function
    | DParam (v, Some {arr_elttype; _}) -> Some (v.var_name, arr_elttype)
    | DParam ({var_type = TVec elt; var_name; _}, None) -> Some (var_name, elt)
    | DParam _ | DLocal _ -> None
    | DShared (name, elt, _) -> Some (name, elt)
  in
  List.filter_map of_param (k.kern_params @ k.kern_locals)

(** {1 Preamble} *)

(** Emitted before [extern "C"] when the kernel has a cooperative-matrix
    statement. The AMD helpers have C++ linkage and one pointer type each, so a
    buffer of the wrong element type is a compile error rather than a
    reinterpretation. *)
let preamble =
  {|#if defined(__HIP_PLATFORM_AMD__)
/* Sarek cooperative matrix, HIP: RDNA3 (gfx11) WMMA, wave32
   (specs/coopmat-hip-wmma.md). For lane l, with l16 = l % 16 and h = l / 16:
   an A fragment holds row l16 (k = 0..15), a B fragment holds column l16, and
   accumulator element j holds row 2*j + h, column l16. */
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX11__)
#error "Sarek cooperative matrix: the HIP lowering targets gfx11 (RDNA3) WMMA only"
#endif
typedef _Float16 sarek_cm_h16 __attribute__((ext_vector_type(16)));
typedef float sarek_cm_f8 __attribute__((ext_vector_type(8)));
typedef int sarek_cm_i4 __attribute__((ext_vector_type(4)));
typedef int sarek_cm_i8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ int sarek_cm_l16() { return (int)(__lane_id() & 15u); }
__device__ __forceinline__ int sarek_cm_h() { return (int)((__lane_id() >> 4) & 1u); }
__device__ __forceinline__ unsigned sarek_cm_pack4(unsigned char b0, unsigned char b1,
                                                   unsigned char b2, unsigned char b3) {
  return (unsigned)b0 | ((unsigned)b1 << 8) | ((unsigned)b2 << 16) | ((unsigned)b3 << 24);
}
__device__ __forceinline__ void sarek_cm_load_a_f16(sarek_cm_h16 &f, const __half *p, int i, int s) {
  const _Float16 *q = (const _Float16 *)p + i + sarek_cm_l16() * s;
  for (int e = 0; e < 16; ++e) f[e] = q[e];
}
__device__ __forceinline__ void sarek_cm_load_b_f16(sarek_cm_h16 &f, const __half *p, int i, int s) {
  const _Float16 *q = (const _Float16 *)p + i + sarek_cm_l16();
  for (int e = 0; e < 16; ++e) f[e] = q[e * s];
}
__device__ __forceinline__ void sarek_cm_load_a_i8(sarek_cm_i4 &f, const unsigned char *p, int i, int s) {
  const unsigned char *q = p + i + sarek_cm_l16() * s;
  for (int w = 0; w < 4; ++w)
    f[w] = (int)sarek_cm_pack4(q[4 * w], q[4 * w + 1], q[4 * w + 2], q[4 * w + 3]);
}
__device__ __forceinline__ void sarek_cm_load_b_i8(sarek_cm_i4 &f, const unsigned char *p, int i, int s) {
  const unsigned char *q = p + i + sarek_cm_l16();
  for (int w = 0; w < 4; ++w)
    f[w] = (int)sarek_cm_pack4(q[(4 * w) * s], q[(4 * w + 1) * s],
                               q[(4 * w + 2) * s], q[(4 * w + 3) * s]);
}
__device__ __forceinline__ void sarek_cm_load_acc_f32(sarek_cm_f8 &f, const float *p, int i, int s) {
  const float *q = p + i + sarek_cm_h() * s + sarek_cm_l16();
  for (int j = 0; j < 8; ++j) f[j] = q[2 * j * s];
}
__device__ __forceinline__ void sarek_cm_store_acc_f32(const sarek_cm_f8 &f, float *p, int i, int s) {
  float *q = p + i + sarek_cm_h() * s + sarek_cm_l16();
  for (int j = 0; j < 8; ++j) q[2 * j * s] = f[j];
}
__device__ __forceinline__ void sarek_cm_load_acc_i32(sarek_cm_i8 &f, const int *p, int i, int s) {
  const int *q = p + i + sarek_cm_h() * s + sarek_cm_l16();
  for (int j = 0; j < 8; ++j) f[j] = q[2 * j * s];
}
__device__ __forceinline__ void sarek_cm_store_acc_i32(const sarek_cm_i8 &f, int *p, int i, int s) {
  int *q = p + i + sarek_cm_h() * s + sarek_cm_l16();
  for (int j = 0; j < 8; ++j) q[2 * j * s] = f[j];
}
#else
/* Sarek cooperative matrix, CUDA: nvcuda::wmma m16n16k16. COMPILE-CHECKED
   ONLY (no NVIDIA GPU was available; specs/coopmat-hip-wmma.md). wmma's
   load/store need a 32-byte aligned pointer and a row stride that is a
   multiple of 16 bytes; Sarek's CM_load promises any index and stride, so a
   launch that violates the precondition traps instead of computing on a
   misaligned fragment. */
#include <mma.h>
__device__ __forceinline__ void sarek_cm_require(const void *p, int s, int bytes) {
  if ((((unsigned long long)p) & 31ull) != 0ull || ((s * bytes) & 15) != 0) __trap();
}
#endif
|}

(** The header comment §6.1 of docs/design/f16-relaxed-accuracy.md requires on a
    generated source whose results are bounded rather than exact. *)
let relaxed_comment (k : kernel) =
  let relaxed =
    List.filter
      (fun cfg -> C.regime cfg = C.Relaxed_bounded)
      (Sarek_ir_analysis.kernel_coopmat_configs k)
  in
  match relaxed with
  | [] -> ""
  | cfgs ->
      Printf.sprintf
        "/* Sarek: %s accumulates in floating point in an order the hardware \
         chooses. Results are within gamma_n * (sum|A*B| + |C|) of the exact \
         value (u = 2^-24), NOT bit-identical to the interpreter: Regime B of \
         docs/design/f16-relaxed-accuracy.md; launching it requires \
         Sarek_coopmat.accept_relaxed_accumulation. */\n"
        (String.concat ", " (List.map C.config_name cfgs))

(** {1 Statements} *)

let amd_type (f : C.fragment) =
  match (f.C.frag_use, f.C.frag_component) with
  | (C.Matrix_a | C.Matrix_b), C.Float16 -> "sarek_cm_h16"
  | (C.Matrix_a | C.Matrix_b), (C.Uint8 | C.Sint8) -> "sarek_cm_i4"
  | C.Accumulator, C.Float32 -> "sarek_cm_f8"
  | C.Accumulator, (C.Uint32 | C.Sint32) -> "sarek_cm_i8"
  | use, comp ->
      unsupported
        "cooperative-matrix fragment"
        (Printf.sprintf
           "HIP: no %s fragment of %s"
           (C.use_name use)
           (C.component_name comp))

(** Suffix of the AMD helper for a fragment: [a_f16], [b_i8], [acc_f32]... *)
let amd_helper_suffix (f : C.fragment) =
  let role =
    match f.C.frag_use with
    | C.Matrix_a -> "a"
    | C.Matrix_b -> "b"
    | C.Accumulator -> "acc"
  in
  let ty =
    match f.C.frag_component with
    | C.Float16 -> "f16"
    | C.Float32 -> "f32"
    | C.Uint8 | C.Sint8 -> "i8"
    | C.Uint32 | C.Sint32 -> "i32"
  in
  role ^ "_" ^ ty

let cuda_elt = function
  | C.Float16 -> "__half"
  | C.Float32 -> "float"
  | C.Uint8 -> "unsigned char"
  | C.Sint8 -> "signed char"
  | C.Uint32 | C.Sint32 -> "int"

let cuda_elt_bytes = function
  | C.Float16 -> 2
  | C.Float32 | C.Uint32 | C.Sint32 -> 4
  | C.Uint8 | C.Sint8 -> 1

let cuda_type (f : C.fragment) =
  let w = "nvcuda::wmma::" in
  let elt = cuda_elt f.C.frag_component in
  match f.C.frag_use with
  | C.Matrix_a ->
      Printf.sprintf
        "%sfragment<%smatrix_a, 16, 16, 16, %s, %srow_major>"
        w
        w
        elt
        w
  | C.Matrix_b ->
      Printf.sprintf
        "%sfragment<%smatrix_b, 16, 16, 16, %s, %srow_major>"
        w
        w
        elt
        w
  | C.Accumulator ->
      Printf.sprintf "%sfragment<%saccumulator, 16, 16, 16, %s>" w w elt

(** The two arms of one statement, under the platform guard. Preprocessor
    directives start at column 0. *)
let both_arms ~indent ~amd ~cuda =
  Printf.sprintf
    "#if defined(__HIP_PLATFORM_AMD__)\n%s%s\n#else\n%s%s\n#endif\n"
    indent
    amd
    indent
    cuda

let gen_decl ~indent name (frag : C.fragment) =
  check_fragment frag ;
  both_arms
    ~indent
    ~amd:(Printf.sprintf "%s %s = {};" (amd_type frag) name)
    ~cuda:
      (Printf.sprintf
         "%s %s; nvcuda::wmma::fill_fragment(%s, 0);"
         (cuda_type frag)
         name
         name)

(* The CUDA pointer for a fragment's buffer: [src + index], cast to the wmma
   element type (an s8 fragment reads a uint8 buffer as signed char). *)
let cuda_ptr ~const (frag : C.fragment) ~buf ~index =
  Printf.sprintf
    "((%s%s *)(%s) + (%s))"
    (if const then "const " else "")
    (cuda_elt frag.C.frag_component)
    buf
    index

let gen_load ~indent ~buffers ~expr_str ~dst ~(frag : C.fragment) ~src ~index
    ~stride =
  check_fragment frag ;
  check_buffer ~buffers ~buf:src frag ;
  let index = expr_str index and stride = expr_str stride in
  let ptr = cuda_ptr ~const:true frag ~buf:src ~index in
  let layout =
    match frag.C.frag_use with
    | C.Accumulator -> ", nvcuda::wmma::mem_row_major"
    | C.Matrix_a | C.Matrix_b -> ""
  in
  both_arms
    ~indent
    ~amd:
      (Printf.sprintf
         "sarek_cm_load_%s(%s, %s, (%s), (%s));"
         (amd_helper_suffix frag)
         dst
         src
         index
         stride)
    ~cuda:
      (Printf.sprintf
         "sarek_cm_require(%s, (%s), %d); nvcuda::wmma::load_matrix_sync(%s, \
          %s, (%s)%s);"
         ptr
         stride
         (cuda_elt_bytes frag.C.frag_component)
         dst
         ptr
         stride
         layout)

let gen_store ~indent ~buffers ~expr_str ~src ~(frag : C.fragment) ~dst ~index
    ~stride =
  check_fragment frag ;
  if frag.C.frag_use <> C.Accumulator then
    unsupported
      "cooperative-matrix store"
      "CUDA/HIP: only an accumulator fragment can be stored (nvcuda::wmma has \
       no store for matrix_a / matrix_b)" ;
  check_buffer ~buffers ~buf:dst frag ;
  let index = expr_str index and stride = expr_str stride in
  let ptr = cuda_ptr ~const:false frag ~buf:dst ~index in
  both_arms
    ~indent
    ~amd:
      (Printf.sprintf
         "sarek_cm_store_%s(%s, %s, (%s), (%s));"
         (amd_helper_suffix frag)
         src
         dst
         index
         stride)
    ~cuda:
      (Printf.sprintf
         "sarek_cm_require(%s, (%s), %d); nvcuda::wmma::store_matrix_sync(%s, \
          %s, (%s), nvcuda::wmma::mem_row_major);"
         ptr
         stride
         (cuda_elt_bytes frag.C.frag_component)
         ptr
         src
         stride)

let gen_muladd ~indent ~dst ~a ~b ~c (cfg : C.config) =
  check_config cfg ;
  let amd =
    if cfg.C.cfg_a = C.Float16 then
      Printf.sprintf
        "%s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(%s, %s, %s);"
        dst
        a
        b
        c
    else
      let signed comp = if comp = C.Sint8 then "true" else "false" in
      Printf.sprintf
        "%s = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(%s, %s, %s, %s, %s, \
         false);"
        dst
        (signed cfg.C.cfg_a)
        a
        (signed cfg.C.cfg_b)
        b
        c
  in
  let cuda =
    if cfg.C.cfg_a = C.Float16 then
      Printf.sprintf "nvcuda::wmma::mma_sync(%s, %s, %s, %s);" dst a b c
    else if cfg.C.cfg_a <> cfg.C.cfg_b then
      (* nvcuda::wmma's integer mma_sync takes two signed or two unsigned
         operands, never one of each. The AMD arm is valid, so this cannot be
         a codegen-time refusal of the shared source; the CUDA toolchain gets
         a named #error instead (and the CUDA launch gate refuses first). *)
      Printf.sprintf
        "\n\
         #error \"Sarek cooperative matrix: nvcuda::wmma has no mixed-sign \
         8-bit multiply (%s)\"\n"
        (C.config_name cfg)
    else
      Printf.sprintf "nvcuda::wmma::mma_sync(%s, %s, %s, %s, false);" dst a b c
  in
  both_arms ~indent ~amd ~cuda

(** [gen_op ~indent ~buffers ~expr_str op] is the source for one cooperative
    matrix statement, both arms. *)
let gen_op ~indent ~buffers ~expr_str = function
  | CM_decl {name; frag} -> gen_decl ~indent name frag
  | CM_load {dst; frag; src; index; stride} ->
      gen_load ~indent ~buffers ~expr_str ~dst ~frag ~src ~index ~stride
  | CM_store {src; frag; dst; index; stride} ->
      gen_store ~indent ~buffers ~expr_str ~src ~frag ~dst ~index ~stride
  | CM_muladd {dst; a; b; c; cfg} -> gen_muladd ~indent ~dst ~a ~b ~c cfg

(** {1 Launch shape} *)

(** The lowering runs the matrix instruction on whole wavefronts (wave32 on
    gfx11) and whole warps; a block whose thread count is not a multiple of 32
    has a partial wave whose inactive lanes would feed the instruction undefined
    operands. Refused before any source is compiled. *)
let check_block (k : kernel) (block : int * int * int) =
  let bx, by, bz = block in
  let threads = bx * by * bz in
  if threads mod 32 <> 0 && Sarek_ir_analysis.kernel_has_coopmat_op k then
    unsupported
      "cooperative-matrix launch shape"
      (Printf.sprintf
         "CUDA/HIP: a cooperative-matrix kernel must run whole 32-wide \
          wavefronts/warps, but the block has %d threads (%dx%dx%d)"
         threads
         bx
         by
         bz)
