# Scaled accumulation of a cooperative-matrix fragment (design, not implemented)

_An operation that adds a per-row and per-column scaled copy of one accumulator
fragment (int32 or f32) into an f32 accumulator, for quantised GEMMs in the style
of llama.cpp's MMQ kernels and for the per-row rescale of online softmax._

**Status:** design only. Implementation is **deferred**: the measurement in §1
shows that 8-bit WMMA is not faster than f16 WMMA on gfx1100, so the first
consumer (a Katra int8 prompt path) would gain only the dequantisation work it
removes, not matrix throughput. **Date:** 2026-10-12. **Requested by:** the Katra
prompt plan (lever C, SPOC items S1 and S2).

## 1. Decisive measurement: iu8 vs f16 WMMA peak on gfx1100

The integer path is only worth a new IR operation if the integer matrix unit is
faster than the f16 one. It is not.

**Kernel.** One `SCoopmat` IR kernel per configuration, lowered by the existing
HIP coopmat code (`sarek/codegen/Sarek_ir_cuda_coopmat.ml`, unchanged): A and B
loaded once, four independent accumulators, a loop of `n` iterations each issuing
four `CM_muladd` (`acc_k = A * B + acc_k`), the four accumulators stored at the
end. Block 256 threads (8 waves), grid 1536 blocks, no other memory traffic. Ops
counted as `2 * 16^3` per `CM_muladd`. The lowered instructions are
`v_wmma_f32_16x16x16_f16` and `v_wmma_i32_16x16x16_iu8` (ISA checked with
`hipcc --offload-arch=gfx1100 -O3 --cuda-device-only -S`, ROCm 7.2.4).

**Machine and conditions.** RX 7900 XTX (gfx1100), ROCm 7.2.4, only the dGPU
visible (`HIP_VISIBLE_DEVICES=0`), run under `scripts/gpu-kernel-log-gate.sh`
(clean log). Other sessions were loading the CPU (load average 3.3 to 4.4) but
not this measurement's GPU queue: runs were serialised behind a shared lock.
Best of 5 launches after one warm-up; two configurations alternated twice, in
both orders. Raw output is kept outside the repository.

| run | n | s8 x s8 + s32 | f16 x f16 + f32 | iu8 / f16 |
|---|---|---|---|---|
| f16 first | 1024 | 130.2, 117.7 TOPS | 137.5, 135.5 TFLOPS | 0.87 to 0.95 |
| iu8 first | 8192 | 118.7, 129.4 TOPS | 124.2, 133.2 TFLOPS | 0.96 to 0.97 |

**Result.** On gfx1100 the 8-bit WMMA rate is at most equal to the f16 rate
(0.87 to 0.97 of it here, never above). This agrees with RDNA3 issuing both
16x16x16 WMMA variants at the same rate; RDNA4 (gfx12) doubles the 8-bit rate,
but Sarek's HIP lowering does not cover gfx12 (`specs/coopmat-hip-wmma.md`).

**Consequence for the consumer.** An int8 prompt path on gfx1100 can save the
dequantise-to-f16 pass, the f16 narrowing of activations and half of the
activation bytes. It cannot save matrix-unit time. Whether that is worth the
work is a Katra measurement (its trace splits dequantisation from GEMM time), not
a SPOC one.

## 2. The operation

### 2.1 What is missing

Fragments are opaque: `CM_decl`, `CM_load`, `CM_store`, `CM_muladd`
(`spoc/ir/Sarek_ir_types.ml`). A quantised GEMM accumulates each 32-element K
block exactly in int32 and must then apply that block's scales in f32:

```
acc_f32[i][j] += (float) acc_i32[i][j] * d_w[i] * d_x[j]   -   m_w[i] * s_x[j]
```

(`d_w`, `m_w`: per-row weight scale and min of the block; `d_x`, `s_x`:
per-column activation scale and scaled sum). Today this needs a `CM_store` of the
int32 fragment to shared memory, a scalar pass, and a `CM_load` back, per K block.

### 2.2 Proposed IR

One new member of `coopmat_op`, so the existing exhaustive matches are forced to
consider it:

```ocaml
| CM_scale_add of {
    dst : string;            (* f32 accumulator fragment, read and written *)
    dst_frag : Sarek_coopmat_types.fragment;
    src : string;            (* s32 or f32 accumulator, same shape *)
    src_frag : Sarek_coopmat_types.fragment;
    row : scale;             (* r.(i) = row.buf.(row.index + i), i < m *)
    col : scale option;      (* c.(j) = col.buf.(col.index + j), j < n; None = 1 *)
    offset : (scale * scale) option;   (* (p, q): subtract p.(i) * q.(j) *)
    accumulate : bool;       (* false: overwrite dst instead of adding to it *)
  }
and scale = {buf : string; index : expr}   (* f32 buffer, global or shared *)
```

`index` is uniform across the subgroup, as for `CM_load`. `dst` and `src` may
name the same f32 fragment: with `col = None`, no `offset` and
`accumulate = false` that is the per-row rescale online softmax needs
(`acc[i][j] <- acc[i][j] * r[i]`). Typed records, no `Obj`.

### 2.3 Semantics: an exact order, no grant

For every element, with every operation correctly rounded to binary32 (round to
nearest even) and **no contraction**:

```
t1 = f32(src[i][j])          (* int -> f32 conversion; identity for f32 src *)
t2 = t1 * r[i]
t3 = t2 * c[j]               (* skipped when col = None *)
t4 = dst[i][j] + t3
dst[i][j] = t4 - p[i] * q[j] (* the product rounded, then the subtraction; skipped without offset *)
```

With `accumulate = false` the result is `t3` (minus the offset term when
present) instead of `t4`; that is the softmax rescale.

Why this is Strict and needs no `Sarek_policy` grant:

- int32 accumulation in `CM_muladd` is already exact (Strict,
  `specs/coopmat-hip-wmma.md`).
- Every step above is a single IEEE operation with a fixed order. A device that
  performs exactly these operations produces the interpreter's bits. The
  int-to-float step is exact when `|src| <= 2^24`, which holds for one MMQ block
  (at most `32 * 128 * 128 = 2^19` for 8-bit by 8-bit); larger values are still deterministic because the
  conversion is correctly rounded on every target below.
- The only way a device can differ is contraction of `t2 * c[j]` with the add,
  or of `p[i] * q[j]` with the subtraction. Each lowering therefore emits
  non-contractible operations (§3). A fused form would remove two roundings per
  element; it is not proposed, because the scale step is a few VALU operations
  per lane against the K-loop's loads and WMMAs, so there is no measured cost to
  buy back. If one appears, the fused form becomes a new named deviation in
  `Sarek_policy` with its bound (one rounding fewer per fused pair:
  `|fused - strict| <= u * (|t3| + |p[i] * q[j]|)` per element, `u = 2^-24`),
  not a silent compiler choice.

### 2.4 Interpreter reference

The interpreter keeps whole fragments per subgroup (`coopmat_key` in
`sarek/interp/Sarek_ir_interp_eval.ml`); the leader lane applies §2.3 element by
element with `Int32.to_float` followed by rounding to binary32 at each step (the
same helper the f32 path already uses). This is the oracle every backend is
compared against bit for bit.

## 3. Lowering per backend (coverage table)

| backend | lowering | status if implemented | reason |
|---|---|---|---|
| Interpreter | §2.4, leader lane, whole fragment | reference | oracle |
| HIP gfx11 | direct, registers only: accumulator element `e` of lane `l` is row `2e + h`, column `l16` (`h = l / 16`, `l16 = l mod 16`, the layout already used by `sarek_cm_load_acc_*`). Each lane loads `c[l16]` and `q[l16]` once and `r`, `p` for its 8 rows; `v_cvt_f32_i32` then `__fmul_rn`, `__fadd_rn`, `__fsub_rn` (hipcc contracts by default, so plain `*` and `+` are not enough) | executed on RX 7900 XTX, bit-identical | the only backend with a documented accumulator layout and a device here |
| Vulkan (KHR cooperative matrix) | the accumulator layout is implementation-defined, so no per-lane form. Portable route: convert the int32 fragment to an f32 accumulator-use matrix (`coopmat<float, …>(src)` is element-wise), load `r` and `c` as broadcast matrices (column-major with stride 0 for `r`, row-major with stride 0 for `c`), multiply component-wise, add. To verify on RADV before relying on it: that stride 0 is accepted and that component-wise `*` on cooperative matrices is supported; otherwise a shared-memory round trip (`coopMatStore`, scalar §2.3, `coopMatLoad`), which needs a per-subgroup scratch array declared by the lowering. The products need `precise` to forbid contraction | executed on RADV (gfx1100) | the plan's coverage decision (HIP and Vulkan executed) |
| CUDA (`nvcuda::wmma`) | the accumulator fragment's element-to-(row, column) map is not documented, so no direct form. Either a shared-memory round trip as for Vulkan, or a move of the 8-bit lowering to `mma.sync` (whose PTX fragment layouts are documented). Named `#error` in the CUDA arm until one is done | compile-only | no NVIDIA GPU; ZLUDA declines `wmma` |
| Metal | integer `src`: refused, as every integer configuration already is (no integer `simdgroup_matrix`; `no_integer_reason` in `sarek/codegen/Sarek_ir_metal_coopmat.ml`). f32 `src` (softmax rescale): threadgroup round trip, later | integer refused with the existing named reason | hardware |
| OpenCL, WGSL, PTX, Native | refused, as every `SCoopmat` statement is | refused | no cooperative matrices |

The capability gate needs nothing new for the integer case: a kernel with
`CM_scale_add` also contains the `CM_muladd` that produced `src`, whose
configuration is what the device must offer. A device that offers the
configuration but has no lowering of the new statement (CUDA, Metal f32 for now)
refuses at code generation with a named reason, never by computing it wrongly.

## 4. Fragment loads of 8-bit operands (plan item S2): no change needed

What `sarek_cm_load_*_i8` compiles to on gfx1100 (ROCm 7.2.4, `-O3`, preamble of
`sarek/codegen/Sarek_ir_cuda_coopmat_preamble.ml` unchanged):

| fragment form | global memory | shared memory |
|---|---|---|
| contiguous (`load_a_i8`: A row-major, B column-major) | one `global_load_b128` per lane | one `ds_load_b128` per lane |
| strided (`load_b_i8`: B row-major, A column-major) | 16 `global_load_u8` per lane | (same gather) |

The contiguous forms are already vector loads, with no alignment precondition in
the IR (gfx11 accepts the unaligned access; a 16-byte aligned `index + l16 *
stride` avoids split accesses). The strided form is a gather by construction; the
remedy is the layout, not the helper: store the activations token-major and load B
column-major, as `CM_load`'s `layout` field already allows. CUDA's
`load_matrix_sync` vectorises inside NVIDIA's header under its existing 32-byte
alignment check, and Vulkan's `coopMatLoad` is the driver's choice.

## 5. If the work is resumed

1. Tests first, failing before: interpreter unit tests of §2.3 on chosen values
   (a product whose contraction would change the bits; `|src| > 2^24`; an offset
   whose rounding matters), a HIP e2e test bit-identical to the interpreter over
   random int32 fragments and scales, a Vulkan e2e test, and refusal tests for
   CUDA, Metal, OpenCL, WGSL, PTX and Native.
2. IR member, interpreter, HIP lowering, Vulkan lowering (after the two RADV
   checks in §3), refusals.
3. The coverage table above in the PR, a reason for each empty cell.
