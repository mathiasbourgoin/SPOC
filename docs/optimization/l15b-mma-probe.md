# L15b tensor-core `mma` implementability probe

**Task:** `mma-probe` (roster RESEARCH, fast mode): GO/NO-GO evidence for
**L15b**, the tensor-core `mma` optimization of the L15a tiled GEMM (PR #239).
Corrected 2026-10-11 (task `zluda-mma-nvrtc13`).

**Scope.** Everything below is about **ZLUDA** (CUDA-on-ROCm) on an RX 7900 XTX.
It says which PTX matrix instructions ZLUDA's translator accepts and what they
compute there. It says nothing about NVIDIA hardware, where every form below is
a native instruction.

**Verdict: NO-GO for L15b as specified, on ZLUDA v7-preview.3.** ZLUDA does
execute some `mma.sync` forms, but the only f16 form it runs is a scalar
emulation that does not use the WMMA units:

| PTX form (one warp) | ZLUDA v7-preview.3, gfx1100 | How ZLUDA implements it |
|---|---|---|
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | **runs**; 33/128 elements inexact, worst error 0.034 × the bound | emulated: per-lane `__ockl_fdot2` dot products, no WMMA |
| `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` | **runs, bit-exact** (0/128) | gfx11: `llvm.zluda.mma.m16n8k32.s32.s8.s8.s32` (WMMA); other GPUs: dot products (`__ockl_sdot4`, scalar on ISA 10.1.3) |
| `mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32` | not translated | n/a |
| `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32` | not translated | n/a |
| `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` | not translated | n/a |
| `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` (f16 accumulate) | not translated | n/a |
| `wmma.load/mma/store`, m16n16k16 f16→f32 | not translated | n/a |
| `wmma.load/mma/store`, m16n16k16 u8→s32 | not translated | n/a |

"Not translated" means the same thing every time. `cuModuleLoadData` succeeds,
the translator logs `Unrecognized statement "<the instruction>"` and drops the
kernel, and `cuModuleGetFunction` then returns `CUDA_ERROR_NOT_FOUND` (500). In
each module the matrix instruction is the only statement rejected
(`zluda-compiler-logs.txt`). A control kernel (the same kernel with the
instruction replaced by `mov`s) loads and runs.

"Bound" is `γ_K · Σ|terms|` (C and the K products), with `γ_K = K·u/(1 − K·u)`
and `u = 2^-24`: the binary32 bound of `docs/design/f16-relaxed-accuracy.md`
§5.2. The operands are integers in [-64, 64], so every product and partial sum
is exact in binary32 (and a correctly rounded implementation returns the exact
result), while most sums exceed 2048 (88/128 for k16), so an implementation that accumulated in
binary16 would round far outside the bound. ZLUDA's emulation is not exact (the
C probe saw `2^-20` where the exact result is 0) but stays well within the
bound. Integer forms and the f16-accumulate form are held to bit-exactness:
their operands keep every partial sum exact. The two positive controls of §5.4
are not part of this probe.

The ZLUDA source agrees (zluda-src `af8a3b1`, `ptx/lib/zluda_ptx_impl.cpp`).
It defines three `mma` entry points:
- m16n8k16 f32.f16 always takes the `fallback_mma_sync_aligned` dot-product path.
- m16n8k16 f32.bf16 and m16n8k32 s32.s8 use the WMMA intrinsic on gfx11
  (ISA 11.x) and fall back to dot products on other GPUs.

The bf16 form was not probed.

### What the 2026-07-24 version of this note got wrong

It concluded "ZLUDA does not execute `mma.sync`" after testing one form,
`m16n8k8.row.col.f32.f16.f16.f32`. That form is indeed not translated, but two
other forms (m16n8k16 f32.f16 and m16n8k32 s8) run. The note's control
experiment and its reading of `CUDA_ERROR_NOT_FOUND` (kernel silently dropped at
translation) were correct, and this run confirms both.

## Environment

| Component | Value |
|-----------|-------|
| GPU | AMD Radeon RX 7900 XTX (RDNA3, `gfx1100`; has WMMA hardware) |
| Driver | ZLUDA v7-preview.3, `~/opt/zluda/libnvcuda.so` built from zluda-src `af8a3b1`, dlopen'd as `libcuda.so.1` |
| Compute capability reported by ZLUDA | **8.8** (`mma` needs sm_80+) |
| Probe | `sarek-cuda/test/test_ptx_mma_probe.ml` with `ptx_mma_forms.ml`: hand-written PTX loaded with `Cuda_api.Kernel.load_from_ptx`, one warp, no DSL or codegen involved |
| Raw output | `docs/measurements/zluda-mma-probe-2026-10-11/` (`probe-output.txt`, `zluda-compiler-logs.txt`), tartarus, 2026-10-10 22:05 UTC |

These statuses were first measured on 2026-10-10 with a stand-alone C
driver-API probe: same statuses for the six `mma` forms (61/128 elements of the
f16 form inexact with its operands); it tested the `wmma` instructions one at a
time and with other spellings (one of them, `wmma.mma…s32.s8`, is not valid
PTX), all rejected.

## The probe

The probe has one Alcotest case per form, plus the control:

- **Not translated** (the kernel does not load): on ZLUDA, or below sm_80, the
  case SKIPs and prints the driver error; a missing capability is an answer, not
  a failure. On any other driver at sm_80 or above every form is a native
  instruction, so a load failure there means the probe's own PTX is wrong: FAIL.
- **Runs**: the case compares every output element with a CPU reference on random
  integer operands packed into the PTX ISA fragment layouts. It FAILS if an
  integer or f16-accumulate form is not bit-exact or an f32-accumulate form
  exceeds the bound. It prints the
  number of inexact elements and the worst error/bound ratio.
- **Loads but does not launch**: FAIL. The control failing to run is a FAIL too.
- On a device whose name contains `ZLUDA`, the case prints the status recorded
  above next to the observed one. A difference is flagged but does not fail, so
  a ZLUDA upgrade shows up as `** DIFFERS from the record **` in the log
  instead of a red suite.

Each kernel declares the lowest PTX ISA version that admits the device's SM
(`load_from_ptx` rewrites `.target` to it): 7.0 for sm_80, 7.1 for sm_86, 7.8 for
sm_89/sm_90, 8.6 for sm_100, 8.7 for sm_120, 9.0 for SMs not listed (including
the sm_88 ZLUDA reports, which it parses).

The wmma forms use all-ones A and B, so every D element is 16 whatever the
fragment layout. Their spellings are the ones nvrtc 13.0 emits for
`nvcuda::wmma`.

Mutation check (`mutation-cd-layout.txt`): with the C/D fragment layout of the
reference deliberately transposed, both forms that run FAIL (f16: 79/128
elements wrong, worst error 1.37e6 × the bound; s8: mismatches) and the control
still passes.

Not verified: that NVIDIA `ptxas` accepts every generated kernel. This machine
has no CUDA toolkit. The 2026-07-24 run assembled the m16n8k8 form with
ptxas 13.3 for sm_80, sm_86 and sm_88.

## Consequence for L15b

L15b was specified as an f16 tensor-core GEMM through `mma`/`wmma` PTX. On ZLUDA:

- the f16→f32 form that runs is the dot-product emulation and does not use the
  matrix units, which L15b exists to reach. Inferred from the ZLUDA source; not
  benchmarked;
- the m16n8k8 f16 form, the f16-accumulate form and every `wmma` form are
  rejected. Emitting a rejected form gives a hard `CUDA_ERROR_NOT_FOUND` when
  the kernel is looked up after module load, not a slow path.

So L15b stays NO-GO on PTX-through-ZLUDA. On this GPU the HIP backend reaches the
matrix units directly: `SCoopmat` lowers to RDNA3 WMMA there (PR #430,
`specs/coopmat-hip-wmma.md`). That, not PTX, is the route to them on AMD.

The s8 form is the exception: ZLUDA maps it to WMMA on gfx11, and the result is
bit-exact. An int8 PTX path is therefore implementable on ZLUDA, but only for
signed s8; the u8 and satfinite forms are not translated.

## Re-probe criteria

Run the probe again in either case:

1. **ZLUDA is upgraded** past v7-preview.3. Look for `DIFFERS from the record`.
   A form that starts to load is checked numerically at once, with no code
   change.
2. **A real NVIDIA sm_80+ device** is available. Every form is expected to run;
   one that does not load FAILs there (the probe's PTX would be at fault).
   The probe does not print the ZLUDA record lines there.

## Reproduce

```
LD_LIBRARY_PATH=$HOME/opt/zluda:$LD_LIBRARY_PATH \
  dune exec sarek-cuda/test/test_ptx_mma_probe.exe -- -v
```

ZLUDA's per-module translation log (`Unrecognized statement ...`) comes from its
tracing shim:

```
ZLUDA_CUDA_LIB=$HOME/opt/zluda/libcuda.so LD_LIBRARY_PATH=$HOME/opt/zluda/trace:$HOME/opt/zluda \
  ZLUDA_LOG_DIR=/some/dir ./_build/default/sarek-cuda/test/test_ptx_mma_probe.exe
```
