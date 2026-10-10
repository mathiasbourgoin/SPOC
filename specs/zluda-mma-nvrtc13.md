---
name: roster-spec
type: spec
status: live
feature: Per-form ZLUDA mma probe; CUDA 13 libnvrtc soname
brief: ZLUDA cooperative-matrix investigation, 2026-10-10 (tartarus, /mnt/ssd-external-2to/zluda-coopmat-probe-2026-10-10)
date: 2026-10-11
version: 1.0.0
---

# Spec — zluda-mma-nvrtc13

Two small corrections. Decisions not dictated by the brief were made by the
agent under the owner's delegation and are marked *(agent decision)*.

## Findings

| ID | Finding | Evidence |
|---|---|---|
| F-1 | `l15b-mma-probe.md` and `test_ptx_mma_probe.ml` generalise "ZLUDA does not execute `mma.sync`" from one form (m16n8k8 f32.f16). On ZLUDA v7-preview.3 (af8a3b1, gfx1100) m16n8k16 f32.f16 runs (emulated with `__ockl_fdot2`, inexact) and m16n8k32 s8 runs bit-exact (WMMA on gfx11); u8, satfinite, m16n8k8 f32.f16, m16n8k16 f16-accumulate and all `wmma.*` are rejected (`Unrecognized statement`, then `cuModuleGetFunction` -> 500). | C probe of 2026-10-10; this branch's probe `docs/measurements/zluda-mma-probe-2026-10-11/`; zluda-src `ptx/lib/zluda_ptx_impl.cpp` |
| F-2 | `Cuda_nvrtc` dlopens `libnvrtc.so`, `.so.12`, `.so.11`, macOS, `nvrtc64_120_0.dll`. CUDA 13's soname is `libnvrtc.so.13`, so an install without the development symlink reports no NVRTC. | `sarek-cuda/Cuda_nvrtc.ml` before this change |

## Requirements

- **FR-001** The probe has one test case per instruction form: the six `mma.sync`
  forms of F-1 and two `wmma` load/mma/store sequences (f16→f32, u8→s32).
- **FR-002** A form whose kernel does not load SKIPs with the driver error
  printed on ZLUDA or below sm_80, and FAILs on any other driver at sm_80+ (the
  form is native there, so the probe's PTX would be wrong). Kernels declare the
  lowest PTX ISA version admitting the device's SM.
- **FR-003** A form that runs is checked element by element against a CPU
  reference on random integer operands in the PTX ISA fragment layouts: integer
  and f16-accumulate forms bit-exact (operands keep partial sums exact),
  f32-accumulate forms within `γ_K · Σ|terms|`, u = 2^-24
  (`f16-relaxed-accuracy.md` §5.2), with operands whose sums exceed 2048 so that
  binary16 accumulation would leave the bound. The §5.4 positive controls are
  out of scope. Out of bound -> FAIL. Inexact count printed.
- **FR-004** A kernel that loads but fails to launch -> FAIL.
- **FR-005** A control kernel (the first form's kernel with the instruction
  replaced by `mov`s) must load and run, else FAIL.
- **FR-006** On a device whose name contains `ZLUDA`, the status recorded for
  v7-preview.3 is printed with the observed one; a difference is reported, not
  failed *(agent decision: a ZLUDA upgrade must not turn the suite red)*.
- **FR-007** The doc states the per-form status, that it is ZLUDA behaviour (not
  NVIDIA), what the previous version got wrong, and the L15b consequence.
- **FR-008** `Cuda_nvrtc.nvrtc_candidates` lists `libnvrtc.so`, `.so.13`, `.so.12`,
  `.so.11`, `libnvrtc.dylib`, `nvrtc64_130_0.dll`, `nvrtc64_120_0.dll` in that
  order; the loader takes the first that opens; the not-found error lists the
  same names. Newest soname first after the development symlink *(agent
  decision: a host with two toolkits usually runs the newer driver; the converse
  fails loudly at module load)*. The CUDA 13 DLL is added for the same reason
  *(agent decision; not testable here)*.
- **FR-009** A pure unit test covers FR-008 with an injected opener (no libnvrtc).

## Scenarios

- GIVEN ZLUDA v7-preview.3 on the RX 7900 XTX WHEN the probe runs THEN control,
  m16n8k16 f32.f16 and m16n8k32 s8 pass, the other six SKIP, and every record
  line matches.
- GIVEN a reference with a wrong C/D layout WHEN the probe runs on ZLUDA THEN both
  executed forms FAIL (mutation check, FR-003 is not vacuous).
- GIVEN no CUDA driver WHEN the probe runs THEN every case SKIPs.
- GIVEN an opener that only opens `libnvrtc.so.13` WHEN the loader searches THEN
  it tries `libnvrtc.so`, then returns `libnvrtc.so.13`.
- GIVEN an opener that opens nothing THEN the search tries every candidate and
  returns `None`.

## Out of scope

- `ci/assert-toolchain.sh` mirrors the candidate list in C. CI installs CUDA 12.6,
  where `.so.12` resolves, so the mirror still passes. Changing CI scripts needs
  owner approval (escalation rule); left as a follow-up.
- Validating the generated PTX with NVIDIA `ptxas` (no toolkit on tartarus).
- The bf16 m16n8k16 form ZLUDA also defines.
- Freeing the probe's device buffers, and the pre-existing module leak in
  `Cuda_api.load_module_from_ptx` when `cuModuleGetFunction` fails (review L1).
- `Cuda_nvrtc.ml` exceeds 500 lines (pre-existing).
