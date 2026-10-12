---
name: roster-spec
type: spec
status: live
feature: Cooperative matrix (SCoopmat) on Metal via simdgroup_matrix 8x8x8; scalar f16 on Metal (its prerequisite)
brief: package H item H3 (backend parity of optimisations)
date: 2026-10-12
version: 1.0.0
---

# Spec — coopmat-metal-simdgroup

Owner rule (2026-10-11): an optimisation is ported to every backend where it is
possible, and where it is not, it stays off and the reason is written down. This
spec ports `SCoopmat` to Metal and records why OpenCL has none. Companion of
`specs/coopmat-hip-wmma.md` (HIP/CUDA); the IR, the capability model and the
opt-in are unchanged.

Binding prior decisions this implements rather than re-opens:
`docs/design/f16-relaxed-accuracy.md` §2 (Metal rows: scalar f16 strict; the
f16 x f16 -> f32 `simdgroup_matrix` is a closed form), §7 slices 5 and 6 (what
MSL offers, measured), and `docs/fp-contraction-policy.md` §10.14 and §10.15.

## What exists before this change

- Metal refused every `SCoopmat` (`Sarek_ir_metal.reject_coopmat_kernel`) and
  every `uint8` buffer, and refused scalar f16 (`reject_float16_kernel`) although
  the narrowing was measured strict (0/63488 on both swept shapes).
- The Metal plugin reported `coopmat = None` ("not probed"), so a launch gate
  refused for lack of a probe.

## Coverage table (this change)

Evidence tiers: *executed* (on a named device and toolchain), *compiler-output*,
*by-construction*, *unverified*.

| backend | SCoopmat before | after | evidence tier |
|---|---|---|---|
| HIP | WMMA (gfx11) | unchanged | executed on RX 7900 XTX gfx1100, hipcc/hiprtc ROCm 7.2.4 (`test_coopmat_hip_e2e` re-run after the `Sarek_gemm.Coopmat` refactor, `H3-hip-regression-*.txt`) |
| CUDA | lowered (`nvcuda::wmma`), compile-only | unchanged | unchanged |
| Vulkan | GLSL coopmat (integer) | unchanged | unchanged |
| Metal | refused; scalar f16 refused | scalar f16 lowered; `simdgroup_matrix` 8x8x8 `f16*f16+f32->f32` lowered; every other configuration refused with a reason | codegen: compiler-output (goldens on tartarus); numerics: executed on Apple M4, macOS 15 (see Status; executed on Apple M4, macOS 15) |
| OpenCL | absent | absent, refusal names the reason (below) | by-construction (clinfo, 2026-10-12) |
| Native | not a coopmat backend | unchanged | unchanged |
| Interpreter | executes (`f16*f16+f32->f32` after the opt-in) | unchanged: it is the oracle | executed |

### Why OpenCL has none

OpenCL has `cl_intel_subgroup_matrix_multiply_accumulate`. Checked with clinfo on
2026-10-12: neither rusticl on the RX 7900 XTX nor the Intel oneAPI CPU runtime
advertises it or any other matrix extension. A lowering could not be executed
or validated on any device this project can reach, so none is written; the
refusal text (`Sarek_ir_opencl.coopmat_refusal`) says so. If a device that
advertises the extension becomes available, the lowering belongs in a new
`Sarek_ir_opencl_coopmat` and the OpenCL probe would report the configuration.

## Metal: what is lowered

Scalar `f16`: `TFloat16` maps to `half`; the f32 -> f16 narrowing is a plain
`(half)` cast with no opacity barrier, because Metal does not fuse it (0/63488
on `f16(x*1.1)` and on `f16(f16(x*1.1)+1000)`, element-wise, with a control that
reports 2912 and 620 on the same shapes, Apple M4,
`tools/probes/metal_f16_narrowing_probe.m`).

`SCoopmat`: one configuration, shape 8x8x8, subgroup scope,

| A x B + C -> D | MSL | regime |
|---|---|---|
| f16 x f16 + f32 -> f32 | `simdgroup_half8x8` x `simdgroup_half8x8` + `simdgroup_float8x8` -> `simdgroup_float8x8` | closed form: accumulator = C, then the 8 products added in index order, all in binary32 (bit-equal on 65536/65536, `tools/probes/metal_simdgroup_matrix_probe.m`) |

Statements:

| IR | MSL |
|---|---|
| `CM_decl` | `simdgroup_half8x8 f = make_filled_simdgroup_matrix<half, 8, 8>(0);` (`float` for an accumulator) |
| `CM_load` | `simdgroup_load(f, (buf + (index)), (ulong)(stride), ulong2(0, 0), transpose);` |
| `CM_store` | `simdgroup_store(f, (buf + (index)), (ulong)(stride), ulong2(0, 0), transpose);` |
| `CM_muladd` | `simdgroup_multiply_accumulate(d, a, b, c);` |

`transpose` is `false` for `Row_major` (element (r, c) at `index + r*stride + c`)
and `true` for `Col_major` (`index + c*stride + r`), which is the layout
definition of `Sarek_coopmat_types.layout`; both are executed for A, B, C and D.
The buffer may be a kernel parameter or a shared array (pointer arithmetic is
the same in the `device` and `threadgroup` address spaces). The values of
`index` and `stride` must be uniform across the SIMD-group.

### Refused, each with its reason (at code generation)

| request | reason |
|---|---|
| integer components (u8, s8, u32, s32) and `uint8` buffers | MSL has no integer `simdgroup_matrix` (the element types are `half`, `float`, `bfloat`; a closed enumeration, `docs/design/f16-relaxed-accuracy.md` §7 slice 6). There is no strict-contract route on Metal. |
| `f16 x f16 + f16 -> f16` | matches no closed-form model (65520/65536 against sequential binary16 on the M4); §5.4 recommends not admitting it |
| bf16 | `bfloat` 8x8 exists but is unmeasured; the IR's bf16 fragment loads a float32 buffer rounded to bfloat16 |
| f32 operands | `float` 8x8 exists, but its products are not exact in binary32, so no bound or oracle is derived for it (`Sarek_coopmat.has_derived_bound`) |
| any shape other than 8x8x8 | MSL offers 8x8 only. A 16x16 fragment composed of 2x2 tiles is possible (see Open decisions); it is not built |
| saturating accumulation, scope other than subgroup | no such `simdgroup_matrix` operation |
| a buffer whose element type is not the fragment's | named in the error |
| a launch block that is not whole 32-wide SIMD-groups | a `simdgroup_matrix` operation needs every lane active; checked in `Metal_plugin.generate_source` where the block is known |

## Capability

`Metal_plugin_base.capabilities` probes by compiling a kernel that uses the
lowered configuration (`Metal_coopmat.probe_source`) on the device, and reads the
SIMD-group width from the pipeline (`threadExecutionWidth`), so
`ds_subgroup_size` and `warp_size` are device facts, not constants. The result is
memoised per device. Compiles: `coopmat = Some` with the one configuration.
Does not compile (pre-Apple7 GPUs, Intel/AMD Macs): `Some` with no configuration
and the compiler's message in `ds_evidence`, so the refusal names the cause
rather than "not probed". The pure rule is `Metal_coopmat.support`, driven by
`test_metal_coopmat_probe` without a device.

## Opt-in and the interpreter

Unchanged. `Sarek_coopmat.policy_verdict` treats `f16 x f16 + f32 -> f32` as
needing `accept_relaxed_accumulation`, and the interpreter refuses without it.
On Metal the configuration is Regime A in the sense of the design document (a
named closed form), so the opt-in is stricter than the measurement requires.
*(agent decision)* It is kept: `policy_verdict` takes a configuration, not a
backend, so exempting Metal means threading the backend through the launch gate
(`Execute.check_device_capabilities`, outside this change), and one opt-in
statement per program is also what keeps a kernel's accuracy contract the same
on HIP and Metal. The interpreter evaluates the same C-first sequential binary32
order, so on Metal the result is bit-identical to it, not merely bounded.

## Exactness contract (tests)

- `f16 x f16 + f32 -> f32` on Metal: bit-identical to the interpreter and to the
  independent OCaml model "accumulator = C, then `acc := fl32(acc + a*b)` for
  `k` in index order", on random binary16 operands and full-mantissa binary32
  C (so the position of C in the sum is observable).
- Positive controls that must go red on the same GPU output: the C-last model,
  a C-dropping run, a run with B loaded in the other layout. A comparison that
  accepts any of them is a test failure.
- Tiled GEMM (`Sarek_gemm.Coopmat.run ~frag:8`, several 8x8 fragments per
  SIMD-group): bit-equal to the sequential chain over `k`, to the interpreter at
  a small shape, and the same under three tilings (dispatch-shape invariance).

## Tests

| test | where | runs |
|---|---|---|
| `test_metal_coopmat_golden` | `sarek/tests/codegen_golden` | everywhere (generator only) |
| `test_metal_coopmat_probe` | `sarek-metal/test` | everywhere (pure rule) |
| `test_coopmat_metal_e2e` | `sarek/tests/e2e`, alias `e2e-metal` | codegen part everywhere; device part on an Apple GPU |
| `test_metal_f16_narrowing_e2e` | `sarek/tests/e2e`, alias `e2e-metal` | host checks everywhere; device part on an Apple GPU |
| `bench_metal_coopmat_gemm` | `sarek-metal/test`, alias `e2e-metal-bench` | Apple GPU, information only |

## Status: executed

Both Metal paths (scalar f16 and `SCoopmat`) are lowered and were run on Apple
hardware from the source this generator emits. Neither needs an opt-in to generate. The policy opt-in
`Sarek_coopmat.accept_relaxed_accumulation` for the float configuration stays
(see above).

Evidence (Apple M4, macOS 15, ladon, 2026-10-12T05:58Z, load 3.6):

| check | result |
|---|---|
| `dune build @sarek-metal/test/runtest` | exit 0 |
| `test_coopmat_metal_e2e`: tile, 16 layouts, offset tile, 64^3 and 128x128x256 GEMM bit-equal to the C-first binary32 model and to the interpreter; all controls rejected; 16-thread block refused with the 32-wide rule | PASS. Tile 64/64 bit-equal to interpreter and model; controls C-last 39/64, C-dropping 64/64, transposed B 64/64 differ; 16 layouts; offset tile; 64^3 GEMM bit-equal (controls 3301, 4096 differ); 128x128x256 bit-equal (controls 14566, 16384) |
| `test_metal_f16_narrowing_e2e`: Sarek-generated f16, 63488 finite values, `f16(x*1.1)` and `f16(f16(x*1.1)+1000)` bit-equal to the interpreter and the model; `const` control differs on 2912 and 620 inputs (host-computed), `demoted` control red | PASS. Metal = model and Metal = interpreter 0/63488 on both shapes; `demoted` control 33088 / 7181 and `const` 2912 / 620 differ |
| `bench_metal_coopmat_gemm 7 5` (load 2.8; scalar baseline is f32 operands) | median us (GFLOP/s) [min-max], n=512: coopmat 4x4/w1 194.6 (1379) [169.0-273.2], 2x2/w2 214.0 (1254), 1x1/w4 391.8 (685), scalar 731.0 (367); n=1024: 704.0 (3050) [699.2-725.8], 891.4 (2409), 1847.4 (1162), scalar 3551.2 (605); n=2048: 5107.4 (3364) [5088.4-5205.2], 7121.2 (2413), 14532.2 (1182), scalar 27986.6 (614) |

What was verified on tartarus: the generated MSL (goldens), the capability rule
(pure test), the Interpreter, the host half of the f16 test (interpreter = model,
controls red), and HIP/CUDA unchanged (logs `H3-*`, `H3f-*` under
`spoc-pipeline-logs/pkgH/`). Hardware behaviour of the configuration also has
the Apple M4 probes of 2026-07-27 (design document §2).

### How to re-run on a Mac

From the repository root, with the opam switch active and no other job on the
machine:

```
dune build @sarek-metal/test/runtest
dune build @sarek/tests/e2e/e2e-metal --force
dune exec sarek/tests/e2e/test_coopmat_metal_e2e.exe
dune exec sarek/tests/e2e/test_metal_f16_narrowing_e2e.exe
dune exec sarek-metal/test/bench_metal_coopmat_gemm.exe -- 7 5 512 1024 2048
```

Pass: both e2e programs end with `PASS` and no `FAIL` line, every `control ...
rejected` / `(red)` line shows a non-zero difference count. Record `uptime` and
`ps` with the benchmark.

## Open decisions

- **16x16x16 on Metal.** A 16x16 fragment is four 8x8 tiles and a 16x16x16
  multiply-add is eight `simdgroup_multiply_accumulate`. It would let
  `Sarek_gemm.Coopmat` and the matrix rewrite run on Metal with their 16x16
  shape. Not built: nobody asked, and the C-first order over `k = 16` would have
  to be re-established (two chained 8-step multiply-adds give the same
  sequential chain, but that is a claim to test, not assume).
- **f32 x f32 + f32 -> f32** on `simdgroup_float8x8` is available; its numerics
  are unmeasured and `policy_verdict` has no derived bound for float operands.
- **Dropping the opt-in on Metal** (Regime A): needs the backend threaded through
  the launch gate.
