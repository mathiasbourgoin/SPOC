# sarek-hip - AMD ROCm/HIP backend

`sarek-hip` (opam package `sarek-hip`, library `sarek-hip.plugin`, framework name `HIP`) runs Sarek kernels on AMD GPUs through ROCm. It generates CUDA-C source (the CUDA/C generator is reused unchanged, since HIP C++ accepts the subset Sarek emits), compiles it at run time with hiprtc into a gfx code object, and launches it with `hipModuleLaunchKernel`. It does not go through ZLUDA. Use it for AMD GPUs; of the backends in this repository it is the one where the [launch window and fusion](../gh-pages/docs/performance.md) are on by default.

## Requirements

- An AMD GPU supported by ROCm, with the ROCm runtime installed. Sarek loads `libamdhip64.so` and `libhiprtc.so` (versioned `.7` and `.6` names are tried too) at run time, not at build time, so ROCm is not needed to compile Sarek (`Hip_bindings.ml:25-56`, `Hip_rtc.ml:106-125`). The libraries are opened, and `hipInit` is called, at program start whenever `sarek-hip.plugin` is linked, unless HIP is disabled (`Hip_plugin.ml:135` calls `Hip_api.is_available`, `Hip_api.ml:723-730`).
- The backend registers only if both libraries load and at least one device is present (`Hip_api.ml:724-730`). Otherwise it is silently absent from the device list.
- Tested with ROCm 7.2 (7.2.4) on an RX 7900 XTX (gfx1100, RDNA3) and on the Ryzen 9 7950X integrated GPU (gfx1036). Other ROCm versions and GPUs are untested.
- The OCaml side needs `ctypes` and `ctypes-foreign` (0.24 or later). The library is marked optional in `dune`: if `ctypes-foreign` is missing, the package is skipped instead of failing the build.

## Build and use

Build the package with the rest of the repository:

```bash
dune build sarek-hip
```

In your own project, add `sarek-hip.plugin` to `libraries` and call `Sarek_hip.Hip_plugin.init ()` before `Device.init`. The complete stanza and the other backends' `init` calls are in [Getting Started](../gh-pages/docs/getting_started.md#using-sarek-in-a-dune-project). To restrict a program to HIP, pass `~frameworks:["HIP"]` to the first `Device.init`.

`Device.best ()` prefers CUDA, then OpenCL, then the first device found (`Device.ml:143-149`). On a machine with both OpenCL and HIP, select the HIP device yourself, for example by filtering `Device.init ()` on `framework = "HIP"`.

## Check what is detected

```bash
dune exec -- sarek-device-info
```

lists every device with its backend. On the development machine (RX 7900 XTX, Ryzen 9 7950X) HIP reports two devices (excerpt: each entry also shows memory, thread limits and features):

```
[0] AMD Radeon RX 7900 XTX
    Backend: HIP
    Compute Capability: 11.0
[1] AMD Ryzen 9 7950X 16-Core Processor
    Backend: HIP
    Compute Capability: 10.3
```

The second entry is not the CPU cores: it is the processor's integrated GPU (gfx1036, compute capability 10.3), which ROCm reports under the processor's name. Expect such an entry on any AMD CPU that has an integrated GPU and a ROCm runtime that sees it. The same GPU also appears under OpenCL and Vulkan when those backends are present, so one physical device can be listed several times (see [Device & Backend Selection](../gh-pages/docs/device_selection.md)).

## Disable HIP

| Variable | Effect |
|---|---|
| `SPOC_DISABLE_HIP=1` | HIP does not register |
| `SPOC_DISABLE_GPU=1` | no GPU backend registers, HIP included |

Both must be set to exactly `1` (`Hip_shared.ml:51-53`). Set `SPOC_DISABLE_HIP=1` when a ROCm runtime misbehaves or to compare against OpenCL and Vulkan on the same GPU.

## What runs

- Kernels from `[%kernel ...]` through `Execute.run` and `Execute.run_vectors`, including `float16` vectors and values.
- Streams, events, ranged copies, device-only vectors, pinned memory (`hipHostMalloc`), `register_host` (`hipHostRegister`) and `stage_to_device`; see [Memory and transfers](../gh-pages/docs/memory.md).
- Cooperative-matrix statements on gfx11, below.
- Not supported: Structure-of-Arrays (SoA) kernel arguments. A request for them is refused rather than ignored (`Hip_plugin.ml:38`).

## Defaults that differ from other backends

| Behaviour | HIP | OpenCL, Vulkan | Others |
|---|---|---|---|
| Launch window (deferred launches, horizontal fusion) | on | on | off (CUDA, Metal) or never (CPU) |
| Epilogue fusion | on | off | off |
| [Matrix rewrite](../gh-pages/docs/matrix.md) of GEMM-shaped kernels | on for integer products; float products only after the opt-ins in `Sarek_coopmat` | off | off (the Interpreter only through an explicit `Sarek_matrix.run`) |

Epilogue fusion needs no grant on HIP (and Metal) because hiprtc is called with `-ffp-contract=off` placed after any option you add, so a merged kernel cannot be contracted into a different result (`Hip_rtc.ml:264-271`; policy in [docs/fp-contraction-policy.md](../docs/fp-contraction-policy.md)). Turn either off with `SAREK_OPT_DISABLE=window` and `SAREK_OPT_DISABLE=epilogue` (comma-separated, one-way); on other backends fusion needs the `Cross_kernel_contraction` grant. The consequences, including errors surfacing in a later call, are in [Launch window and fusion](../gh-pages/docs/performance.md).

## Matrix units (gfx11 WMMA)

On gfx11 (RDNA3) devices with a 32-wide wavefront, cooperative-matrix statements (`SCoopmat`) of shape 16x16x16 are lowered to the `__builtin_amdgcn_wmma_*_16x16x16_*_w32` instructions. The probe offers these configurations (`Hip_coopmat.ml:43-54`):

| A and B | Accumulator |
|---|---|
| `float16` | `float32` |
| `bfloat16` | `float32` |
| `uint8` or `int8` (any mix) | `int32` or `uint32` |

Every other device is probed as having no configuration and a kernel using cooperative matrices is refused with a message naming the rule: that includes gfx10 and the gfx1036 integrated GPU, gfx12 and CDNA, because their matrix instructions are not lowered (`Hip_coopmat.ml:55-80`). Validation on hardware is recorded for gfx1100 (RX 7900 XTX) in `specs/coopmat-hip-wmma.md`.

## Warp shuffles

`warp_shuffle`, `warp_shuffle_down`, `warp_shuffle_up`, `warp_shuffle_xor`
(`int32`, `float32`, `float64`), `warp_size` and `lane_id` are lowered to
`__shfl`, `__shfl_down`, `__shfl_up` and `__shfl_xor` with `width = warpSize`
(the generator is shared with CUDA C and selects its dialect with
`__HIP_PLATFORM_AMD__`). The width is the device's own (32 on RDNA, 64 on
GCN and CDNA); a delta or mask outside `[0, warpSize)` is clamped to 0, which
returns the lane's own value, as the contract of `sarek/sarek/BSP.md` says. A
kernel that shuffles must be launched with a block that is a multiple of the
width. Validated on gfx1100 (wave32): `test_warp_shuffle_parity`,
`test_warp_shuffle_reduce`. Not run on a 64-wide device.

## Tests

```bash
dune build @sarek-hip/test/e2e-hip
```

runs the hardware tests; most skip themselves when no HIP device is present (`test_hip_backend.ml:139-142`). `dune runtest` runs only the tests that need no device (`test_hip_soa_refusal`, `test_hip_coopmat_probe`) and the cross-backend end-to-end tests under `sarek/tests/e2e/`, which use every device that registers. The benchmarks `sarek-hip/test/bench_*.ml` run under the separate `e2e-hip-bench` alias.

Measurements against OpenCL and Vulkan on the RX 7900 XTX: [docs/benchmarks/hip-vs-opencl-vulkan-2026-07-24.md](../docs/benchmarks/hip-vs-opencl-vulkan-2026-07-24.md).

## Related

- [Memory and transfers](../gh-pages/docs/memory.md), [Launch window and fusion](../gh-pages/docs/performance.md)
- [Device & Backend Selection](../gh-pages/docs/device_selection.md)
- CUDA code on AMD through ZLUDA instead: see "AMD GPUs" in the [top-level README](../README.md#amd-gpus).
