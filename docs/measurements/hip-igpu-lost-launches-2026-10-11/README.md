# Kernel results lost across host idle gaps on the Ryzen 9 7950X iGPU (gfx1036), issue #427

Machine: tartarus (7950X iGPU = hip:1, RX 7900 XTX = hip:0), ROCm 7.2.4, Linux 7.1.2-3-cachyos,
2026-10-11. The host was loaded by other agents (load average 50-170), so rates move; compare
only rows taken together. Every number below has a raw log in this directory, except the `lost3` counts marked as reported.

## What happens

On the iGPU, dirty device-memory writes are lost across host idle gaps. Kernels run (the
reviewer counted every launch with `atomicAdd_system` on fine-grained host memory, and
host-memory outputs were intact), but data a kernel wrote into `hipMalloc` memory before a
host pause is gone when it is read back: the lost range is the prefix up to the launch before
the last pause (`reviewer_lost3.hip.cpp`). Data written by `hipMemcpy` host-to-device was not
affected in our runs. The RX 7900 XTX never loses anything.

This is the upstream bug https://gitlab.freedesktop.org/drm/amd/-/work_items/5923
("[gfx1036] GPU memory writes vanish after idle with agent-scope dispatch fences"): dirty L2
lines are dropped when the GPU idles before a system-scope release; GFX clock gating is
implicated there. GFXOFF is not the mechanism: toggling it through debugfs changed nothing
(`gfxoff-test.txt`: 482/482/483 of 2691 lost with GFXOFF on, 478/478/418 with it off). The
same bug in AMDGPU.jl, worked around with `AMD_OPT_FLUSH=0`:
https://github.com/JuliaGPU/AMDGPU.jl/pull/1119.

## The fix in SPOC

`Hip_api.Memory` allocates every device buffer on an integrated HIP device
(`hipDeviceAttributeIntegrated`) with `hipExtMallocWithFlags(..., hipDeviceMallocUncached)`.
Discrete devices keep `hipMalloc`. `SAREK_HIP_IGPU_UNCACHED=0` switches it off (UNSAFE; for
before/after runs). `AMD_OPT_FLUSH=0` is not used: it is process-wide and would also slow the
dGPU. The rule "every integrated device" over-covers (any APU, not only gfx1036), which is
safe, and the measured cost is below. Detecting by the integrated flag does not depend on
`gcnArchName`, which `HSA_OVERRIDE_GFX_VERSION` can change.

Downstream hazard: a buffer a kernel writes and that then sits across an idle gap on an
integrated GPU (KV caches, intermediate activations) can lose data if it is allocated with
plain `hipMalloc`. Allocate it through SPOC (as the fix does) or uncached.

## Regression check

`make stress-427` runs `sarek-hip/test/stress_hip_staged_launches.exe` once per HIP device
(300 launches, 20 ms host pauses, `SPOC_STRESS_GC_PRESSURE=sleep`) and fails on any bad
slice. It is hand-run only and not part of `dune runtest` or CI. Do not loop it, raise the run
count or run it alongside other GPU work on the iGPU: heavy iGPU queue load hard-reset
tartarus on 2026-10-11 ("data fabric sync flood").

## Evidence

| File | What it is |
|---|---|
| `repro427.hip.cpp` | the issue's loop in plain HIP: 300x (host write, `hipMemcpy` HtoD, kernel into its own slice); modes 0-5 in its header |
| `idle_gap.hip.cpp` | sleep, one launch, sync, read: no copies in the loop |
| `idle_gap_data.hip.cpp` | tag written by a kernel or by `hipMemcpy`, gap, read back; `default` vs `uncached` allocation; `batch-*` writers |
| `reviewer_lost3.hip.cpp` | the reviewer's repro that counts launches and separates device from host memory |
| `uncached_bandwidth.hip.cpp` | cost of uncached memory |
| `plain-hip-runs.txt`, `idle-gap-env-runs-plain-hip.txt`, `idle-gap-data-runs.txt`, `uncached-bandwidth.txt`, `gfxoff-test.txt` | raw output of the above |
| `spoc-stress-runs-before-fix.txt`, `spoc-stress-before-after.txt` | raw SPOC `stress_hip_staged_launches` runs (`SAREK_HIP_IGPU_UNCACHED=0` = before) |

Build the C files with `hipcc -O1 --offload-arch=gfx1036 --offload-arch=gfx1100 FILE -o NAME`.

**Data written by a kernel is lost, data written by hipMemcpy is not** (`idle_gap_data`,
iGPU, 200 batches of 300 writes, a 20 ms sleep before about 5% of the writes,
`idle-gap-data-runs.txt`):

| Writer | hipMalloc | uncached |
|---|---|---|
| kernel | 186 of 200 batches lost words (47963 of 60000 words) | 0 of 200 |
| hipMemcpy HtoD | 0 of 200 | 0 of 200 |

The dGPU: 0 of 100 batches for both writers. With one tag per trial the loss is rare (0-1
of 1000), so it needs many dirty lines before the gap.

**Plain HIP copy + launch loop** (`plain-hip-runs.txt`, iGPU, runs of 300 iterations):
mode 0 6 of 100 runs; mode 5 (20 ms sleep before about 5% of iterations) 100 of 100; the
dGPU 0 of 50 in each mode run on it (0, 3, 4 and 5). Earlier runs under heavier load (`spoc-stress-runs-before-fix.txt`
for SPOC, `idle-gap-env-runs-plain-hip.txt` for the env switches) gave higher rates: the
rate depends on host load, the pause dependence does not. `HIP_FORCE_DEV_KERNARG=1`,
`HSA_ENABLE_INTERRUPT=0`, `AMD_SERIALIZE_KERNEL=3` and `HSA_ENABLE_SDMA=0` did not remove it.

**SPOC stress, before and after** (`spoc-stress-before-after.txt`, 300 iterations per run,
iGPU, `SAREK_LAUNCH_WINDOW=0`): see the file for the final strict runs. With
`SPOC_STRESS_GC_PRESSURE=sleep`, 50 of 50 runs were bad with hipMalloc (twice) and 0 of 50
(twice) uncached. The coordinator's own `lost3` measurement (`reviewer_lost3.hip.cpp`, reported; its raw
output was not kept): hipMalloc lost 430 and 445 of 2691, fine-grained 470 and 471, uncached
0 and 0.

**Cost** (`uncached-bandwidth.txt`, 512 MiB, 5 reps, iGPU, load average 82): copy
22-26 GB/s with hipMalloc vs 20-24 uncached (mean 24.6 vs 22.5 GB/s, about 8-9% slower;
measured under load, a quiet-host re-measure is pending);
read-only sweep 23-26 vs 23-27 GB/s. Kernels with heavy cache reuse (GEMM) were not measured.

## Ruled out

SPOC finalisers and kernarg frees (keeping every kernarg block alive changed nothing), the
launch window, the staging ring, device switching, GFXOFF, SDMA, `HIP_FORCE_DEV_KERNARG`,
`HSA_ENABLE_INTERRUPT`; serialisation reduces the loss but does not remove it.
