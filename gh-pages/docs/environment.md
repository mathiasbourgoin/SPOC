---
layout: page
title: Environment Variables
---

# Environment Variables

This page lists every `SAREK_*` and `SPOC_*` environment variable that the libraries and the PPX read, with the values each one accepts. Check here before relying on a value such as `true` or `0`: the variables do not share one parsing rule, and a value a variable does not recognise is usually ignored without a message.

Test and benchmark programs read further variables of their own (for example `SAREK_REQUIRE_PTX`, `SPOC_BENCH_*`); they are not part of the libraries' interface and are not listed.

## Value conventions

| Convention | Variables |
|---|---|
| Only the exact string `1` has an effect | `SPOC_DISABLE_*`, `SAREK_EXPLAIN` (any other value is off; an unknown one warns once) |
| Only the exact string `0` switches off | `SAREK_HIP_IGPU_UNCACHED`, `SAREK_DISK_CACHE` |
| A comma-separated list of names, case-insensitive, spaces and empty items ignored; an unknown item warns and is ignored | `SAREK_OPT_DISABLE` |
| `1`, `true` or `yes` (lowercase) switches on | `SAREK_PPX_DEBUG` |
| A positive number (surrounding spaces ignored, except for `SPOC_METAL_BATCH` and `SPOC_VULKAN_BATCH`); an invalid value falls back to the default without a message | `SAREK_DOMAIN_COUNT`, `SAREK_CPU_POOL_MAX_DOMAINS`, `SAREK_CPU_POOL_IDLE_MS`, `SPOC_METAL_BATCH`, `SPOC_VULKAN_BATCH` |
| A positive number; an invalid value is an error | `SPOC_VULKAN_FENCE_TIMEOUT_S` |
| Set to anything, even empty | `SPOC_GPU_STATS` |

So `SPOC_DISABLE_GPU=true` does nothing.

"Read at" says when the value is taken: **PPX** is compile time (the environment of the `dune build` that expands `[%kernel ...]`), **start** is when the library or plugin module is loaded at program start, and **launch** or **code generation** is later, while the program runs.

## Backends and devices

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SPOC_DISABLE_GPU` | `1` | unset | start, and at each plugin `init ()` | No GPU backend registers (CUDA, HIP, OpenCL, Vulkan, Metal). OpenCL CPU devices go with the OpenCL backend. Native and Interpreter stay. |
| `SPOC_DISABLE_CUDA` | `1` | unset | start, and at each `init ()` | The CUDA backends (PTX and CUDA/C) do not register. |
| `SPOC_DISABLE_HIP` | `1` | unset | start, and at each `init ()` | The HIP backend does not register. |
| `SPOC_DISABLE_OPENCL` | `1` | unset | start, and at each `init ()` | The OpenCL backend does not register. |
| `SPOC_DISABLE_VULKAN` | `1` | unset | start, and at each `init ()` | The Vulkan backend does not register. |
| `SPOC_DISABLE_METAL` | `1` | unset | start, and at each `init ()` | The Metal backend does not register. |
| `SAREK_CUDA_INCLUDE` | `:`-separated directories (empty is ignored) | unset | first CUDA/C kernel compile | Extra include directories for NVRTC, tried before `CUDA_PATH`, `CUDA_HOME`, `CUDA_ROOT` and the conventional install roots; only directories that exist are kept. Needed for kernels that include `cuda_fp16.h`. |
| `SAREK_HIP_IGPU_UNCACHED` | `0` | unset (uncached allocation on integrated GPUs) | start | `0` makes HIP allocate device buffers on integrated GPUs with plain `hipMalloc` again. The only warning is a warn-level log line (stdout, once, at the first integrated-GPU allocation) printed only when `SAREK_DEBUG` enables the `device` component; otherwise nothing signals the override. **Unsafe**: on the gfx1036 iGPU, kernel writes into such memory can be lost across host idle gaps and results are silently wrong (issue #427). It exists to reproduce that loss. Discrete GPUs always use `hipMalloc`. |
| `SPOC_METAL_BATCH` | integer >= 1 | `32` | start | Most dispatches one Metal command buffer collects before it is committed while the GPU is still busy. |
| `SPOC_VULKAN_BATCH` | integer >= 1 | `1` | start | Most dispatches one Vulkan command buffer collects before it is submitted. `1` (the default) submits and waits for every launch, and a failed launch then raises at the launch instead of at the first wait. Batching is opt-in (`32` is the measured setting) because `test_vulkan_async` caused GPU write page faults and ring resets on RADV (Sarek #504). |
| `SPOC_VULKAN_HOST_IMPORT` | `1` | unset (off) | start | Allows Vulkan `register_host` (import of host memory, `VK_EXT_external_memory_host`); `alloc_pinned` does not import and needs no opt-in. Refused on AMD drivers even when set: importing host pages was followed by a kernel oops in `amdgpu_hmm_invalidate_gfx` (Sarek #503). |
| `SPOC_VULKAN_HOST_IMPORT_AMDGPU_I_ACCEPT_KERNEL_OOPS` | `1` | unset | start | With `SPOC_VULKAN_HOST_IMPORT=1`, allows the import on AMD drivers anyway. For disposable sessions only. |
| `SPOC_VULKAN_FENCE_TIMEOUT_S` | positive number of seconds | unset (no bound) | start; a malformed value is refused when a Vulkan device is created | Bounds how long Vulkan waits for submitted work; meant as a hang detector in tests. |

Device selection is explained in [Device and backend selection](device_selection.html); the Metal batch in [Memory and transfers](memory.html); the Vulkan batch in [Launch window and fusion](performance.html); the fence bound in the [Vulkan README](https://github.com/mathiasbourgoin/Sarek/blob/main/sarek-vulkan/README.md).

## Optimisation policy: restrict and explain

SPOC applies some optimisations on its own (the launch window, epilogue fusion, the matrix-unit rewrite, the exact-narrowing proof, loop unrolling), and some only after the program has accepted a deviation from the interpreter's result (a grant, see [Matrix units](matrix.html) and [Launch window and fusion](performance.html)). The environment can **only restrict**: no variable grants precision, turns an optimisation on that the program or the backend does not allow, or accepts a deviation. Both variables are read once, when the program starts, and a restriction is one-way for the process.

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_OPT_DISABLE` | comma-separated items, below | unset | start | Takes the named optimisations away, and revokes the named grants, for the whole process. |
| `SAREK_EXPLAIN` | `1` | unset (off) | start | Writes one line on stderr for every distinct decision of every optimisation. Any other value is off; a value other than empty or `0` warns once. A reporter installed with `Sarek_policy.set_reporter` replaces the stderr one. |

### `SAREK_OPT_DISABLE`

Items are separated by commas, compared case-insensitively, and surrounding spaces and empty items are ignored (`Window, ,Unroll` works). An unknown item prints one line on stderr naming the item and the valid items, and is ignored; the other items apply.

| Item | Effect |
|---|---|
| `window` | No deferred launches and no horizontal fusion, on every backend. `Execute.Window.set_enabled` stays for programs that need the window off in part of their run. |
| `epilogue` | No epilogue fusion, on every backend. |
| `matrix` | No matrix-unit rewrite, for `Execute.run_vectors` and `Sarek_matrix.run`. A kernel the program wrote with `Sarek_gemm.Coopmat` is not a rewrite and still runs. |
| `narrowing-proof` | The f32 to f16 narrowing keeps its barrier everywhere, even where one rounding is provably equal to two. |
| `unroll` | No automatic `#pragma unroll` hint on small constant-trip loops. |
| `accumulation-order`, `operand-rounding-f16`, `operand-rounding-bf16`, `operand-rounding-bf16x3`, `single-rounding`, `cross-kernel-contraction`, `automatic-float-rewrite` | Revokes that grant for the whole process, whatever scope the program grants it at. The program's call still succeeds and is recorded, but no decision sees the grant. `SAREK_EXPLAIN` prints `grant-revoked` for it. |

There is no `all`. Other variables that used to switch these off are gone (see `CHANGES.md`): `SAREK_LAUNCH_WINDOW`, `SAREK_LAUNCH_EPILOGUE`, `SAREK_MATRIX_REWRITE`, `SAREK_AUTO_UNROLL` and `SAREK_MATRIX_REPORT` are no longer read, and setting them does nothing.

The generated source and the launch memo follow the policy: a restriction cannot make an earlier check stale.

### `SAREK_EXPLAIN`

One line per **distinct** (kernel, target, optimisation, decision): a decision that comes back after another one is not printed again (a window switched off and on per prompt chunk prints two lines, not one per switch). The memory of what was printed is bounded (4096 distinct decisions, then forgotten). Fields are separated by single spaces; spaces inside a kernel name are written `_`, and an epilogue pair `p + c` is written `p+c`.

The **target** says where the decision was taken, in one scheme:

| Target | Used by | Example |
|---|---|---|
| `<framework>:<device id>` | decisions about a device: `window`, `matrix`, `coopmat-gate` | `HIP:0`, `OpenCL:1` |
| `<framework>` | decisions about a backend: `epilogue` | `HIP`, `CUDA/PTX` |
| `gen:<generator>` | code generation, which does not know the device: `narrowing-proof`, `unroll` | `gen:cuda-c` (HIP and CUDA/C), `gen:opencl`, `gen:metal`, `gen:glsl` |

```
sarek-explain <kernel> <target> <optimisation> applied proof=<text>
sarek-explain <kernel> <target> <optimisation> applied granted=<grant>[,<grant>] reason="<reason>" [lowering="<description>"]
sarek-explain <kernel> <target> <optimisation> applied exact
sarek-explain <kernel> <target> <optimisation> refused <reason>
sarek-explain <kernel> <target> coopmat-gate applied|refused ...
sarek-explain <kernel> - policy refused grant-on-generated-name <grant>
sarek-explain <kernel> - policy refused grant-revoked <grant>
```

`<optimisation>` is `window`, `epilogue`, `matrix`, `narrowing-proof` or `unroll`; `<grant>` is one of the grant names above. `reason="..."` holds only what the program gave when it granted (each reason once, in order, joined with `; `). `lowering="..."` is the matrix rewrite's own description (lowering, variant, fragment configuration), present on the float rewrites. `coopmat-gate` is an event kind of its own, not an optimisation: it reports the accuracy gate of the **device backends** for a kernel the program wrote with `Sarek_gemm.Coopmat` (the interpreter's gate is not reported), `SAREK_OPT_DISABLE` has no name for it, and nothing restricts it. The last two lines say that a grant the program made has no effect: a `Kernel` grant on a generated name (a fused, derived or static-fusion kernel) is never counted, and a revoked grant is not obeyed; each is printed once, at the next report for that kernel, also when its decisions settled before. A reporter set with `Sarek_policy.set_reporter` is called outside the policy's lock, possibly from several domains: lines of different domains may interleave. Sample lines from a run on an AMD RX 7900 XTX (HIP) are in `CHANGES.md`.

**Revoking an accuracy grant has a visible effect.** `accumulation-order` is the closest equivalent of the old `SAREK_MATRIX_REWRITE=exact` for float paths: it refuses every float matrix-unit path (the rewrite and `Sarek_gemm.Coopmat`) and keeps `int32`. An explicit cooperative-matrix kernel the program wrote then **fails at launch** with a capability refusal that says `accumulation-order is revoked by SAREK_OPT_DISABLE`. Revoking `operand-rounding-bf16` or `-bf16x3` does the same for bfloat16 kernels. Revoking one operand rounding can make the rewrite pick another precision the program also granted (for example float16 instead of bfloat16), which is less accurate; revoke all of them to rule that out.

## Launch window and fusion

The launch window and epilogue fusion have no variable of their own. `SAREK_OPT_DISABLE=window` and `=epilogue` take them away; they are explained in [Launch window and fusion](performance.html).

## Matrix units

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_MATRIX_SCAN` | a file path | unset | PPX | Appends one line per analysed kernel and finding to the file. |

The rewrite is restricted with `SAREK_OPT_DISABLE=matrix` or `=automatic-float-rewrite` and reported with `SAREK_EXPLAIN=1`. See [Matrix units](matrix.html).

## Code generation

`SAREK_OPT_DISABLE=unroll` switches off the automatic unroll hint on small constant-trip `for` loops in the CUDA, HIP, OpenCL, Metal and GLSL generators. See [Kernel language notes](kernel_language.html).

## CPU backends

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_DOMAIN_COUNT` | a positive integer | `Domain.recommended_domain_count ()` (4 if that fails) | each time a CPU pool is created, and at each barrier-free Native launch (work split) | Number of OCaml domains used by the Native runtime and the Interpreter, before the `SAREK_CPU_POOL_MAX_DOMAINS` cap. The Native parallel pool uses the value as its worker count; without it, it uses one fewer than the domain count (at least 1). A value that is not a positive integer is ignored. Meant for tests that need more domains than cores. |
| `SAREK_CPU_POOL_MAX_DOMAINS` | a positive integer | unset (no cap) | each time a CPU pool is created, and at each barrier-free Native launch (work split) | Upper bound on the workers of each CPU pool (the Interpreter and Native block pool, the Native thread and parallel pools), so a program can keep cores for itself. `Cpu_pool_lifecycle.set_max_domains` takes precedence. A value that is not a positive integer is ignored. |
| `SAREK_CPU_POOL_IDLE_MS` | a positive integer (milliseconds) | `1000` | at each poll of the pool reaper, while a pool is live | A CPU pool (or a fission-queue launch dispatcher) that no launch has used for this long is retired: its worker domains exit, and the next launch creates it again. Idle domains still take part in every minor collection, so retiring them keeps host allocation fast. `Cpu_pool_lifecycle.set_idle_timeout` takes precedence. A value that is not a positive integer is ignored. |

## Compile cache

Vulkan (SPIR-V) and OpenCL (program binaries) keep compiled kernels on disk so that a new process does not compile them again. The files are in the `spoc` directory under `$XDG_CACHE_HOME`, or under `$HOME/.cache` when that is unset; the parent directory must already exist. The cache is used only when that directory is owned by the current user and not writable by group or others (a directory Sarek creates is mode 0700), and never from the shared temporary directory: without `XDG_CACHE_HOME` and `HOME` it is off. OpenCL entries are native device binaries that the driver loads, so a directory others can write to would let them substitute code. A refused directory prints one warning and the program compiles as if the cache were off.

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_DISK_CACHE` | `0` turns it off | unset (on) | each cache lookup and store | With `0`, no disk cache is read or written: Vulkan compiles GLSL again and OpenCL builds from source in every process. Nothing else changes. The in-process kernel caches are not affected. Metal has no disk cache of ours (not needed: OS shader cache, measured, see [Performance](performance.html#first-launch-in-a-new-process)). |
| `XDG_CACHE_HOME` | a directory | `$HOME/.cache` | each cache lookup and store | Where the `spoc` cache directory lives. Not a Sarek variable; set it to move or isolate the cache (the tests point it at a temporary directory). |

An OpenCL entry is keyed by the platform name and version, the device name and version, the driver version, the exact build-option string and the source, so a driver update or a different option set compiles again and writes a new entry. A cached binary that the driver refuses, or that fails to build, is ignored: the kernel is compiled from source and the entry is overwritten. No error reaches the program. A driver that returns no binary for a program (size 0) is never cached. An entry is a checksum followed by the binary; a truncated or damaged file fails the checksum and is treated the same way. Entries are never evicted; delete the `spoc` directory to clear them. Set `SAREK_DEBUG=device` to see each lookup (`[Cache]` lines). 

The key cannot see a driver rebuilt from new sources under the same version string: OpenCL uses `CL_DRIVER_VERSION`, so entries made by the old build are reused. Vulkan keys on the device name and the Vulkan API version, not on the shaderc version, so a new shaderc leaves old SPIR-V in use. After rebuilding or replacing such a component, delete the `spoc` cache directory or run once with `SAREK_DISK_CACHE=0`.

## Diagnostics

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_DEBUG` | a comma-separated list; components `transfer`, `kernel`, `device`, `memory`, `execute`, `all` or `*`; levels `debug`, `info`, `warn`, `error` (case-insensitive, unknown items ignored) | unset (no logging) | start | Runtime logging, per component. |
| `SAREK_DEBUG` (framework logs) | the whole value is exactly `all`, or one comma-separated item is exactly `device` (no trimming, lowercase) | unset | start | Also enables the `[Cache]` and `[Framework]` messages of the backend cache and registry. |
| `SAREK_PPX_DEBUG` | `1`, `true`, `yes` | unset | PPX | Prints the PPX's `SAREK DEBUG:` trace on stderr. |
| `SAREK_DEBUG` (PPX) | `1`, `true`, `yes` | unset | PPX | The same trace, when `SAREK_PPX_DEBUG` is not set to one of those values. |
| `SPOC_GPU_STATS` | any value | unset | start | At exit, prints device-memory statistics (current and peak bytes, allocations, frees, GC-triggered frees) on stderr. |

`SAREK_DEBUG` is read in three places with three meanings; the rows above are the three. The runtime list is described in the [`sarek.core` README](https://github.com/mathiasbourgoin/Sarek/blob/main/sarek/core/README.md).
