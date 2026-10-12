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
| Only the exact string `1` has an effect | `SPOC_DISABLE_*`, `SAREK_MATRIX_REPORT` |
| Only the exact string `0` switches off | `SAREK_AUTO_UNROLL`, `SAREK_HIP_IGPU_UNCACHED`, `SAREK_DISK_CACHE` |
| Exactly `0`, `false`, `off` or `no` (lowercase) switches off; anything else leaves it on | `SAREK_LAUNCH_WINDOW`, `SAREK_LAUNCH_EPILOGUE` |
| `1`, `true` or `yes` (lowercase) switches on | `SAREK_PPX_DEBUG` |
| A mode name, case-insensitive; an unknown value warns once | `SAREK_MATRIX_REWRITE` |
| A positive number (surrounding spaces ignored, except for `SPOC_METAL_BATCH` and `SPOC_VULKAN_BATCH`); an invalid value falls back to the default without a message | `SAREK_DOMAIN_COUNT`, `SAREK_CPU_POOL_MAX_DOMAINS`, `SAREK_CPU_POOL_IDLE_MS`, `SPOC_METAL_BATCH`, `SPOC_VULKAN_BATCH` |
| A positive number; an invalid value is an error | `SPOC_VULKAN_FENCE_TIMEOUT_S` |
| Set to anything, even empty | `SPOC_GPU_STATS` |

So `SPOC_DISABLE_GPU=true` and `SAREK_LAUNCH_WINDOW=OFF` do nothing.

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
| `SPOC_VULKAN_BATCH` | integer >= 1 | `32` | start | Most dispatches one Vulkan command buffer collects before it is submitted. `1` submits and waits for every launch, and a failed launch then raises at the launch instead of at the first wait. |
| `SPOC_VULKAN_FENCE_TIMEOUT_S` | positive number of seconds | unset (no bound) | start; a malformed value is refused when a Vulkan device is created | Bounds how long Vulkan waits for submitted work; meant as a hang detector in tests. |

Device selection is explained in [Device and backend selection](device_selection.html); the Metal batch in [Memory and transfers](memory.html); the Vulkan batch in [Launch window and fusion](performance.html); the fence bound in the [Vulkan README](https://github.com/mathiasbourgoin/Sarek/blob/main/sarek-vulkan/README.md).

## Launch window and fusion

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_LAUNCH_WINDOW` | `0`, `false`, `off`, `no` turn it off | on for HIP, OpenCL, Vulkan, CUDA and Metal; never on CPU devices | start | Switches the launch window (deferred launches and horizontal fusion) off for all backends. |
| `SAREK_LAUNCH_EPILOGUE` | `0`, `false`, `off`, `no` turn it off | on for HIP only | start | Switches epilogue fusion off. The variable cannot turn it on for another backend. |

Both are explained in [Launch window and fusion](performance.html).

## Matrix units

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_MATRIX_REWRITE` | `off`, `0`, `false`, `no`; `exact`; `auto`, `on`, `1`, `true`, `yes` (any case); other values warn once on stderr and mean `auto` | `auto` (unset or empty) | start | Mode of the automatic matrix-unit rewrite in `Execute.run_vectors`: none, `int32` only, or `int32` plus the float products the program has opted into. |
| `SAREK_MATRIX_REPORT` | `1` | unset | start | Prints each distinct rewrite decision once on stderr. |
| `SAREK_MATRIX_SCAN` | a file path | unset | PPX | Appends one line per analysed kernel and finding to the file. |

A value such as `0` or `OFF` switches the rewrite off. See [Matrix units](matrix.html).

## Code generation

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_AUTO_UNROLL` | `0` turns it off | on | code generation (the generated source is memoised per kernel, so set it before the kernel's first launch) | Switches off the automatic unroll hint on small constant-trip `for` loops in the CUDA, HIP, OpenCL, Metal and GLSL generators. |

See [Kernel language notes](kernel_language.html).

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
| `SAREK_DISK_CACHE` | `0` turns it off | unset (on) | each cache lookup and store | With `0`, no disk cache is read or written: Vulkan compiles GLSL again and OpenCL builds from source in every process. Nothing else changes. The in-process kernel caches are not affected. |
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
