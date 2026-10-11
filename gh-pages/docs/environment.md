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
| Only the exact string `0` switches off | `SAREK_AUTO_UNROLL` |
| Exactly `0`, `false`, `off` or `no` (lowercase) switches off; anything else leaves it on | `SAREK_LAUNCH_WINDOW`, `SAREK_LAUNCH_EPILOGUE` |
| `1`, `true` or `yes` (lowercase) switches on | `SAREK_PPX_DEBUG` |
| A mode name, case-insensitive; an unknown value warns once | `SAREK_MATRIX_REWRITE` |
| A positive number; an invalid value falls back to the default without a message | `SAREK_DOMAIN_COUNT`, `SPOC_METAL_BATCH` |
| A positive number; an invalid value is an error | `SPOC_VULKAN_FENCE_TIMEOUT_S` |
| Set to anything, even empty | `SPOC_GPU_STATS` |

So `SPOC_DISABLE_GPU=true` and `SAREK_LAUNCH_WINDOW=OFF` do nothing.

"Read at" says when the value is taken: **PPX** is compile time (the environment of the `dune build` that expands `[%kernel ...]`), **start** is when the library or plugin module is loaded at program start, and **launch** or **code generation** is later, while the program runs.

## Backends and devices

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SPOC_DISABLE_GPU` | `1` | unset | start, and at each plugin `init ()` | No GPU backend registers (CUDA, HIP, OpenCL, Vulkan, Metal). Native and Interpreter stay. |
| `SPOC_DISABLE_CUDA` | `1` | unset | start, and at each `init ()` | The CUDA backends (PTX and CUDA/C) do not register. |
| `SPOC_DISABLE_HIP` | `1` | unset | start, and at each `init ()` | The HIP backend does not register. |
| `SPOC_DISABLE_OPENCL` | `1` | unset | start, and at each `init ()` | The OpenCL backend does not register. |
| `SPOC_DISABLE_VULKAN` | `1` | unset | start, and at each `init ()` | The Vulkan backend does not register. |
| `SPOC_DISABLE_METAL` | `1` | unset | start, and at each `init ()` | The Metal backend does not register. |
| `SAREK_CUDA_INCLUDE` | `:`-separated directories (empty is ignored) | unset | first CUDA/C kernel compile | Extra include directories for NVRTC, tried before `CUDA_PATH`, `CUDA_HOME`, `CUDA_ROOT` and the conventional install roots; only directories that exist are kept. Needed for kernels that include `cuda_fp16.h`. |
| `SPOC_METAL_BATCH` | integer >= 1 | `32` | start | Most dispatches one Metal command buffer collects before it is committed while the GPU is still busy. |
| `SPOC_VULKAN_FENCE_TIMEOUT_S` | positive number of seconds | unset (no bound) | start; a malformed value is refused when a Vulkan device is created | Bounds how long Vulkan waits for submitted work; meant as a hang detector in tests. |

Device selection is explained in [Device and backend selection](device_selection.html); the Metal batch in [Memory and transfers](memory.html); the fence bound in the [Vulkan README](https://github.com/mathiasbourgoin/Sarek/blob/main/sarek-vulkan/README.md).

## Launch window and fusion

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_LAUNCH_WINDOW` | `0`, `false`, `off`, `no` turn it off | on | start | Switches the launch window (deferred launches and horizontal fusion) off for all backends. |
| `SAREK_LAUNCH_EPILOGUE` | `0`, `false`, `off`, `no` turn it off | on for HIP only | start | Switches epilogue fusion off. The variable cannot turn it on for another backend. |

Both are explained in [Launch window and fusion](performance.html).

## Matrix units

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_MATRIX_REWRITE` | `off`, `0`, `false`, `no`; `exact`; `auto`, `on`, `1`, `true`, `yes` (any case); other values warn once on stderr and mean `auto` | `auto` (unset or empty) | start | Mode of the automatic matrix-unit rewrite in `Execute.run_vectors`: none, `int32` only, or `int32` plus the float products the program has opted into. |
| `SAREK_MATRIX_REPORT` | `1` | unset | start | Prints each distinct rewrite decision once on stderr. |
| `SAREK_MATRIX_SCAN` | a file path | unset | PPX | Appends one line per analysed kernel and finding to the file. |

The accepted values of `SAREK_MATRIX_REWRITE` are those after [#440](https://github.com/mathiasbourgoin/Sarek/pull/440). Before it only the exact strings `off` and `exact` were recognised, and every other value, including `0` and `OFF`, silently meant `auto`. See [Matrix units](matrix.html).

## Code generation

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_AUTO_UNROLL` | `0` turns it off | on | code generation (the generated source is memoised per kernel, so set it before the kernel's first launch) | Switches off the automatic unroll hint on small constant-trip `for` loops in the CUDA, HIP, OpenCL, Metal and GLSL generators. |

See [Kernel language notes](kernel_language.html).

## CPU backends

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_DOMAIN_COUNT` | a positive integer | `Domain.recommended_domain_count ()` (4 if that fails) | first use of a CPU pool | Number of OCaml domains used by the Native runtime and the Interpreter. The Native parallel pool uses the value as its worker count; without it, it uses one fewer than the domain count (at least 1). A value that is not a positive integer is ignored. Meant for tests that need more domains than cores. |

## Diagnostics

| Variable | Accepted values | Default | Read at | Effect |
|---|---|---|---|---|
| `SAREK_DEBUG` | a comma-separated list; components `transfer`, `kernel`, `device`, `memory`, `execute`, `all` or `*`; levels `debug`, `info`, `warn`, `error` (case-insensitive, unknown items ignored) | unset (no logging) | start | Runtime logging, per component. |
| `SAREK_DEBUG` (framework logs) | the list contains `all` or `device` (exact, lowercase) | unset | start | Also enables the `[Cache]` and `[Framework]` messages of the backend cache and registry. |
| `SAREK_PPX_DEBUG` | `1`, `true`, `yes` | unset | PPX | Prints the PPX's `SAREK DEBUG:` trace on stderr. |
| `SAREK_DEBUG` (PPX) | `1`, `true`, `yes` | unset | PPX | The same trace, when `SAREK_PPX_DEBUG` is not set to one of those values. |
| `SPOC_GPU_STATS` | any value | unset | start | At exit, prints device-memory statistics (current and peak bytes, allocations, frees, GC-triggered frees) on stderr. |

`SAREK_DEBUG` is read in three places with three meanings; the rows above are the three. The runtime list is described in the [`sarek.core` README](https://github.com/mathiasbourgoin/Sarek/blob/main/sarek/core/README.md).
