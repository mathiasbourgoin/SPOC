---
layout: page
title: Sarek Backends
---

# Supported Backends

Sarek is designed to be backend-agnostic. You write your kernel once in OCaml, and Sarek compiles it to the appropriate shading language (CUDA C, OpenCL C, GLSL, MSL) or executes it natively on the CPU.

| Backend | Target | Status | Requirements |
|---------|--------|--------|--------------|
| **CUDA** | NVIDIA GPUs | Stable | NVIDIA Driver + CUDA Toolkit |
| **OpenCL** | Multi-vendor (AMD, Intel, NVIDIA) | Stable | OpenCL Runtime / ICD |
| **Vulkan** | Cross-platform (Linux, Windows, Android) | Stable | Vulkan SDK + glslangValidator |
| **Metal** | Apple Silicon & Intel Macs | Stable | macOS 10.13+ |
| **Native** | CPU (Multicore) | Stable | OCaml 5.4.0+ |
| **Interpreter**| CPU (Debug) | Stable | None |

## CUDA Backend (`sarek-cuda`)

Targets NVIDIA GPUs using the CUDA Driver API and NVRTC (Runtime Compilation).

- **Features**: Shared memory, atomics, warp intrinsics, dynamic parallelism.
- **Performance**: Native CUDA performance; uses JIT compilation to optimize for the specific GPU architecture.
- **Setup**: Requires `libcuda.so` and `libnvrtc.so` in your library path.

Build it from the repository (`dune build sarek-cuda`), list `sarek-cuda.plugin` in your `libraries`, and call `Sarek_cuda.Cuda_plugin.init ()` (CUDA/PTX, the default path) and, if you want the NVRTC path, `Sarek_cuda.Cuda_c_plugin.init ()` (CUDA/C) before `Device.init` (see [Getting Started](getting_started.html#using-sarek-in-a-dune-project)).

## OpenCL Backend (`sarek-opencl`)

Targets a wide range of devices including AMD GPUs, Intel integrated graphics, and FPGAs.

- **Features**: Work-group barriers, local memory, math intrinsics.
- **Compatibility**: Tested on NVIDIA, AMD, and Intel platforms.
- **Setup**: Requires an OpenCL ICD loader (e.g., `ocl-icd-libopencl1` on Linux).

Build it from the repository (`dune build sarek-opencl`), list `sarek-opencl.plugin` in your `libraries`, and call `Sarek_opencl.Opencl_plugin.init ()` before `Device.init` (see [Getting Started](getting_started.html#using-sarek-in-a-dune-project)).

## Vulkan Backend (`sarek-vulkan`)

Uses GLSL compute shaders and SPIR-V for modern cross-platform GPU support.

- **Pipeline**: Generates GLSL -> Compiles to SPIR-V (via `glslang` or `shaderc`) -> Executes on Vulkan.
- **Features**: Push constants, SSBOs (Storage Buffers), specialization constants.
- **Platform**: Ideal for modern Linux desktops, Android devices, and Windows.

Build it from the repository (`dune build sarek-vulkan`), list `sarek-vulkan.plugin` in your `libraries`, and call `Sarek_vulkan.Vulkan_plugin.init ()` before `Device.init` (see [Getting Started](getting_started.html#using-sarek-in-a-dune-project)).

## Metal Backend (`sarek-metal`)

Native support for Apple hardware (M1/M2/M3 chips) using Metal Shading Language (MSL).

- **Features**: Threadgroup memory, SIMD-group functions.
- **Limitations**: No double precision (`float64`) support on most Apple hardware.
- **Platform**: macOS and iOS only.

Build it from the repository (`dune build sarek-metal`), list `sarek-metal.plugin` in your `libraries`, and call `Sarek_metal.Metal_plugin.init ()` before `Device.init` (see [Getting Started](getting_started.html#using-sarek-in-a-dune-project)).

## Native CPU Backend (`sarek.native`)

Executes kernels directly on the host CPU without GPU compilation.

- **Mechanism**: Uses OCaml 5 Domains for parallel execution.
- **Use Case**: High-performance fallback when no GPU is available, or for debugging logic with standard debuggers.

## Interpreter Backend (`sarek.interpreter`)

Walks through the Sarek IR (Intermediate Representation) step-by-step.

- **Use Case**: Deep debugging of kernel logic, verification of IR transformations, and educational purposes.
- **Performance**: Slow (interpreted), but provides full visibility into execution.

## WebGPU / WGSL Codegen Target

Sarek includes a **WGSL** (WebGPU Shading Language) code-generation backend that emits compute shaders for browser-side execution via the WebGPU API.

- **Output**: WGSL source (`@compute @workgroup_size(...)` entry points)
- **Deployment**: Runs in any modern browser with WebGPU support (Chrome/Chromium 113+)
- **Note**: WGSL is a *transpiler target*, not a runtime device plugin — it does not appear in the device enumeration table above. Use it to ship kernel logic to the browser, combined with a JavaScript WebGPU host.
- **Try it**: The live [Playground](/Sarek/playground.html) lets you transpile any Sarek kernel to WGSL instantly, and the [Learn course](/Sarek/learn/) runs kernels directly on your GPU via WebGPU.
