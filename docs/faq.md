---
layout: index_sample
title: FAQ
---

# Frequently Asked Questions

## General Questions

### Do I need an NVIDIA GPU to use Sarek?
No. While Sarek supports **CUDA** for NVIDIA hardware, it also supports **OpenCL** and **Vulkan**, which work on AMD GPUs, Intel Integrated Graphics, and even some FPGAs. If no GPU is available, Sarek can fall back to the **Native CPU** backend.

### Can I run Sarek on macOS?
Yes. Sarek has a dedicated **Metal** backend for Apple Silicon (M1/M2/M3) and Intel-based Macs. It also supports OpenCL on macOS.

### Does Sarek work on Windows?
Sarek has limited testing on Windows, but it should work via the OpenCL and Vulkan backends. Using **WSL2** is recommended for the best experience.

## Technical Questions

### What is the overhead of using OCaml for GPU programming?
Sarek uses a JIT compilation approach for GPU backends. The logic you write in OCaml is compiled into native GPU code (CUDA C, GLSL, etc.). Once the kernel is compiled and cached, the execution performance is comparable to hand-written C/CUDA code.

### Does the OCaml Garbage Collector interfere with the GPU?
No. The GPU has its own memory. Data is explicitly transferred between the OCaml heap and the GPU memory via `Vector` objects. The GC manages the OCaml "handle" to the GPU memory, but does not touch the data residing on the device.

### Can I use custom OCaml types in my kernels?
Yes. By using the `[@@sarek.type]` attribute on record definitions, Sarek automatically generates the corresponding C structures for the GPU and handles the memory layout.

### Is OCaml 5 required?
Yes. The latest version of Sarek leverages **OCaml 5 Effects and Domains** for the Native CPU backend to provide high-performance parallel execution on multi-core processors.
