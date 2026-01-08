---
layout: index_sample
title: Building a New Backend
---

# Building a New Backend Plugin

Sarek is designed to be extensible. Adding a new target (e.g., WebGPU, HIP, or a custom FPGA backend) involves implementing the standard `BACKEND` signature.

## 1. The `BACKEND` Signature

Your plugin must implement the `BACKEND` signature defined in `spoc.framework`. The core modules you need to provide are:

- **Device**: Logic to enumerate hardware and query capabilities (max threads, shared memory size).
- **Memory**: Functions to allocate device buffers and transfer data between the OCaml heap and the device.
- **Kernel**: The heart of the plugin. It must handle:
    - **Compilation**: Taking a string of source code and producing a runnable handle.
    - **Launching**: Setting up arguments and triggering execution on the hardware.
- **Intrinsics**: A registry of backend-specific functions (e.g., mapping Sarek's `thread_idx_x` to your target's equivalent).

## 2. Implementing Code Generation

Most backends are **JIT (Just-In-Time)**. This means you need to write a translator from **Sarek IR** to your target language.

Sarek provides helper modules to traverse the IR:
- `Sarek_ir_types.kernel`: The typed AST of the kernel.
- `Sarek_ir_analysis`: Tools for dependency and convergence analysis.

Your generator typically walks the IR and builds a source string (e.g., GLSL for Vulkan or MSL for Metal).

## 3. Handling Types

You must handle the mapping between OCaml/Sarek types and your backend's types:
- **Scalars**: `int32`, `float32`, `bool`, etc.
- **Vectors**: Represented as pointers in most C-like targets.
- **Custom Records**: Must be translated to C-style `structs`.
- **Variants**: Typically implemented as tagged unions.

## 4. Plugin Registration

Once your module is ready, register it with the framework using the registry:

```ocaml
(* In your plugin's main module *)
let () =
  Spoc_framework.Framework_registry.register_backend
    ~priority:80 (* Priority for auto-selection *)
    (module MyNewBackend : Spoc_framework.Framework_sig.BACKEND)
```

## 5. Development Workflow

1. **Clone an existing backend**: Use `sarek-cuda` or `sarek-opencl` as a reference.
2. **Define Error Handling**: Use the `Backend_error` functor for consistent error reporting.
3. **Write Unit Tests**: Focus on the code generator first (IR → Source).
4. **Run E2E Tests**: Use the existing test suite in `sarek/tests/e2e` to verify your backend produces correct results across all standard algorithms.
