---
layout: page
title: Building a New Backend
---

# Building a New Backend Plugin

Sarek is designed to be extensible. Adding a new target (e.g., HIP, SYCL, Level Zero, or a custom FPGA backend) involves implementing the standard `BACKEND` signature.

> **Note**: A WGSL/WebGPU codegen target already exists (`sarek/codegen/Sarek_ir_wgsl.ml`). It is a transpiler-only output (not a runtime device plugin), so it does not implement the `BACKEND` signature described here — it is a standalone IR-to-WGSL pass.

## 1. Choosing an Execution Model

Backends in Sarek follow one of two primary execution models:

- **JIT (Just-In-Time)**: The backend generates source code (CUDA C, OpenCL C, GLSL) from Sarek IR at runtime, compiles it using a GPU driver, and launches the binary.
- **Direct**: The backend executes pre-compiled OCaml functions. This is how the **Native CPU** backend works, using OCaml 5 Domains for parallelism without any runtime compilation.

Your implementation must specify this in the `execution_model` field of the signature.

## 2. Implementing Intrinsics

**Intrinsics** are the bridge between Sarek's high-level OCaml syntax and backend-specific hardware features (like thread IDs, barriers, or specialized math functions).

### The Intrinsic Registry
Every backend exposes an `Intrinsics` module of type `Framework_sig.INTRINSIC_REGISTRY`: a table from intrinsic names to the backend's implementation. `Intrinsic_registry.Make ()` creates such a table with a ready-made entry type (`intr_name`, `intr_codegen`, `intr_convergence`). The library is `sarek.framework.registry`.

For a **JIT backend**, `intr_codegen` is the source text the generator emits:
```ocaml
module Intrinsics = Spoc_framework_registry.Intrinsic_registry.Make ()

(* Mapping Sarek's block_barrier to OpenCL's barrier *)
let () =
  Intrinsics.register
    "block_barrier"
    {
      Spoc_framework_registry.Intrinsic_registry.intr_name = "block_barrier";
      intr_codegen = "barrier(CLK_LOCAL_MEM_FENCE);";
      intr_convergence = Spoc_framework.Framework_sig.Sync;
    }
```

A **Direct backend** (Native) executes pre-compiled OCaml functions, so its entries only name the runtime call; the Native backend registers them as placeholders:
```ocaml
(* Naming the runtime call behind a thread index *)
let () =
  Intrinsics.register
    "thread_id_x"
    {
      Spoc_framework_registry.Intrinsic_registry.intr_name = "thread_id_x";
      intr_codegen = "Sarek_cpu_runtime.thread_id_x ()";
      intr_convergence = Spoc_framework.Framework_sig.Divergent;
    }
```

The intrinsics that kernels call (`global_thread_id`, `block_barrier`, math functions) are declared once with `let%sarek_intrinsic`, which carries the device spelling per framework; see `sarek/ppx_intrinsic/README.md` in the repository.

## 3. Handling Native Code Blocks

Sarek allows users to embed backend-specific code directly using `[%native.backend "..."]`. 

If you are building a backend named `mygpu`, you must ensure your code generator looks for `SNative ("mygpu", code)` nodes in the IR. This allows power users to bypass the compiler and use hand-optimized assembly or specialized hardware features.

## 4. The `BACKEND` Signature Modules

Your plugin must implement several core modules:

- **Device**: Logic to enumerate hardware and query `capabilities` (max threads, shared memory size, FP64 support).
- **Memory**: Functions to allocate device buffers and transfer data. You should support:
    - `host_to_device` / `device_to_host`
    - `alloc_zero_copy`: (Optional) for devices that share memory with the CPU.
- **Kernel**: Handles the lifecycle of a kernel:
    - `compile`: Takes a source string and produces a runnable handle.
    - `launch`: Marshals `exec_arg` values into the hardware's expected format.

## 5. Plugin Registration

Register your backend with a `priority`. Priorities order the backends registered under one family name and decide `Framework_registry.best_backend`. The bundled backends use 100 for CUDA/PTX, 95 for Metal, 90 for OpenCL, 80 for Vulkan and HIP, 10 for Native and 5 for the Interpreter.

A plugin registers when its module is initialised and exposes an `init` function that forces the registration. A program that calls `init` links the module, which a mere `libraries` entry does not do (see [Using Sarek in a dune project](getting_started.html#using-sarek-in-a-dune-project)). The bundled plugins register only when the backend reports itself available:

```ocaml
module Make_plugin (B : Spoc_framework.Framework_sig.BACKEND) = struct
  let registered_backend =
    lazy
      (if B.is_available () then
         Spoc_framework_registry.Framework_registry.register_backend
           ~priority:80
           (module B : Spoc_framework.Framework_sig.BACKEND))

  let () = Lazy.force registered_backend

  let init () = Lazy.force registered_backend
end
```

## 6. Testing Your Backend

1. **IR Tests**: Ensure your translator produces valid source code for all Sarek IR nodes (loops, matches, record assignments).
2. **Intrinsic Tests**: Verify that all standard Sarek intrinsics are correctly mapped.
3. **E2E Tests**: Run the suite in `sarek/tests/e2e`. A backend is considered "stable" if it passes the `test_vector_add`, `test_reduce`, and `test_matmul` benchmarks.