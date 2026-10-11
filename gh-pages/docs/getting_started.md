--- 
layout: page
title: Getting Started with Sarek
---

# Getting Started with Sarek

Sarek is a high-performance framework for GPGPU programming in OCaml. It allows you to write kernels directly in OCaml syntax and execute them on various backends including CUDA, OpenCL, Vulkan, and Metal.

## Installation

### Prerequisites

- **OCaml**: 5.4.0+ (required for effects and domains support)
- **Dune**: 3.15+
- **GPU Drivers** (optional):
  - **CUDA**: NVIDIA drivers + CUDA Toolkit
  - **OpenCL**: OpenCL runtime (Intel NEO, ROCm, etc.)
  - **Vulkan**: Vulkan SDK + glslangValidator
  - **Metal**: macOS 10.13+ (included with Xcode)

### Installing from Source

Sarek is not yet in the official opam repository. Install from source:

```bash
# Clone the repository
git clone https://github.com/mathiasbourgoin/Sarek.git
cd Sarek

# Install dependencies
opam install . --deps-only -y

# Build
dune build

# Optional: install locally in your opam switch
opam install .
```

GPU backends (CUDA, OpenCL, Vulkan, Metal) are automatically detected and enabled based on available drivers and SDKs on your system.

## Your First Kernel: Vector Addition

This section builds a complete program that adds two vectors and prints one element of the result. It runs on any machine: the Native backend executes the kernel on the CPU when no GPU is available.

Sarek uses the `[%kernel ...]` syntax to define code that runs on the device.

<!-- snippet: program -->
```ocaml
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* 1. Define the kernel *)
let vector_add =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (c : float32 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) + b.(tid)]

let () =
  (* 2. Initialize input data *)
  let n = 1024 in
  let a = Vector.create Vector.float32 n in
  let b = Vector.create Vector.float32 n in
  let c = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set a i (float_of_int i) ;
    Vector.set b i (float_of_int (i * 2))
  done ;

  (* 3. Register the backends, then select a device. [best] returns the first
     CUDA device, else the first OpenCL device, else the first device found. *)
  Sarek_native.Native_plugin.init () ;
  let _devices = Device.init () in
  match Device.best () with
  | None -> prerr_endline "No device available"
  | Some dev ->
      Printf.printf "Using device: %s (%s)\n" dev.Device.name dev.Device.framework ;

      (* 4. Get the IR and launch the kernel *)
      let _, kirc = vector_add in
      let ir =
        match kirc.Sarek.Kirc_types.body_ir with
        | Some ir -> ir
        | None -> failwith "No IR"
      in
      let block = Execute.dims1d 256 in
      let grid = Execute.dims1d ((n + 255) / 256) in
      Execute.run_vectors
        ~device:dev
        ~ir
        ~args:[Vec a; Vec b; Vec c; Int n]
        ~block
        ~grid
        () ;
      Transfer.flush dev ;

      (* 5. Check the result *)
      Printf.printf "c[10] = %f\n" (Vector.get c 10)
```

`Device.init ()` enumerates the devices of every backend that has registered. A backend registers when its module is initialised, and `dune` links a library module only if the program refers to it; the call `Sarek_native.Native_plugin.init ()` is that reference. See [Using Sarek in a dune project](#using-sarek-in-a-dune-project) for the other backends.

## Shared Memory & Synchronization

Sarek supports shared memory and barriers. This kernel sums each block of a vector into one element of `output`: every block of 256 threads loads its slice into shared memory and halves the active threads at each step. `block_barrier ()` makes all threads of the block wait for each other.

```ocaml
let reduce_sum =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      (* Allocate shared memory for the thread block *)
      let%shared (sdata : float32) = 256l in

      let tid = thread_idx_x in
      let gid = global_thread_id in

      (* Load data into shared memory *)
      sdata.(tid) <- (if gid < n then input.(gid) else 0.0) ;

      (* Synchronize all threads in the block *)
      block_barrier () ;

      (* Tree reduction in shared memory *)
      let stride = mut 128l in
      while stride > 0l do
        if tid < stride then sdata.(tid) <- sdata.(tid) +. sdata.(tid + stride) ;
        block_barrier () ;
        stride := stride / 2l
      done ;

      (* Write the block result to global memory *)
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]
```

## Using Sarek in a dune project

A project that uses Sarek needs the `sarek.ppx` preprocessor, the libraries it calls, and one backend library per backend it wants to use:

```lisp
(executable
 (name my_program)
 (libraries sarek sarek.core sarek.stdlib sarek.native)
 (preprocess
  (pps sarek.ppx))
 (flags
  (:standard -w -33)))
```

The `flags` line is needed in dune's default `dev` profile: the code `[%kernel ...]` expands to an `open` of a module the kernel may not use, and warning 33 (unused open) is an error there.

| Library | Provides | Backend init call |
|---|---|---|
| `sarek.core` | `Spoc_core.Device`, `Vector`, `Transfer` | none |
| `sarek.stdlib` | `Sarek_stdlib.Std`, the kernel intrinsics | none |
| `sarek.native` | `Native` CPU backend (OCaml 5 domains) | `Sarek_native.Native_plugin.init ()` |
| `sarek.interpreter` | `Interpreter` CPU backend (debugging) | `Sarek_interpreter.Interpreter_plugin.init ()` |
| `sarek-cuda.plugin` | `CUDA/PTX` (default CUDA path) | `Sarek_cuda.Cuda_plugin.init ()` |
| `sarek-cuda.plugin` | `CUDA/C` (NVRTC) | `Sarek_cuda.Cuda_c_plugin.init ()` (it is not registered by the PTX call) |
| `sarek-hip.plugin` | `HIP` | `Sarek_hip.Hip_plugin.init ()` |
| `sarek-opencl.plugin` | `OpenCL` | `Sarek_opencl.Opencl_plugin.init ()` |
| `sarek-vulkan.plugin` | `Vulkan` | `Sarek_vulkan.Vulkan_plugin.init ()` |
| `sarek-metal.plugin` | `Metal` | `Sarek_metal.Metal_plugin.init ()` |

Adding a backend library to `libraries` is not enough: the program must call that backend's `init` (or otherwise refer to one of its modules) before `Device.init`, otherwise the linker drops the library and the backend never registers. Call the `init` of every backend you list; an `init` is harmless when the driver is missing, and the backend then does not appear among the devices. Backends are tried in the order `Device.init` lists them (CUDA, HIP, OpenCL, Vulkan, Metal, Native, Interpreter); pass `~frameworks:["OpenCL"; "Native"]` to choose. The `~frameworks` list only applies to the first `Device.init` call (or the first after `Device.reset ()`); later calls return the devices already enumerated.

```ocaml
(* Register every backend this program was linked with. *)
let register_backends () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;      (* CUDA/PTX *)
  Sarek_cuda.Cuda_c_plugin.init () ;    (* CUDA/C *)
  Sarek_hip.Hip_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init () ;
  Sarek_metal.Metal_plugin.init ()
```

Build and run it:

```bash
dune exec ./my_program.exe

# CPU only: no GPU backend registers
SPOC_DISABLE_GPU=1 dune exec ./my_program.exe
```

## Next Steps

- **[Examples](../examples/)** - Learn through practical examples (vector add, matrix multiply, reduction, transpose, mandelbrot)
- **[Concepts](concepts.html)** - Understand Sarek's design and programming model
- **[Benchmarks](../benchmarks/)** - See performance data across different GPUs and backends
- **[Backends](backends.html)** - Learn about CUDA, OpenCL, Vulkan, Metal, and WebGPU/WGSL support
- **[API Documentation](../spoc_docs/index.html)** - Complete API reference
- **[Try it in the browser](../playground.html)** - Live kernel transpiler (no install needed)
- **[Interactive Learn course](../learn/)** - Run kernels on your GPU straight from the page
