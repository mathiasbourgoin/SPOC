# sarek/sarek - Launching Kernels

The `sarek` library (OCaml module `Sarek`) takes the IR that `[%kernel ...]` produces and runs it on a device. Its entry point is `Execute.run_vectors`: it transfers the vector arguments, runs the kernel with the execution path of the device's backend, and marks the arguments for read-back.

A program that launches kernels links `sarek`, `sarek.core`, `sarek.stdlib` and at least one backend, preprocesses with `sarek.ppx`, and calls the backend's `init` before enumerating devices. The complete stanza and the list of backend `init` calls are in [Using Sarek in a dune project](../../gh-pages/docs/getting_started.md#using-sarek-in-a-dune-project). Vectors, devices and transfers are in [sarek/core/README.md](../core/README.md); the kernel language is in [sarek/ppx/README.md](../ppx/README.md).

## Launch a kernel

The snippets in this page form one program: later ones reuse the module aliases and definitions of the first two.

```ocaml
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* y <- alpha * x + y *)
let saxpy =
  [%kernel
    fun (x : float32 vector) (y : float32 vector) (alpha : float32) (n : int32) ->
      let open Sarek_stdlib.Std in
      let i = global_thread_id in
      if i < n then y.(i) <- alpha *. x.(i) +. y.(i)]

(* A kernel is a pair; the IR is in the second component *)
let kernel_ir kernel =
  let _, kirc = kernel in
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"
```

The host side creates the vectors, picks a device and launches:

```ocaml
let run_saxpy dev =
  let n = 1_000_000 in
  let x = Vector.create Vector.float32 n in
  let y = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set x i (float_of_int i) ;
    Vector.set y i 1.0
  done ;
  let block = Execute.dims1d 256 in
  let grid = Execute.grid_for ~problem_size:n ~block_size:256 in
  Execute.run_vectors
    ~device:dev
    ~ir:(kernel_ir saxpy)
    ~args:[Vec x; Vec y; Float32 2.0; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  Printf.printf "y[10] = %f\n" (Vector.get y 10)

let () =
  Sarek_native.Native_plugin.init () ;
  let _devices = Device.init () in
  match Device.best () with
  | Some dev -> run_saxpy dev
  | None -> prerr_endline "no device available"
```

## Arguments

`~args` is a list of `Execute.vector_arg`, one per kernel parameter, in order.

| Constructor | Kernel parameter | Notes |
|---|---|---|
| `Vec v` | `t vector` | The vector is transferred to the device first; its length is passed along for bounds. |
| `Int n` | `int32` | An OCaml `int`, converted to 32 bits. |
| `Int32 n` | `int32` | |
| `Int64 n` | `int64` | |
| `Float32 f` | `float32` | An OCaml `float`, rounded to 32 bits by the device. |
| `Float64 f` | `float64` | |

The argument count and types are checked against the kernel before the launch; a mismatch raises `Execute_error.Execution_error`.

`~block` and `~grid` are `Framework_sig.dims` values built with `Execute.dims1d`, `dims2d` or `dims3d`; `Execute.grid_for ~problem_size ~block_size` gives the 1D grid that covers a problem. `?shared_mem` is the dynamic shared memory in bytes (default 0), and `?stream` enqueues the launch on a `Transfer.stream` (HIP, CUDA and Metal; synchronous on Native and Interpreter; refused elsewhere).

Records declared with `[@@sarek.type]` are passed as custom vectors; the record layout is shared by the host and the device code. See [Transpose](../../gh-pages/examples/transpose.md) for an example.

## Execution paths

`Execute.run_vectors` looks the device's backend up in the registry and follows its `execution_model`:

| Model | Backends | What happens |
|---|---|---|
| `JIT` | CUDA/PTX, CUDA/C, HIP, OpenCL, Vulkan, Metal | Source (or PTX) is generated from the IR, compiled by the driver and cached; the kernel is launched on the device. |
| `Direct` | Native | The kernel runs on the CPU, in parallel on OCaml 5 domains, without a device compiler. |
| `Custom` | Interpreter | The IR is evaluated directly, sequentially; slow, for checking kernel logic. |

On a JIT backend, the first launch of a kernel includes code generation and driver compilation; later launches reuse the cached result. HIP, OpenCL and Vulkan defer launches made without `~stream` until the next synchronisation point and may fuse independent launches (`Execute.Window`, spec `specs/launch-window.md`). A device compilation or launch failure is then raised by the call that flushes the window, for example `Transfer.flush dev`, not by `run_vectors`.

A launch whose IR is a matrix product that the device's matrix units can compute may run a rewritten kernel with identical outputs (`specs/matrix-pattern-dynamic.md`). `Execute.run_vectors_outcome` returns which kernel ran.

## Errors

Launch failures raise `Execute_error.Execution_error` with a structured payload; `Execute_error.error_to_string` formats it, and a printer is registered so uncaught errors print readably.

```ocaml
let guarded_launch dev =
  try run_saxpy dev with
  | Execute_error.Execution_error (Execute_error.Backend_error {backend; message}) ->
      Printf.eprintf "backend %s: %s\n" backend message
  | Execute_error.Execution_error (Execute_error.Invalid_dimensions {reason; _}) ->
      Printf.eprintf "invalid launch dimensions: %s\n" reason
  | Execute_error.Execution_error e ->
      prerr_endline (Execute_error.error_to_string e)
  | Sarek_backend_error.Backend_error.Backend_error e ->
      prerr_endline (Sarek_backend_error.Backend_error.to_string e)
```

The constructors are `Unbound_variable`, `Type_mismatch`, `Unsupported_argument`, `Backend_error`, `Compilation_failed`, `Invalid_dimensions`, `Missing_ir`, `Missing_native_fn`, `Transfer_failed`, `Interp_error`, `Invalid_file` and `Type_helper_not_found`. Backend-level failures (driver errors, refused features) surface as `Sarek_backend_error.Backend_error.Backend_error`, handled in the last case above (source: `spoc/framework_error/Backend_error.ml`).

## Running source you already have

`Execute.run_source` launches a kernel written in a backend's source language instead of Sarek IR (CUDA source, OpenCL C, PTX, SPIR-V, GLSL); `Execute.supports_lang dev lang` tells whether a device accepts a language.

## Modules

| Module | Role |
|---|---|
| `Execute` | `run_vectors`, `run_vectors_outcome`, `run_source`, `dims1d/2d/3d`, `grid_for`, launch window |
| `Soa_launch` | `run_soa`: launch with record vectors lowered as Structure-of-Arrays (CUDA/PTX) |
| `Kirc_types`, `Kirc_kernel` | The kernel pair produced by `[%kernel]` and its IR |
| `Sarek_fusion` | Kernel fusion; see [FUSION.md](FUSION.md) |
| `Sarek_cpu_runtime` | Runtime of the Native backend (thread state, barriers) |
| `Sarek_ir_interp` | IR interpreter used by the Interpreter backend |
| `Execute_error`, `Interp_error`, `Kirc_error`, `Fusion_error` | Structured errors |

## Tests

```bash
# Unit tests of this library
dune runtest sarek/sarek/test

# All tests under sarek/tests (Native and Interpreter always; GPU backends when present)
dune build @sarek/tests/runtest

# One end-to-end test, choosing the backend and device
dune exec sarek/tests/e2e/test_vector_add.exe -- --native
dune exec sarek/tests/e2e/test_vector_add.exe -- -d 0
```

The end-to-end tests accept `-d ID` (device), `--native`, `--interpreter`, `--vulkan`, `--metal`, `-s SIZE` (problem size), `-b SIZE` (block size) and `--benchmark`. Without GPU drivers, set `SPOC_DISABLE_GPU=1`.

## See also

- [Getting Started](../../gh-pages/docs/getting_started.md): a first program, project setup
- [sarek/core/README.md](../core/README.md): vectors, devices, transfers
- [sarek/ppx/README.md](../ppx/README.md): the kernel compiler and language
- [Device selection](../../gh-pages/docs/device_selection.md): environment variables
