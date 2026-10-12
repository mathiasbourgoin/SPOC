# sarek/core - Runtime Core

`sarek.core` (OCaml module `Spoc_core`) is the runtime layer between the kernel compiler and the backend plugins. It enumerates devices, stores vectors that track where their data lives, moves data between host and device, and exposes the raw compile-and-launch interface that `Sarek.Execute` builds on.

Use it to create vectors, choose devices and control transfers. To launch a Sarek kernel, use `Sarek.Execute.run_vectors` ([sarek/sarek/README.md](../sarek/README.md)). A program must link `sarek.core` and register at least one backend before enumerating devices; see [Using Sarek in a dune project](../../gh-pages/docs/getting_started.md#using-sarek-in-a-dune-project).

| Module | Role |
|---|---|
| `Device` | Enumerate devices of the registered backends; filters and predicates |
| `Vector` | Typed vectors with host storage and location tracking; bulk operations |
| `Transfer` | Host/device movement, streams, events, pinned memory, device-only vectors |
| `Host_ptr_buffer` | Explicit opt-in: a CPU OpenCL buffer over a vector's host memory (`CL_MEM_USE_HOST_PTR`) |
| `Memory` | Raw device buffers |
| `Kernel`, `Kernel_arg`, `Runtime` | Compile source text and launch it with explicit arguments |
| `Log`, `Error`, `Profiling`, `Advanced` | Logging, validation helpers, profiling, optional features |
| `Soa_vector`, `Soa` | Structure-of-Arrays layout for record vectors (CUDA/PTX) |

## Quick start

The snippets in this page form one program: later ones reuse the three module aliases defined here.

```ocaml
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () =
  (* Register a backend, then enumerate devices *)
  Sarek_native.Native_plugin.init () ;
  let devices = Device.init () in
  Printf.printf "%d device(s)\n" (Array.length devices) ;

  (* Create vectors and fill them on the host *)
  let a = Vector.create Vector.float32 1024 in
  let b = Vector.create Vector.float32 1024 in
  for i = 0 to 1023 do
    Vector.set a i (float_of_int i) ;
    Vector.set b i (float_of_int (i * 2))
  done ;

  match Device.best () with
  | None -> prerr_endline "no device"
  | Some dev ->
      (* Explicit transfer; a kernel launch does this automatically *)
      Transfer.to_device a dev ;
      Transfer.to_device b dev ;

      (* ... launch a kernel with Sarek.Execute.run_vectors ... *)

      (* Bring results back and read them *)
      Transfer.to_cpu a ;
      Printf.printf "a[10] = %f\n" (Vector.get a 10)
```

## Devices

`Device.t` is a record `{id; backend_id; name; framework; capabilities}`. `id` is global across backends, `backend_id` is the index within the backend, and `framework` is the backend name (`"CUDA/PTX"`, `"CUDA/C"`, `"HIP"`, `"OpenCL"`, `"Vulkan"`, `"Metal"`, `"Native"`, `"Interpreter"`).

```ocaml
let () =
  (* Enumerate once; later calls to [Device.all] return the same array *)
  let devices = Device.init ~frameworks:["HIP"; "OpenCL"; "Native"] () in
  Array.iter (fun dev -> print_endline (Device.to_string dev)) devices ;

  (* [best] is the first CUDA device, else the first OpenCL device, else the
     first device of the list *)
  let _best = Device.best () in

  (* Filters and queries *)
  let _cuda = Device.filter_cuda () in
  let _opencl = Device.by_framework "OpenCL" in
  let _fp64 = Device.with_fp64 () in
  match Device.first () with
  | Some dev ->
      let caps = dev.Device.capabilities in
      Printf.printf
        "%s: cpu=%b atomics=%b warp=%d max threads/block=%d shared=%d B\n"
        dev.Device.name
        (Device.is_cpu dev)
        caps.Spoc_framework.Framework_sig.supports_atomics
        caps.Spoc_framework.Framework_sig.warp_size
        caps.Spoc_framework.Framework_sig.max_threads_per_block
        caps.Spoc_framework.Framework_sig.shared_mem_per_block
  | None -> ()
```

`Device.init` visits the frameworks in the order given (default: CUDA, HIP, OpenCL, Vulkan, Metal, Native, Interpreter), skips those that are not registered or not available, and numbers the devices in that order. `~frameworks` applies only to the first call, or the first after `Device.reset ()`; later calls return the devices already enumerated. The family name `"CUDA"` resolves to every registered `CUDA/...` backend. `SPOC_DISABLE_GPU=1`, or `SPOC_DISABLE_CUDA`, `SPOC_DISABLE_HIP`, `SPOC_DISABLE_OPENCL`, `SPOC_DISABLE_VULKAN`, `SPOC_DISABLE_METAL`, keep a GPU backend from registering ([device selection](../../gh-pages/docs/device_selection.md)).

Double and 64-bit integer support are device features, not booleans: use `Device.allows_fp64` and `Device.allows_int64`.

## Vectors

A vector is `('a, 'b) Vector.t`: `'a` is the OCaml element type and `'b` the Bigarray element tag. Predefined kinds are `Vector.float16`, `float32`, `float64`, `int32`, `int64`, `char` and `complex32`; record types declared with `[@@sarek.type]` get a custom kind (`Vector.create_custom`). Host storage is a Bigarray for scalar kinds and a C-layout buffer for custom kinds.

```ocaml
let () =
  let v = Vector.create Vector.float32 8 in
  Vector.fill v 1.0 ;
  Vector.set v 3 42.0 ;

  (* Constructors, bulk operations *)
  let squares = Vector.init Vector.int32 8 (fun i -> Int32.of_int (i * i)) in
  let copy = Vector.copy v in
  let doubled = Vector.map (fun x -> 2.0 *. x) Vector.float32 v in
  let total = Vector.fold_left ( +. ) 0.0 v in
  Printf.printf
    "len=%d sum=%f squares[3]=%ld copy[3]=%f doubled[3]=%f\n"
    (Vector.length v)
    total
    (Vector.get squares 3)
    (Vector.get copy 3)
    (Vector.get doubled 3) ;

  (* Window onto part of a vector, sharing its storage *)
  let _tail = Vector.sub_vector v ~start:4 ~len:4 () in
  let _as_array = Vector.to_array v in
  ()
```

**Location.** `Vector.location` is `CPU`, `GPU d`, `Both d`, `Stale_CPU d` (the device holds newer data) or `Stale_GPU d` (the host holds newer data). `Vector.get` and `Vector.set` synchronise automatically. After a kernel launch the arguments are `Stale_CPU`, so the first `get` downloads the whole vector; read results with `Transfer.to_cpu` once, not element by element.

```ocaml
let describe vec =
  match Vector.location vec with
  | Vector.CPU -> "host only"
  | Vector.GPU d -> Printf.sprintf "device %d only" d.Device.id
  | Vector.Both d -> Printf.sprintf "synchronised with device %d" d.Device.id
  | Vector.Stale_CPU d -> Printf.sprintf "device %d is newer" d.Device.id
  | Vector.Stale_GPU d -> Printf.sprintf "host is newer than device %d" d.Device.id
```

## Transfers

By default every kernel launch transfers its vector arguments (`Transfer.is_auto ()` is `true`). `Transfer.disable_auto ()` turns that off for code that manages placement itself.

```ocaml
let transfers dev v =
  Transfer.to_device v dev ;  (* host to device *)
  Transfer.sync v ;           (* make host and device agree *)
  Transfer.to_cpu v ;         (* device to host *)
  Transfer.flush dev          (* wait for the device's pending work *)
```

`Transfer.to_device_all`, `to_cpu_all` and `sync_all` take lists. On CPU backends (Native, Interpreter) device memory is host memory and the transfers are no-ops. A CPU OpenCL device copies like a GPU unless the vector was given a host-pointer buffer with `Host_ptr_buffer` (see [Memory and transfers](../../gh-pages/docs/memory.md)).

### Device-only vectors, ranged copies, streams, pinned memory

For streaming workloads, such as copying expert weights into a VRAM-resident cache while the GPU computes, `Transfer` also provides:

- `create_device_only`: vectors with no host storage on a GPU;
- ranged copies in both directions (`copy_to_device`, `copy_bytes_to_device`, `copy_ptr_to_device`, `copy_to_host`, `copy_bytes_to_host`), with an optional `?stream`;
- streams and events (`create_stream`, `record_event`, `stream_wait_event`, ...);
- pinned host memory (`alloc_pinned`, `register_host`) and `stage_to_device` for small host writes.

Support differs per backend: HIP, CUDA, Metal, Vulkan, Native and Interpreter support them (Metal and Vulkan complete copies on return; Vulkan pins with driver-allocated mapped memory on every device; registering needs `VK_EXT_external_memory_host` and the explicit opt-in `SPOC_VULKAN_HOST_IMPORT=1`, which AMD drivers refuse, Sarek #503); OpenCL and WebGPU refuse with `Feature_not_supported`. The user guide [Memory and transfers](../../gh-pages/docs/memory.md) has the support table, the rules, worked examples and the measured transfer pitfalls (`Vector.set` on a device-resident vector, `register_host` on ROCm). Specs: `specs/device-buffers-async-copy.md`, `specs/copy-bytes-to-host.md`, `specs/transfer-pitfalls.md`, `specs/metal-ranged-copies.md`.

## Raw buffers and kernels

`Memory`, `Kernel` and `Runtime` work below `Vector`: you allocate buffers, compile source text for a device, and bind arguments by position. `Sarek.Execute` uses them after generating device source from the kernel IR.

```ocaml
module Memory = Spoc_core.Memory
module Runtime = Spoc_core.Runtime

let launch_source (dev : Device.t) ~name ~source =
  let n = 1024 in
  let buf = Memory.alloc dev n Bigarray.float32 in
  Runtime.run
    dev
    ~name
    ~source
    ~args:[Runtime.ArgBuffer buf; Runtime.ArgInt32 (Int32.of_int n)]
    ~grid:(Runtime.dims1d ((n + 255) / 256))
    ~block:(Runtime.dims1d 256)
    () ;
  Memory.free buf
```

`Runtime.run` compiles (cached per device and source) and launches. `Kernel.compile`, `Kernel.create_args` and `Kernel.launch` expose the same steps separately. `Kernel_arg` is a typed view of an argument list for code that folds or maps over arguments.

## Debugging

`SAREK_DEBUG` is a comma-separated list read when the library loads. Components: `transfer`, `kernel`, `device`, `memory`, `execute`, or `all` (also `*`). Levels: `debug`, `info`, `warn`, `error`. Naming a component without a level sets the level to `debug`; a level in the list wins (the last one, if several). Messages go to stdout.

```bash
SAREK_DEBUG=transfer ./my_program.exe
SAREK_DEBUG=transfer,kernel,debug ./my_program.exe
SAREK_DEBUG=all ./my_program.exe
```

`Spoc_core.Log` offers the same switches in code (`Log.enable`, `Log.set_level`), and `Spoc_core.Error` has the validation helpers (`enable_validation`, `check_block_size`, `debug_vector`, `debug_device`).

## Tests

```bash
# All core tests
dune runtest sarek/core/test

# One test executable
dune exec sarek/core/test/test_vector.exe
dune exec sarek/core/test/test_device.exe
```

The test executables are `test_device`, `test_vector`, `test_vector_transfer`, `test_memory`, `test_kernel`, `test_kernel_arg`, `test_gpu_memory`, `test_staging_ring` and `test_nativeint_round_trip`.

## See also

- [sarek/sarek/README.md](../sarek/README.md): launching kernels with `Execute.run_vectors`
- [sarek/ppx/README.md](../ppx/README.md): the kernel compiler
- [Device selection](../../gh-pages/docs/device_selection.md): environment variables that choose backends
