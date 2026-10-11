# sarek/core - Runtime Core

`sarek.core` (OCaml module `Spoc_core`) is the runtime layer between the kernel compiler and the backend plugins. It enumerates devices, stores vectors that track where their data lives, moves data between host and device, and exposes the raw compile-and-launch interface that `Sarek.Execute` builds on.

Use it to create vectors, choose devices and control transfers. To launch a Sarek kernel, use `Sarek.Execute.run_vectors` ([sarek/sarek/README.md](../sarek/README.md)). A program must link `sarek.core` and register at least one backend before enumerating devices; see [Using Sarek in a dune project](../../gh-pages/docs/getting_started.md#using-sarek-in-a-dune-project).

| Module | Role |
|---|---|
| `Device` | Enumerate devices of the registered backends; filters and predicates |
| `Vector` | Typed vectors with host storage and location tracking; bulk operations |
| `Transfer` | Host/device movement, streams, events, pinned memory, device-only vectors |
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

`Device.init` visits the frameworks in the order given (default: CUDA, HIP, OpenCL, Vulkan, Metal, Native, Interpreter), skips those that are not registered or not available, and numbers the devices in that order. The family name `"CUDA"` resolves to every registered `CUDA/...` backend. `SPOC_DISABLE_GPU=1`, or `SPOC_DISABLE_CUDA`, `SPOC_DISABLE_HIP`, `SPOC_DISABLE_OPENCL`, `SPOC_DISABLE_VULKAN`, `SPOC_DISABLE_METAL`, keep a GPU backend from registering ([device selection](../../gh-pages/docs/device_selection.md)).

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

`Transfer.to_device_all`, `to_cpu_all` and `sync_all` take lists. On CPU backends (Native, Interpreter) device memory is host memory and the transfers are no-ops.

### Device-only vectors, ranged copies, streams, pinned memory

These serve streaming workloads, such as copying expert weights into a VRAM-resident cache while the GPU computes (spec: `specs/device-buffers-async-copy.md`).

| Capability | HIP | CUDA | Metal | Native / Interpreter | OpenCL, Vulkan, WebGPU |
|---|---|---|---|---|---|
| device-only vectors, ranged copies | yes | yes | yes (shared MTLBuffer, memcpy) | yes (memmove) | refused |
| copies / launches on a stream | yes | yes | launches asynchronous on the stream's queue; copies complete on return | accepted, synchronous | refused |
| events: create / record / synchronize | yes | yes | yes (newest command buffer of the stream) | trivially complete | yes (existing `Event`) |
| `stream_wait_event`, `event_query` | yes | yes | wait implied by buffer tracking; query real | trivially complete | refused |
| pinned host memory | `hipHostMalloc` (portable) | `cuMemHostAlloc` (portable) | page-aligned host memory | plain host memory | refused |
| `register_host` | `hipHostRegister` | `cuMemHostRegister` (not on ZLUDA) | no-op | no-op | refused |
| `stage_to_device` | asynchronous, staged | asynchronous, staged | synchronous ranged copy | synchronous ranged copy | refused |

Metal (spec `specs/metal-ranged-copies.md`): every buffer is `MTLResourceStorageModeShared`; on Apple silicon CPU and GPU share one physical memory, so a ranged copy is one `memcpy` into the buffer's `contents`. Kernel launches are asynchronous: dispatches are batched into one command buffer per queue, committed when the queue's previous command buffer has completed or once the batch holds `SPOC_METAL_BATCH` dispatches (default 32), otherwise by the next launch, wait, event record or synchronisation. Every host access to a buffer first waits for the pending command buffers that bind it. A buffer larger than the device's `maxBufferLength` is refused with an error naming the limit.

"Refused" is `Backend_error (Plugin {error = Feature_not_supported _})`. Caller errors (bounds, wrong device, wrong location, freeing in-flight memory) raise `Invalid_argument`. Nothing silently falls back.

The core of the interface, written as a signature that the compiler checks against `Transfer`:

```ocaml
module type Streaming = sig
  type stream
  type event

  val create_stream : Device.t -> stream
  val synchronize_stream : stream -> unit
  val destroy_stream : stream -> unit
  val create_event : Device.t -> event
  val record_event : event -> stream -> unit
  val stream_wait_event : stream -> event -> unit
  val event_query : event -> bool
  val synchronize_event : event -> unit
  val destroy_event : event -> unit

  (* Device-only: no host mirror on a GPU; host access is refused *)
  val create_device_only :
    Device.t -> ('a, 'b) Vector.kind -> int -> ('a, 'b) Vector.t

  (* Pinned host memory, explicit lifetime (like munmap) *)
  val alloc_pinned :
    Device.t ->
    ('a, 'b) Bigarray.kind ->
    int ->
    ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t

  val free_pinned : (_, _, Bigarray.c_layout) Bigarray.Array1.t -> unit
  val is_pinned : (_, _, Bigarray.c_layout) Bigarray.Array1.t -> bool
end

module _ : Streaming = Transfer
```

Ranged copies take an optional `?stream`. Without it they complete on return; with it they are enqueued, and SPOC keeps the Bigarrays and vectors alive until the stream (or a later event) is synchronised:

| Function | Moves | Offset unit |
|---|---|---|
| `copy_to_device src ~dst ~dst_offset` | Bigarray to vector | elements |
| `copy_bytes_to_device src ~dst ~dst_byte_offset` | Bigarray to vector | bytes |
| `copy_ptr_to_device ~src_ptr ~byte_size ~dst ~dst_byte_offset ()` | raw address to vector (the caller keeps the source alive) | bytes |
| `copy_to_host ~src ~src_offset dst` | vector to Bigarray | elements |
| `copy_bytes_to_host ~src ~src_byte_offset dst` | vector to Bigarray | bytes |
| `stage_to_device src ~dst ~dst_offset` | at most `max_staged_bytes` (64 KiB), without waiting for the GPU; the source may be reused on return | elements |
| `register_host dev ba` / `unregister_host ba` | pin existing host memory | - |

The streaming pattern, with a copy stream and a compute stream:

```ocaml
let streaming_step dev ~ir ~cache ~x ~y ~slot ~slot_bytes ~block ~grid =
  let copy = Transfer.create_stream dev
  and compute = Transfer.create_stream dev in
  let ready = Transfer.create_event dev in
  let staged = Transfer.alloc_pinned dev Bigarray.char slot_bytes in
  (* ... fill [staged], for example with pread ... *)
  Transfer.copy_bytes_to_device
    ~stream:copy
    staged
    ~dst:cache
    ~dst_byte_offset:(slot * slot_bytes) ;
  Transfer.record_event ready copy ;
  Transfer.stream_wait_event compute ready ;
  Sarek.Execute.run_vectors
    ~stream:compute
    ~device:dev
    ~ir
    ~args:[Vec cache; Vec x; Vec y]
    ~block
    ~grid
    ()
```

Rules worth knowing:

- A ranged copy needs the vector's device copy to be current (`GPU`, `Both`, `Stale_CPU` on that device, or device-only); otherwise call `to_device` first. A ranged write turns `Both` into `Stale_CPU`. Custom (record) kinds are refused.
- A read-back (`copy_to_host`, `copy_bytes_to_host`) leaves the vector's location unchanged. `copy_bytes_to_host` copies `size_in_bytes dst` bytes whatever the two element kinds are. With `~stream`, the destination is undefined until the copy has completed, and `free_pinned`/`unregister_host` refuse it meanwhile.
- On CPU devices a device-only vector keeps host storage and host access works. Do not rely on it.
- Freeing a device-only vector discards its contents. Freeing anything a pending stream operation references is refused.
- `synchronize_stream`, `flush`, `synchronize_event` and a `true` `event_query` release what the stream kept alive. Destroy streams you create: a dropped stream keeps its pending work rooted.
- Streams from `create_stream` are blocking streams, and the auto-sync path depends on that. A synchronous transfer on the default stream serialises with them, so a streaming loop should not mix in auto-transfers.
- Vector lengths are capped at `Int32.max_int` elements (kernels index with int32); `Execute` refuses longer vectors.
- Vector location and buffer tables are not synchronised across domains: use a vector from one domain at a time. The keep-alive ledger and the pinned registry are mutex-protected.

**Small host writes to device-resident vectors.** `Vector.set` on a vector a GPU kernel has used downloads the whole vector (after every queued kernel), and the next launch re-uploads it with a synchronous copy. `Transfer.stage_to_device` copies the source into a 1 MiB per-device pinned staging ring and enqueues the copy on the stream, so it neither waits for the GPU nor requires the source to stay unchanged. Reusing one caller-owned pinned buffer with `copy_to_device ~stream` instead is a race: a queued copy reads whatever the buffer holds when it runs. Measured with `sarek-hip/test/bench_hip_small_writes.exe` (RX 7900 XTX, ROCm 7.2, 2026-10-10, shared machine; 8 int32 + 8 float32 rewritten, 300 us host work, 4 launches per iteration; three runs):

| Mode | us / iteration | of which in the writes |
|---|---|---|
| no rewrite | 320-324 | 0 |
| `Vector.set` | 709-712 | 339-344 |
| `stage_to_device` | 333-343 | 10-13 |

**Registering host memory on ROCm.** Measured with `sarek-hip/test/bench_hip_register_host.exe` (raw output in `docs/measurements/transfer-pitfalls-2026-10-10/`; 8 GiB, synchronous 5.5 MiB copies; tartarus 2026-10-10, under memory pressure from other workloads). Details and caveats are on `Transfer.register_host`.

- Copies right after registering are sometimes slow: 7 of 50 first passes ran at 0.95-6.17 GB/s (about 26 GB/s otherwise), whatever the preparation. The effect is intermittent and state-dependent.
- A copy may not span two registrations: `hipErrorInvalidValue`.
- ROCm does not count registered pages as pinned (`VmPin: 0 kB`); whether that explains slow copies under memory pressure is not established.

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

`SAREK_DEBUG` is a comma-separated list read when the library loads. Components: `transfer`, `kernel`, `device`, `memory`, `execute`, or `all` (also `*`). Levels: `debug`, `info`, `warn`, `error`.

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
