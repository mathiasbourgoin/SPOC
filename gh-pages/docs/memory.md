---
layout: page
title: Memory and Transfers
---

# Memory and Transfers

This page shows how to keep data on the device, copy parts of a vector without moving the whole vector, order copies with kernel launches on streams, use pinned host memory, and avoid the transfer patterns that cost the most. It applies to code that manages placement itself, for example a program that streams weights into a VRAM cache while the GPU computes. For plain `Vector.get`/`set` and `Transfer.to_device`/`to_cpu`, see the [`sarek.core` README](https://github.com/mathiasbourgoin/Sarek/blob/main/sarek/core/README.md).

The snippets on this page form one program: later ones reuse the aliases defined first.

```ocaml
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
```

## Which backend supports what

| Capability | HIP | CUDA | Metal | Vulkan | Native, Interpreter | OpenCL, WebGPU |
|---|---|---|---|---|---|---|
| Device-only vectors, ranged copies | yes | yes | yes (shared MTLBuffer, `memcpy`) | yes (mapped `memcpy`, DMA from pinned memory, or staging) | yes (`memmove`) | refused |
| Copies and launches on a stream | yes | yes | launches asynchronous on the stream's queue; copies complete on return | launches batched on the stream; copies complete on return | accepted, synchronous | refused |
| Events: create, record, synchronize | yes | yes | yes (newest command buffer of the stream) | yes (fence) | trivially complete | yes (existing `Event`) |
| `stream_wait_event`, `event_query` | yes | yes | wait implied by buffer tracking; query real | wait implied by the single queue and barriers; query real (`vkGetFenceStatus`) | trivially complete | refused |
| Pinned host memory (`alloc_pinned`) | `hipHostMalloc` | `cuMemHostAlloc` | page-aligned host memory | driver-allocated mapped host memory (HOST_VISIBLE, HOST_COHERENT, HOST_CACHED when available); copies need no staging; works on every device, AMD included; unmapped when its device is destroyed | plain host memory | refused |
| `register_host` | `hipHostRegister` | `cuMemHostRegister` (not on ZLUDA) | no-op | import of the pages around the range (same extension) | no-op | refused |
| `stage_to_device` | asynchronous, staged | asynchronous, staged | synchronous ranged copy | synchronous ranged copy | synchronous ranged copy | refused |

"Refused" is `Backend_error (Plugin {error = Feature_not_supported _})`. Caller errors (bounds, wrong device, wrong location, freeing in-flight memory) raise `Invalid_argument`. Nothing silently falls back.

## Device-only vectors and ranged copies

A device-only vector has no host storage on a GPU. Host element access, `to_cpu` and auto-sync raise `Invalid_argument`; data moves only through ranged copies and kernels. No host copy is kept, so no whole-vector transfer can happen by accident.

```ocaml
let device_cache dev =
  (* 1024 int32 elements that live only on [dev]. *)
  let cache = Transfer.create_device_only dev Vector.int32 1024 in
  (* Write 16 elements at element offset 32. *)
  let src = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout 16 in
  Bigarray.Array1.fill src 7l ;
  Transfer.copy_to_device src ~dst:cache ~dst_offset:32 ;
  (* Read 64 bytes back, starting at byte 128, into a char buffer. *)
  let out = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 64 in
  Transfer.copy_bytes_to_host ~src:cache ~src_byte_offset:128 out ;
  out
```

The ranged copies, all with an optional `?stream`:

| Function | Moves | Offset unit |
|---|---|---|
| `copy_to_device src ~dst ~dst_offset` | Bigarray to vector | elements |
| `copy_bytes_to_device src ~dst ~dst_byte_offset` | Bigarray to vector | bytes |
| `copy_ptr_to_device ~src_ptr ~byte_size ~dst ~dst_byte_offset ()` | raw address to vector (the caller keeps the source alive) | bytes |
| `copy_to_host ~src ~src_offset dst` | vector to Bigarray | elements |
| `copy_bytes_to_host ~src ~src_byte_offset dst` | vector to Bigarray | bytes |
| `stage_to_device src ~dst ~dst_offset` | at most `max_staged_bytes` (64 KiB), without waiting for the GPU; the source may be reused on return | elements |
| `register_host dev ba` / `unregister_host ba` | pin existing host memory | not applicable |

`copy_bytes_to_host` copies `Bigarray.Array1.size_in_bytes dst` bytes whatever the element kinds of the vector and the destination are, and does no byte-order conversion. Slice the destination with `Bigarray.Array1.sub` for a shorter copy. Use it to move an `int32` device range into a `char` host buffer; `copy_to_host` requires the kinds to match.

Rules:

- A ranged copy needs the vector's device copy to be current (`GPU`, `Both`, `Stale_CPU` on that device, or device-only); otherwise call `Transfer.to_device` first. A ranged write turns `Both` into `Stale_CPU`. Record (custom) kinds are refused.
- A read-back leaves the vector's location unchanged.
- `create_device_only` refuses lengths outside `1 .. Int32.max_int` (kernels index with `int32`) and record kinds. `Execute` likewise refuses vectors longer than `Int32.max_int` elements.
- On CPU devices (Native, Interpreter) a device-only vector keeps host storage and host access works. Do not rely on it.
- Freeing a device-only vector discards its contents. Freeing anything a pending stream operation references is refused.
- Vector location and buffer tables are not synchronised across domains: use a vector from one domain at a time. The keep-alive ledger and the pinned registry are mutex-protected.

On Metal every buffer is shared storage, so a ranged copy is one `memcpy`. Launches are batched into one command buffer per queue, committed when the previous command buffer has completed or once the batch holds `SPOC_METAL_BATCH` dispatches (default 32), and otherwise by the next launch, wait, event record or synchronisation. A host access to a buffer first waits for the pending command buffers that bind it. A buffer larger than the device's `maxBufferLength` is refused with an error naming the limit. On Vulkan launches are also batched (`SPOC_VULKAN_BATCH`, default 32 dispatches per command buffer; `SPOC_VULKAN_BATCH=1` waits at every launch and is the fallback, Sarek #504); every host access to a vector first completes the pending launches, and a buffer release is queued until the pending launches have completed. Every Vulkan copy also completes them: a copy is complete on return and ordered against the launches of every stream (there are no asynchronous Vulkan copies yet). On Vulkan `alloc_pinned` takes memory from the driver (mapped, no import), so it works everywhere with no opt-in. Host-memory import is off by default: `register_host` refuses with `Feature_not_supported` unless `SPOC_VULKAN_HOST_IMPORT=1`, and it is refused on AMD drivers (RADV, AMDVLK, amdgpu-pro) even then, because importing pageable host pages was followed by a kernel oops in `amdgpu_hmm_invalidate_gfx` on Linux 7.1.2 (Sarek #503). A host range from `alloc_pinned` (or `register_host` where enabled) is copied by DMA, with no staging buffer and no host `memcpy`; measured on an RX 7900 XTX, 64 MiB host-to-device 21.5-22.3 GB/s pinned against 5.5 GB/s through staging, and device-to-host 21.8-22.0 GB/s against 2.5 GB/s (provisional, machine under load; `bench_vulkan_pinned_copy`). Device memory that the host maps through the PCIe BAR is write-combined and uncached, so SPOC never reads it with `memcpy` on Vulkan.

## Streams and events

Copies and launches with `~stream` are enqueued, not completed on return. SPOC keeps the Bigarrays and vectors alive until the stream, or a later event, is synchronised. The destination of a stream read-back is undefined until then; `free_pinned` and `unregister_host` refuse an overlapping buffer meanwhile.

This step loads a slot of a cache on a copy stream while a compute stream runs, using an event to order them:

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

- `synchronize_stream`, `Transfer.flush`, `synchronize_event` and a `true` `event_query` release what the stream kept alive. Destroy streams you create: a dropped stream keeps its pending work rooted.
- Streams from `create_stream` are blocking by default: a synchronous transfer on the default stream serialises with them, so a streaming loop should not mix in automatic transfers.
- `create_stream ~non_blocking:true` makes a stream the default stream does not wait for (CUDA/HIP non-blocking flag; OpenCL, Metal and Vulkan queues are already independent; Native and Interpreter are synchronous). Order it against other streams only with `record_event` / `stream_wait_event`. The implicit paths (read-back, `sync`, `to_device`, default-stream launches via `Execute.run`/`run_vectors`/`run_source`, default-stream ranged copies and `stage_to_device`) wait on the host for a non-blocking stream that still references the vector, so they stay correct but do not overlap; operations given an explicit `~stream` (including `default_stream`) are never settled this way, and neither are host buffers (a pinned buffer used by a pending copy). A stream is meant to have one owning domain.
- Stream launches are never held in the [launch window](performance.html); they flush it first.

The core of the streaming interface as a signature that the compiler checks against `Transfer`:

```ocaml
module type Streaming = sig
  type stream
  type event

  val create_stream : ?non_blocking:bool -> Device.t -> stream
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

## Pinned host memory

Use `Transfer.alloc_pinned dev kind n` for page-locked memory as a Bigarray, so that a stream copy runs at the link rate. It is released only by `Transfer.free_pinned`; using the Bigarray or any sub-array afterwards is undefined, as after `munmap`. `Bigarray.float16` is refused: allocate a `char` buffer and use `copy_bytes_to_device`. HIP and CUDA may stage a stream copy from pageable memory through a driver bounce buffer and block the host call, so use pinned memory when overlap matters.

`Transfer.register_host dev ba` pins memory you already own, and `unregister_host` undoes it. On HIP, registration has these measured pitfalls (RX 7900 XTX, ROCm 7.2, 2026-10-10, under memory pressure from other workloads; `bench_hip_register_host`, raw output in `docs/measurements/transfer-pitfalls-2026-10-10/`):

- Copies right after registering are sometimes slow: 7 of 50 first passes ran at 0.95 to 6.17 GB/s, against about 26 GB/s otherwise, whatever the preparation (touched, untouched, populated). The effect is intermittent and state-dependent.
- A copy whose host range spans two registrations fails with `hipErrorInvalidValue`. If you register a large buffer in regions, slice every copy inside one region.
- ROCm does not count registered pages as pinned (`VmPin: 0 kB`). Whether that explains slow copies under memory pressure is not established. Memory from `alloc_pinned` is allocated by the driver instead.

## Sharing host memory with a CPU OpenCL device

By default a vector on a CPU OpenCL device (the Intel CPU runtime, for example) gets a buffer of its own, as on any other device: uploads and read-backs copy. To let kernels read and write the vector's host memory in place, create the buffer explicitly with `Spoc_core.Host_ptr_buffer` (OpenCL `CL_MEM_USE_HOST_PTR`):

```ocaml
module Host_ptr_buffer = Spoc_core.Host_ptr_buffer

(* [pool] is page-aligned host memory you own, possibly several GiB. *)
let region dev (pool : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t) ~off ~len =
  match Host_ptr_buffer.of_bigarray dev Vector.Float32 (Bigarray.Array1.sub pool off len) with
  | Ok v -> v
  | Error e -> failwith (Host_ptr_buffer.error_message e)
```

`Host_ptr_buffer.attach vec dev` does the same for an existing vector that has no buffer on `dev` yet. `Host_ptr_buffer.host_link vec dev` says what a vector's buffer is: `Separate`, `Host_storage` (Native, Interpreter) or `Host_ptr`.

- Devices: CPU-type OpenCL devices only. A GPU OpenCL device is refused with `Host_ptr_not_cpu_device`; CUDA, HIP, Vulkan, Metal and WebGPU with `Host_ptr_unsupported`. Native and Interpreter accept, since their device memory is the host array already.
- Alignment and size: the first byte must be a multiple of the device's `CL_DEVICE_MEM_BASE_ADDR_ALIGN` (128 bytes on the Intel CPU runtime; a 4096-aligned slice always qualifies), else `Host_ptr_misaligned`, because the runtime would keep a private copy. One buffer holds at most `CL_DEVICE_MAX_MEM_ALLOC_SIZE` bytes (`Host_ptr_too_large`); expose a larger region as several vectors over `Bigarray.Array1.sub` slices. Kernels index with int32, so keep each slice below 2^31 elements.
- Lifetime: the vector roots the Bigarray (a `sub` roots its parent). Releasing the buffer (`Transfer.free_buffer`, `Transfer.free_all_buffers` or the vector's finaliser) first waits for the device's queue, so no queued kernel still reads the memory once it can be collected. Memory you release yourself (`Transfer.free_pinned`, a C allocation viewed as a Bigarray) must outlive the buffer. The memory must stay mapped and in place while the vector exists. `Gpu_memory` counts the buffer's bytes, as it does for Native buffers: it reports bytes held in SPOC buffers, not device memory.
- Synchronisation: the array's contents at `attach` are what the first launch sees. After a launch, `Host_ptr_buffer.acquire_device_writes vec dev` (or any Vector read) maps the buffer for reading, which waits for the queued kernels, and the Bigarray then holds their writes. After writing the Bigarray directly, call `Host_ptr_buffer.publish_host_writes vec dev` before the next launch (map and unmap; it declares the host copy authoritative). `Vector.set` and `Transfer.to_device` go through the same path. Launches and maps of an OpenCL device share its default queue, so a map waits for every earlier launch. The Intel runtime happens to alias the memory without a map; SPOC does not rely on that.

Measured with `sarek/tests/e2e/bench_opencl_host_ptr_rss.exe` (Ryzen 9 7950X, Intel CPU OpenCL runtime 2026.21.3.0.31, 2026-10-11, shared machine, load average about 4.5; one run each; raw output in `docs/measurements/opencl-host-ptr-2026-10-11/`): a kernel reading every element of a 3 GiB array grew VmRSS by 116.6 MiB through a host-pointer buffer and by 3188.7 MiB through the default copy path; with 4 GiB, the host-pointer growth was 116.7 MiB.

## Small host writes to device vectors

`Vector.set` on a vector a kernel has used downloads the whole vector (after every queued kernel), and the next launch uploads it again with a synchronous copy that waits for the queue. Writing a few elements per iteration this way more than doubled the iteration time in the measurement below.

`Transfer.stage_to_device` avoids both waits for sources up to `Transfer.max_staged_bytes` (64 KiB): it copies the source into a 1 MiB pinned staging ring per device, allocated on first use and kept for the process lifetime, and enqueues the copy on the stream (default: the device's default stream). The source can be overwritten as soon as the call returns. It needs the same preconditions as `copy_to_device`; larger transfers belong to `alloc_pinned` and `copy_to_device`. Reusing one caller-owned pinned buffer with `copy_to_device ~stream` instead is a race: a queued copy reads whatever the buffer holds when it runs.

```ocaml
let publish_ids dev ids_vec (ids : (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t) =
  (* [ids_vec] must be device-current. The next kernel on the default stream
     sees [ids]; the host never waits for the GPU. *)
  Transfer.stage_to_device ids ~dst:ids_vec ~dst_offset:0
```

Measured with `sarek-hip/test/bench_hip_small_writes.exe` (RX 7900 XTX on the development machine, ROCm 7.2, 2026-10-10, shared machine; 8 int32 and 8 float32 rewritten, 300 us of host work, 4 launches per iteration; three runs; raw output in `docs/measurements/transfer-pitfalls-2026-10-10/bench_hip_small_writes.txt`), as the range over the three runs:

| Mode | us per iteration | of which in the writes |
|---|---|---|
| no rewrite | 322.0 to 322.5 | 0 |
| `Vector.set` | 705.2 to 715.1 | 341.3 to 346.4 |
| `stage_to_device` | 330.1 to 338.9 | 9.1 to 11.5 |

Known issue: on the Ryzen 9 7950X integrated GPU through HIP (ROCm 7.2), about 1 run in 7 of 300 staged writes interleaved with launches lost one launch. A caller-managed pinned ring shows it too, and `Gc.full_major ()` before the loop removed it in every run, so finaliser work during the loop is the lead; the mechanism is not established. The RX 7900 XTX never showed it (`stress_hip_staged_launches`, 20 runs per device, 2026-10-10: 3 of 20 runs on the integrated GPU, 0 of 20 on the RX 7900 XTX; raw output `docs/measurements/transfer-pitfalls-2026-10-10/stress_hip_staged_launches.txt`). Reproducer: `sarek-hip/test/stress_hip_staged_launches.ml`.

On backends without asynchronous copies (Native, Interpreter, Metal), `stage_to_device` is a synchronous ranged copy, complete on return.

## Pinned memory as an explicit handle: `Advanced.Pinned`

`Spoc_core.Advanced.Pinned` wraps `alloc_pinned` and `register_host` in a handle that owns the release. Nothing in SPOC pins memory unless you call it: it is opt-in, and so is every copy that uses the memory.

```ocaml
module Pinned = Spoc_core.Advanced.Pinned

let upload_through_pinned dev (dst : (int32, Bigarray.int32_elt) Vector.t) n =
  if not (Pinned.is_available dev) then invalid_arg "no pinned memory here" ;
  Pinned.with_alloc dev Bigarray.int32 n (fun staging ->
      let ba = Pinned.bigarray staging in
      for i = 0 to n - 1 do
        Bigarray.Array1.set ba i (Int32.of_int i)
      done ;
      (* This copy is synchronous. For a stream copy, pass the stream as
         [Pinned.with_alloc ~stream]: it is synchronised before the release. *)
      Transfer.copy_to_device ba ~dst ~dst_offset:0)
```

- `Pinned.alloc dev kind n` returns a handle on `n` elements of page-locked memory; `Pinned.register dev ba` page-locks an array you own. `Pinned.bigarray` gives the array, `Pinned.release` frees or unregisters it (idempotent), `Pinned.with_alloc` releases on return or exception, and `Pinned.is_live` says which state a handle is in.
- `with_alloc ?stream` synchronises `stream` before releasing. If a stream-ordered copy is still pending (other stream, no `?stream`), the release is refused, `Invalid_argument` is raised, and the allocation stays pinned for the process lifetime because the handle is unreachable: use `alloc` and `release` yourself when you need to retry.
- Release only through `Pinned.release`. Freeing or unregistering `Pinned.bigarray` with `Transfer` directly leaves the handle live, and its later `release` could act on another allocation at the same address.
- `Pinned.bigarray` on a released handle raises `Invalid_argument`. The array itself cannot be invalidated: do not keep it past `release`, as with `Transfer.free_pinned`.
- `release` is refused with `Invalid_argument`, and the handle stays live, while a stream-ordered copy may still use the memory. Synchronise the stream (or an event recorded after it) first.
- `Pinned.is_available dev` probes with a one-byte allocation. It costs a driver allocation and free per call on GPU backends. It answers `true` on Native, Interpreter and Metal, where pinned memory is plain host memory, and on backends whose driver allocates it; it answers `false` where the backend raises `Feature_not_supported`. Any other error propagates.
- The memory is a `Bigarray`, not a `Vector.t`: a vector's host mirror is garbage collected, pinned memory is not. `Bigarray.float16` is refused; use a `char` buffer and `copy_bytes_to_device`.
- The pitfalls of `Transfer.register_host` above apply to `Pinned.register`. Allocating multi-GiB pinned buffers can be slow on HIP: measure on the target machine, and compare with registering memory you already own.

## Do not use `Advanced.Unified`

`Spoc_core.Advanced.Unified.create` returns an ordinary pageable vector, and `Advanced.Unified.is_available` always returns `false`. It is a placeholder.

## See also

- [Launch window and fusion](performance.html): which operations flush deferred launches.
- [Device & Backend Selection](device_selection.html): choosing the device a vector lives on.
