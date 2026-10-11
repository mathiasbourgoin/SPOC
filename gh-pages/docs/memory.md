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

| Capability | HIP | CUDA | Metal | Native, Interpreter | OpenCL, Vulkan, WebGPU |
|---|---|---|---|---|---|
| Device-only vectors, ranged copies | yes | yes | yes (shared MTLBuffer, `memcpy`) | yes (`memmove`) | refused |
| Copies and launches on a stream | yes | yes | launches asynchronous on the stream's queue; copies complete on return | accepted, synchronous | refused |
| Events: create, record, synchronize | yes | yes | yes (newest command buffer of the stream) | trivially complete | yes (existing `Event`) |
| `stream_wait_event`, `event_query` | yes | yes | wait implied by buffer tracking; query real | trivially complete | refused |
| Pinned host memory (`alloc_pinned`) | `hipHostMalloc` | `cuMemHostAlloc` | page-aligned host memory | plain host memory | refused |
| `register_host` | `hipHostRegister` | `cuMemHostRegister` (not on ZLUDA) | no-op | no-op | refused |
| `stage_to_device` | asynchronous, staged | asynchronous, staged | synchronous ranged copy | synchronous ranged copy | refused |

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

On Metal every buffer is shared storage, so a ranged copy is one `memcpy`. Launches are batched into one command buffer per queue, committed when the previous command buffer has completed or once the batch holds `SPOC_METAL_BATCH` dispatches (default 32), and otherwise by the next launch, wait, event record or synchronisation. A host access to a buffer first waits for the pending command buffers that bind it. A buffer larger than the device's `maxBufferLength` is refused with an error naming the limit.

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
- Streams from `create_stream` are blocking, and the auto-sync path depends on that. A synchronous transfer on the default stream serialises with them, so a streaming loop should not mix in automatic transfers.
- Stream launches are never held in the [launch window](performance.html); they flush it first.

The core of the streaming interface as a signature that the compiler checks against `Transfer`:

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

## Pinned host memory

Use `Transfer.alloc_pinned dev kind n` for page-locked memory as a Bigarray, so that a stream copy runs at the link rate. It is released only by `Transfer.free_pinned`; using the Bigarray or any sub-array afterwards is undefined, as after `munmap`. `Bigarray.float16` is refused: allocate a `char` buffer and use `copy_bytes_to_device`. HIP and CUDA may stage a stream copy from pageable memory through a driver bounce buffer and block the host call, so use pinned memory when overlap matters.

`Transfer.register_host dev ba` pins memory you already own, and `unregister_host` undoes it. On HIP, registration has these measured pitfalls (RX 7900 XTX, ROCm 7.2, 2026-10-10, under memory pressure from other workloads; `bench_hip_register_host`, raw output in `docs/measurements/transfer-pitfalls-2026-10-10/`):

- Copies right after registering are sometimes slow: 7 of 50 first passes ran at 0.95 to 6.17 GB/s, against about 26 GB/s otherwise, whatever the preparation (touched, untouched, populated). The effect is intermittent and state-dependent.
- A copy whose host range spans two registrations fails with `hipErrorInvalidValue`. If you register a large buffer in regions, slice every copy inside one region.
- ROCm does not count registered pages as pinned (`VmPin: 0 kB`). Whether that explains slow copies under memory pressure is not established. Memory from `alloc_pinned` is allocated by the driver instead.

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

## Do not use `Advanced.Pinned` or `Advanced.Unified`

`Spoc_core.Advanced.Pinned.create` and `Advanced.Unified.create` return ordinary pageable vectors, and their `is_available` functions always return `false`. They are placeholders. For page-locked memory, use `Transfer.alloc_pinned` or `Transfer.register_host` as above.

## See also

- [Launch window and fusion](performance.html): which operations flush deferred launches.
- [Device & Backend Selection](device_selection.html): choosing the device a vector lives on.
