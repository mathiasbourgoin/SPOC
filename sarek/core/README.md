# sarek/core - Core Runtime Abstractions

The `sarek/core` package provides the foundational runtime abstractions for Sarek GPU computing. It sits between the Sarek PPX (IR generation) and backend plugins (CUDA/OpenCL/Vulkan/Native/Interpreter), providing type-safe device management, vector storage, memory transfers, and kernel execution.

**Key responsibilities:**
- Device enumeration and abstraction across backends
- Type-safe vector storage with location tracking (CPU/GPU)
- Memory buffer management with zero-copy for CPU backends
- Explicit and automatic data transfer control
- Unified logging and error handling

## Table of Contents

- [Quick Start](#quick-start)
- [Core Abstractions](#core-abstractions)
- [Module Overview](#module-overview)
- [Usage Examples](#usage-examples)
- [Design Principles](#design-principles)
- [Testing](#testing)
- [Debugging](#debugging)

## Quick Start

```ocaml
open Sarek_core

(* Initialize devices *)
let () = Device.init () |> ignore

(* Get first GPU device *)
let dev = Device.get_default () in

(* Create vectors *)
let a = Vector.create_float32 1024 in
let b = Vector.create_float32 1024 in
let c = Vector.create_float32 1024 in

(* Initialize data on CPU *)
Vector.init_float32 a (fun i -> float_of_int i);
Vector.init_float32 b (fun i -> float_of_int (i * 2));

(* Explicit transfer to device *)
Transfer.to_device a dev;
Transfer.to_device b dev;
Transfer.to_device c dev;

(* Kernel execution happens here (via sarek/sarek Execute module) *)
(* ... *)

(* Explicit transfer back to CPU *)
Transfer.to_cpu c;

(* Access results *)
Printf.printf "c[10] = %f\n" (Vector.get_float32 c 10)
```

## Core Abstractions

### Device Abstraction

Devices represent compute units (GPUs, CPUs) from any backend:

```ocaml
type Device.t = {
  id : int;                              (* Global device ID *)
  backend_id : int;                      (* Backend-specific ID *)
  name : string;                         (* Human-readable name *)
  framework : string;                    (* "CUDA", "OpenCL", etc. *)
  capabilities : Framework_sig.capabilities;
}
```

**Capabilities** let you query device features:
- `is_cpu` / `is_gpu` - Device type
- `supports_fp64` - Double precision support
- `supports_atomics` - Atomic operations
- `warp_size` - Warp/wavefront size
- `max_work_group_size` - Max threads per block

### Vector Abstraction

Vectors track data location and handle synchronization:

```ocaml
type ('a, 'b) Vector.t = {
  length : int;
  kind : ('a, 'b) kind;              (* Scalar or Custom type *)
  location : location;                (* Where data lives *)
  host : host_storage;                (* CPU-side storage *)
  buffers : device_buffer list;       (* GPU buffers (lazy) *)
}

type location =
  | CPU                                 (* Data only on CPU *)
  | GPU of Device.t                    (* Data only on device *)
  | Both of Device.t                   (* Synchronized on both *)
  | Stale_CPU of Device.t              (* Device has newer data *)
  | Stale_GPU of Device.t              (* CPU has newer data *)
```

**Two storage modes:**
- **Bigarray**: Built-in types (int32, float32, float64, etc.)
- **Custom**: User-defined ctypes structures

### Memory Buffers

Device buffers encapsulate backend-specific memory:

```ocaml
module type Vector.DEVICE_BUFFER = sig
  val device : Device.t
  val size : int                       (* Number of elements *)
  val elem_size : int                  (* Bytes per element *)
  val device_ptr : nativeint          (* Pointer for kernel args *)
  val bind_to_kargs : kargs -> int -> unit
  val host_ptr_to_device : nativeint -> byte_size:int -> unit
  val device_to_host_ptr : nativeint -> byte_size:int -> unit
  val free : unit -> unit
end
```

Zero-copy used for CPU backends when possible.

## Module Overview

### Device Management

#### `Device.ml` - Device Enumeration (236 lines)

Discovers and manages compute devices across all backends.

**Key functions:**
```ocaml
val init : ?frameworks:string list -> unit -> Device.t array
  (* Initialize backends and enumerate devices *)

val all : unit -> Device.t array
  (* Get all initialized devices *)

val get_default : unit -> Device.t
  (* Get default device (first GPU, or first CPU if no GPU) *)

val get : int -> Device.t
  (* Get device by global ID *)

val is_cpu : Device.t -> bool
val is_gpu : Device.t -> bool
  (* Query device type *)

val allows_fp64 : Device.t -> bool
  (* Check double precision support *)

val filter : (Device.t -> bool) -> Device.t array
  (* Filter devices by predicate *)
```

**Device initialization order:**
1. Try each framework in order: CUDA → OpenCL → Vulkan → Metal → Native → Interpreter
2. Skip unavailable backends
3. Assign sequential global IDs
4. Return array of all discovered devices

### Vector Storage & Access

#### `Vector.ml` - High-Level Vector Interface (512 lines)

Unified vector abstraction with location tracking.

**Creation:**
```ocaml
val create_scalar : 
  ('a, 'b) scalar_kind -> int -> ('a, 'b) t

val create_float32 : int -> (float, float32_elt) t
val create_float64 : int -> (float, float64_elt) t
val create_int32 : int -> (int32, int32_elt) t
(* Convenience constructors for common types *)

val create_custom : 
  (module CUSTOM_TYPE with type t = 'a) -> 
  int -> ('a, unit) t
  (* Create vector of custom ctypes structures *)

val of_bigarray : 
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> 
  ('a, 'b) scalar_kind -> 
  ('a, 'b) t
  (* Wrap existing bigarray as vector *)
```

**Element access:**
```ocaml
val get : ('a, 'b) t -> int -> 'a
  (* Get element (auto-syncs if needed) *)

val set : ('a, 'b) t -> int -> 'a -> unit
  (* Set element (marks CPU as authoritative) *)

val get_float32 : (float, float32_elt) t -> int -> float
val set_float32 : (float, float32_elt) t -> int -> float -> unit
  (* Type-specific convenience functions *)
```

**Initialization:**
```ocaml
val init_float32 : (float, float32_elt) t -> (int -> float) -> unit
val fill_float32 : (float, float32_elt) t -> float -> unit
  (* Initialize vector with function or constant *)
```

**Subvectors:**
```ocaml
val sub_vector : ('a, 'b) t -> start:int -> len:int -> unit -> ('a, 'b) t
  (* Create view into parent vector (shares storage) *)

val partition : ('a, 'b) t -> int -> ('a, 'b) t array
  (* Split into N equal chunks *)
```

**Metadata:**
```ocaml
val length : ('a, 'b) t -> int
val location : ('a, 'b) t -> location
val kind : ('a, 'b) t -> ('a, 'b) kind
val device : ('a, 'b) t -> Device.t option
```

#### `Vector_types.ml` - Type Definitions (224 lines)

Core type definitions for vectors:
- `scalar_kind` - Bigarray kinds (Int32, Float32, Float64, etc.)
- `custom_type` - User-defined structure metadata
- `kind` - Scalar or Custom GADT
- `location` - Where data resides
- `host_storage` - Bigarray or Custom storage

#### Storage Management — `Spoc_core_base.Make(Ctypes_ops)`

Vector creation, allocation, copying, and sub-vector operations live in the pure
`spoc_core_base` functor (`sarek/core_base/Spoc_core_base.ml`), instantiated
natively as `Vector_types = Spoc_core_base.Make(Ctypes_ops)` and re-exported
through `Vector`. (The numeric path is FFI-free; the browser build instantiates
the same functor with `Js_ops`.)

#### `Vector_transfer.ml` - Sync Callbacks (48 lines)

Synchronization hooks for element access:
- Host pointer extraction
- Auto-sync on get/set
- Custom sync callbacks

### Memory Management

#### `Memory.ml` - Buffer Abstraction (190 lines)

Type-safe buffer operations.

**Buffer module type:**
```ocaml
module type BUFFER = sig
  val device : Device.t
  val size : int
  val elem_size : int
  val device_ptr : nativeint
  val bind_to_kargs : Framework_sig.kargs -> int -> unit
  val host_ptr_to_device : nativeint -> byte_size:int -> unit
  val device_to_host_ptr : nativeint -> byte_size:int -> unit
  val free : unit -> unit
end
```

**Functions:**
```ocaml
val alloc_scalar : 
  Device.t -> int -> ('a, 'b) Vector.scalar_kind -> 
  (module BUFFER)

val alloc_custom :
  Device.t -> int -> elem_size:int ->
  (module BUFFER)

val to_device : (module BUFFER) -> ('a, 'b) Vector.t -> unit
val from_device : (module BUFFER) -> ('a, 'b) Vector.t -> unit
```

Zero-copy used when:
- Device is CPU-type (OpenCL CPU, Native)
- Backend supports `alloc_zero_copy`
- Vector uses Bigarray storage

### Data Transfer

#### `Transfer.ml` - Transfer Control (451 lines)

Explicit and automatic data movement between CPU and device.

**Auto-transfer mode:**
```ocaml
val enable_auto : unit -> unit
val disable_auto : unit -> unit
val is_auto : unit -> bool
  (* Control automatic synchronization *)
```

**Explicit transfers:**
```ocaml
val to_device : ('a, 'b) Vector.t -> Device.t -> unit
  (* CPU → Device *)

val to_cpu : ('a, 'b) Vector.t -> unit
  (* Device → CPU *)

val ensure_on_device : ('a, 'b) Vector.t -> Device.t -> unit
  (* Transfer only if not already there *)

val ensure_on_cpu : ('a, 'b) Vector.t -> unit
  (* Transfer only if stale on CPU *)
```

**Buffer management:**
```ocaml
val alloc_scalar_buffer : 
  Device.t -> int -> ('a, 'b) Vector.scalar_kind -> 
  Vector.device_buffer

val alloc_scalar_buffer_zero_copy :
  Device.t -> int -> ('a, 'b) Vector.scalar_kind ->
  Vector.device_buffer option
  (* Returns None if zero-copy not supported *)

val get_or_alloc_buffer :
  ('a, 'b) Vector.t -> Device.t -> Vector.device_buffer
  (* Get existing buffer or allocate new one *)
```

**Transfer flow:**
1. Check if transfer needed (location tracking)
2. Allocate device buffer if needed
3. Get host pointer from vector storage
4. Call backend's transfer operation
5. Update vector location

**Zero-copy path:**
- Backend allocates pinned/shared memory
- Host pointer = device pointer
- Transfers become no-ops
- Used automatically for CPU backends

#### Device-only vectors, ranged and stream-ordered copies, pinned memory

Added for streaming workloads (spoc_llm M3: MoE expert weights copied into a
VRAM-resident cache while the GPU computes). Spec:
`specs/device-buffers-async-copy.md`.

| Capability | HIP | CUDA | Metal | Native / Interpreter | OpenCL, Vulkan, WebGPU |
|---|---|---|---|---|---|
| device-only vectors, ranged copies | yes | yes | yes (shared MTLBuffer, memcpy) | yes (memmove) | refused |
| copies / launches on a stream | yes | yes | launches asynchronous on the stream's queue; copies complete on return | accepted, synchronous | refused |
| events: create / record / synchronize | yes | yes | yes (newest command buffer of the stream) | trivially complete | yes (existing `Event`) |
| `stream_wait_event`, `event_query` | yes | yes | wait implied by buffer tracking; query real | trivially complete | refused |
| pinned host memory | `hipHostMalloc` (portable) | `cuMemHostAlloc` (portable) | page-aligned host memory | plain host memory | refused |
| `register_host` | `hipHostRegister` | `cuMemHostRegister` (not on ZLUDA) | no-op | no-op | refused |
| `stage_to_device` | asynchronous, staged | asynchronous, staged | synchronous ranged copy | synchronous ranged copy | refused |

Metal (spec `specs/metal-ranged-copies.md`): every buffer is
`MTLResourceStorageModeShared`; on Apple silicon CPU and GPU share one physical
memory, so a ranged copy is one `memcpy` into the buffer's `contents`. Kernel
launches are asynchronous: dispatches are batched into one command buffer per
queue; at each launch the batch is committed if the queue's previous command
buffer has completed or once it holds `SPOC_METAL_BATCH` (default 32)
dispatches, otherwise by the next launch, wait, event record or
synchronisation. Every host access to a buffer (copy, read-back, upload) first
waits for the pending command buffers that bind it, and a launch waits for
pending work on other queues that binds its buffers, so a copy never races a
kernel and always sees what it wrote. A buffer larger than the device's
`maxBufferLength` is refused with an error naming the limit;
`capabilities.total_global_mem` reports `recommendedMaxWorkingSetSize`. A
failed command buffer raises `kernel_launch_failed`.

"Refused" is `Backend_error (Plugin {error = Feature_not_supported _})`; caller
errors (bounds, wrong device, wrong location, freeing in-flight memory) raise
`Invalid_argument`. Nothing silently falls back.

```ocaml
(* Device-only: no host mirror on a GPU. Host access refuses. *)
val create_device_only : Device.t -> ('a, 'b) Vector.kind -> int -> ('a, 'b) Vector.t
val Vector.is_device_only : ('a, 'b) Vector.t -> bool

(* Ranged copies. Without ~stream: complete on return. With it: enqueued;
   SPOC roots the Bigarrays and vectors until the stream (or a later event)
   is synchronised. *)
val copy_to_device : ?stream:stream -> ('a, 'b, c_layout) Bigarray.Array1.t ->
  dst:('a, 'b) Vector.t -> dst_offset:int -> unit          (* elements *)
val copy_bytes_to_device : ?stream:stream -> (_, _, c_layout) Bigarray.Array1.t ->
  dst:(_, _) Vector.t -> dst_byte_offset:int -> unit       (* bytes *)
val copy_ptr_to_device : ?stream:stream -> src_ptr:nativeint -> byte_size:int ->
  dst:(_, _) Vector.t -> dst_byte_offset:int -> unit -> unit  (* caller roots src *)
val copy_to_host : ?stream:stream -> src:('a, 'b) Vector.t -> src_offset:int ->
  ('a, 'b, c_layout) Bigarray.Array1.t -> unit                (* elements *)
val copy_bytes_to_host : ?stream:stream -> src:(_, _) Vector.t ->
  src_byte_offset:int -> (_, _, c_layout) Bigarray.Array1.t -> unit  (* bytes *)

(* Pinned host memory, explicit lifetime (like munmap). *)
val alloc_pinned : Device.t -> ('a, 'b) Bigarray.kind -> int -> ('a, 'b, c_layout) Bigarray.Array1.t
val free_pinned : (_, _, c_layout) Bigarray.Array1.t -> unit
val register_host : Device.t -> (_, _, c_layout) Bigarray.Array1.t -> unit
val unregister_host : (_, _, c_layout) Bigarray.Array1.t -> unit

(* Small writes (<= max_staged_bytes = 64 KiB) to a vector kernels read,
   without waiting for the GPU; the source may be reused on return. *)
val stage_to_device : ?stream:stream -> ('a, 'b, c_layout) Bigarray.Array1.t ->
  dst:('a, 'b) Vector.t -> dst_offset:int -> unit
val is_pinned : (_, _, c_layout) Bigarray.Array1.t -> bool

(* Events; Execute.run / run_vectors take ?stream. *)
val create_event : Device.t -> event
val record_event : event -> stream -> unit
val stream_wait_event : stream -> event -> unit
val event_query : event -> bool
val synchronize_event : event -> unit
val destroy_event : event -> unit
```

The streaming pattern:

```ocaml
let copy = Transfer.create_stream dev and compute = Transfer.create_stream dev in
let staged = Transfer.alloc_pinned dev Bigarray.char slot_bytes in   (* pread into it *)
Transfer.copy_bytes_to_device ~stream:copy staged ~dst:cache ~dst_byte_offset:(slot * slot_bytes) ;
Transfer.record_event ready copy ;
Transfer.stream_wait_event compute ready ;
Execute.run_vectors ~stream:compute ~device:dev ~ir ~args:[Vec cache; Vec x; Vec y] ~block ~grid ()
```

Rules worth knowing:

- A ranged copy needs the vector's device copy to be current (`GPU`, `Both`,
  `Stale_CPU` on that device, or device-only); otherwise call `to_device`
  first. A ranged write turns `Both` into `Stale_CPU`, so auto-sync still
  reads the right data. Custom (record) kinds are refused.
- A read-back (`copy_to_host`, `copy_bytes_to_host`) leaves the vector's
  location unchanged. `copy_bytes_to_host` copies `size_in_bytes dst` bytes
  whatever the two element kinds are (e.g. an `int32` vector range into a
  `char` slot). With `~stream`, the destination is undefined until the copy
  has completed: reading it may see old or partial data, writing it races
  with the DMA, and `free_pinned`/`unregister_host` refuse it meanwhile.
- On CPU devices device memory *is* host memory: a device-only vector keeps
  host storage there and host access works. Do not rely on it.
- Freeing a device-only vector discards its contents. Freeing anything a
  pending stream operation references is refused.
- The keep-alive ledger is released by `synchronize_stream`, `flush`,
  `synchronize_event`/a `true` `event_query`, and transitively: synchronising
  a stream that waited on an event releases the work that event covered.
- Streams from `create_stream` are blocking streams; the auto-sync path
  depends on it. Any synchronous transfer on the default stream serialises
  with them (HIP/CUDA legacy default-stream semantics), so a streaming loop
  should not mix in auto-transfers.
- A stream dropped without `destroy_stream` keeps everything its pending
  work references rooted (and its backend stream alive): destroy streams.
- Vector lengths are capped at `Int32.max_int` elements (kernels index with
  int32); `Execute` refuses longer vectors instead of wrapping.
- Vector location and buffer tables are not synchronised across domains:
  use a vector from one domain at a time. The ledger and pinned registry are
  mutex-protected.

#### Small host writes to device-resident vectors

`Vector.set` on a vector a GPU kernel has used is expensive. Every launch
leaves its vector arguments `Stale_CPU`, so the first `set` downloads the
whole vector (after every queued kernel), and the next launch re-uploads it
with a synchronous copy that waits for the queue again. Rewriting a few
routing indices per layer this way serialises host and GPU.
`Transfer.stage_to_device` is the cheap path: it copies the source into a
1 MiB per-device pinned staging ring and enqueues the copy on the stream
(default stream by default), so it neither waits for the GPU nor requires the
caller to keep the source unchanged. Reusing one caller-owned pinned buffer
with `copy_to_device ~stream` instead is a race: a copy still queued reads
whatever the buffer holds when it runs (seen as wrong results in a first
version of the benchmark).

`sarek-hip/test/bench_hip_small_writes.exe` (RX 7900 XTX, ROCm 7.2,
2026-10-10, shared machine; 8 int32 + 8 float32 rewritten, 300 us host work,
4 launches per iteration; three runs):

| Mode | us / iteration | of which in the writes |
|---|---|---|
| no rewrite | 320–324 | 0 |
| `Vector.set` | 709–712 | 339–344 |
| `stage_to_device` | 333–343 | 10–13 |

#### Registering host memory on ROCm

Measured with `sarek-hip/test/bench_hip_register_host.exe` and earlier
versions of it (raw output in `docs/measurements/transfer-pitfalls-2026-10-10/`; 8 GiB,
synchronous 5.5 MiB copies; tartarus 2026-10-10, under memory pressure from
other workloads). Details and caveats are on `Transfer.register_host`.

- Copies right after registering are sometimes slow: 7 of 50 first passes
  ran at 0.95–6.17 GB/s (about 26 otherwise), whatever the preparation —
  memory written before registering, never touched, written after
  registering, or faulted in first with `madvise(MADV_POPULATE_WRITE)`.
  spoc_llm attributed its 0.45 GB/s to registering untouched memory; these
  runs do not support that, only an intermittent, state-dependent effect.
- A copy may not span two registrations: `hipErrorInvalidValue`.
- One large registration slowing to 2.3 GB/s after CPU writes was observed by
  spoc_llm once and did not reproduce (26.3 GB/s); measure region sizes on
  the target machine.
- ROCm does not count registered pages as pinned (`VmPin: 0 kB`); whether that
  explains slow copies under memory pressure is not established.

### Kernel Execution

#### `Kernel.ml` - Kernel Compilation & Launch (170 lines)

Type-safe kernel operations.

**Kernel module type:**
```ocaml
module type KERNEL = sig
  type t
  val device : t -> Device.t
  val launch : 
    t -> 
    args:(module ARGS) -> 
    grid:Framework_sig.dims -> 
    block:Framework_sig.dims -> 
    shared_mem:int -> 
    unit -> unit
  val profile_launch : ... -> float
  val free : t -> unit
end
```

**Functions:**
```ocaml
val compile_source : 
  Device.t -> 
  source:string -> 
  lang:Framework_sig.source_lang -> 
  kernel_name:string -> 
  (module KERNEL)

val launch :
  (module KERNEL) ->
  args:(module ARGS) ->
  grid:Framework_sig.dims ->
  block:Framework_sig.dims ->
  ?shared_mem:int ->
  unit -> unit
```

#### `Kernel_arg.ml` - Type-Safe Arguments (119 lines)

GADT-based kernel argument handling.

**Argument types:**
```ocaml
type _ t =
  | Vec : ('a, 'b) Vector.t -> ('a, 'b) Vector.t t
  | Scalar_Int : int -> int t
  | Scalar_Int32 : int32 -> int32 t
  | Scalar_Int64 : int64 -> int64 t
  | Scalar_Float32 : float -> float t
  | Scalar_Float64 : float -> float t

type any = Any : 'a t -> any
```

No `Obj.t` - types preserved through compilation.

### Runtime Helpers

#### `Runtime.ml` - High-Level Convenience (127 lines)

Convenience wrappers for common operations.

**Dimension helpers:**
```ocaml
val dims1d : int -> Framework_sig.dims
val dims2d : int -> int -> Framework_sig.dims
val dims3d : int -> int -> int -> Framework_sig.dims

val grid_for_size : problem_size:int -> block_size:int -> int
  (* Calculate grid size: (problem_size + block_size - 1) / block_size *)
```

**Memory allocation shortcuts:**
```ocaml
val alloc_float32 : Device.t -> int -> (module Memory.BUFFER)
val alloc_float64 : Device.t -> int -> (module Memory.BUFFER)
val alloc_int32 : Device.t -> int -> (module Memory.BUFFER)
```

**Transfer shortcuts:**
```ocaml
val to_device : (module Memory.BUFFER) -> 
  (float, float32_elt) Vector.t -> unit
val from_device : (module Memory.BUFFER) -> 
  (float, float32_elt) Vector.t -> unit
```

### Debugging & Monitoring

#### `Log.ml` - Unified Logging (137 lines)

Component-based debug logging controlled by `SAREK_DEBUG` environment variable.

**Log components:**
```ocaml
type component =
  | Device    (* Device enumeration *)
  | Memory    (* Buffer allocation/free *)
  | Transfer  (* Data movement *)
  | Kernel    (* Kernel compilation/launch *)
  | Execute   (* Execution dispatch *)
```

**Usage:**
```bash
# Enable specific components
SAREK_DEBUG=transfer,kernel ./my_program

# Enable all debug output
SAREK_DEBUG=all ./my_program
```

**Functions:**
```ocaml
val debugf : component -> ('a, unit, string, unit) format4 -> 'a
val infof : component -> ('a, unit, string, unit) format4 -> 'a
val warnf : component -> ('a, unit, string, unit) format4 -> 'a
val errorf : component -> ('a, unit, string, unit) format4 -> 'a
```

Example output:
```
[Device] Framework CUDA: 2 device(s)
[Device]   [0] NVIDIA GeForce RTX 3090 (CUDA)
[Transfer] host_ptr_to_device: ptr=139824567890432 size=4096
[Kernel] Compiling kernel 'vector_add' for device 0
```

#### `Error.ml` - Error Handling (222 lines)

Structured error types for runtime failures.

#### `Profiling.ml` - Performance Monitoring (230 lines)

Kernel profiling and timing utilities.

#### `Advanced.ml` - Advanced Features (242 lines)

Advanced memory operations and optimizations.

## Usage Examples

### Basic Vector Operations

```ocaml
open Sarek_core

let () =
  (* Create and initialize vector *)
  let v = Vector.create_float32 1024 in
  Vector.init_float32 v (fun i -> float_of_int i);
  
  (* Element access *)
  let x = Vector.get_float32 v 10 in
  Printf.printf "v[10] = %f\n" x;
  
  (* Modify element *)
  Vector.set_float32 v 10 99.0;
  
  (* Copy vector *)
  let v2 = Vector.copy v in
  
  (* Create subvector (shares storage) *)
  let sub = Vector.sub_vector v ~start:100 ~len:50 () in
  Printf.printf "Subvector length: %d\n" (Vector.length sub)
```

### Device Selection

```ocaml
(* Get all GPUs *)
let gpus = Device.filter Device.is_gpu (Device.all ()) in
Printf.printf "Found %d GPU(s)\n" (Array.length gpus);

(* Get first device with FP64 support *)
let fp64_dev = 
  Array.find Device.allows_fp64 (Device.all ()) in
Printf.printf "FP64 device: %s\n" fp64_dev.name;

(* Get specific backend *)
let cuda_devices = 
  Device.filter (fun d -> d.framework = "CUDA") (Device.all ()) in

(* Check capabilities *)
let dev = Device.get_default () in
let caps = dev.capabilities in
if caps.supports_atomics then
  Printf.printf "Device supports atomic operations\n";
Printf.printf "Warp size: %d\n" caps.warp_size
```

### Explicit Transfer Control

```ocaml
(* Disable auto-transfer for manual control *)
Transfer.disable_auto ();

let dev = Device.get_default () in
let a = Vector.create_float32 1024 in
let b = Vector.create_float32 1024 in
let c = Vector.create_float32 1024 in

(* Initialize on CPU *)
Vector.init_float32 a (fun i -> float_of_int i);
Vector.init_float32 b (fun i -> float_of_int (i * 2));

(* Explicit transfer to device *)
Transfer.to_device a dev;
Transfer.to_device b dev;
Transfer.to_device c dev;

(* Run kernel (via sarek/sarek Execute module) *)
(* ... kernel execution ... *)

(* Explicit transfer back *)
Transfer.to_cpu c;

(* Access results (no auto-sync needed) *)
let result = Vector.get_float32 c 10 in
Printf.printf "Result: %f\n" result
```

### Zero-Copy Optimization

```ocaml
(* Zero-copy for CPU backends *)
let cpu_dev = 
  Array.find Device.is_cpu (Device.all ()) in

let v = Vector.create_float32 1024 in
Vector.init_float32 v (fun i -> float_of_int i);

(* Transfer uses zero-copy (no actual copy) *)
Transfer.to_device v cpu_dev;

(* Modifications visible on both sides *)
Vector.set_float32 v 0 999.0;
(* Device sees updated value immediately *)
```

### Custom Types

```ocaml
open Ctypes

(* Define custom struct *)
type particle = {
  x : float;
  y : float;
  vx : float;
  vy : float;
  mass : float;
}

(* Create ctypes structure *)
let particle_struct : particle structure typ = structure "particle"
let x_field = field particle_struct "x" float
let y_field = field particle_struct "y" float
let vx_field = field particle_struct "vx" float
let vy_field = field particle_struct "vy" float
let mass_field = field particle_struct "mass" float
let () = seal particle_struct

(* Custom type module *)
module Particle_custom = struct
  type t = particle
  let typ = particle_struct
  let size = sizeof particle_struct
  
  let get ptr =
    let s = !@ (from_voidp particle_struct (to_voidp ptr)) in
    {
      x = getf s x_field;
      y = getf s y_field;
      vx = getf s vx_field;
      vy = getf s vy_field;
      mass = getf s mass_field;
    }
  
  let set ptr p =
    let s_ptr = from_voidp particle_struct (to_voidp ptr) in
    setf (!@ s_ptr) x_field p.x;
    setf (!@ s_ptr) y_field p.y;
    setf (!@ s_ptr) vx_field p.vx;
    setf (!@ s_ptr) vy_field p.vy;
    setf (!@ s_ptr) mass_field p.mass
end

(* Create vector of custom type *)
let particles = Vector.create_custom (module Particle_custom) 10000 in

(* Initialize *)
for i = 0 to 9999 do
  Vector.set particles i {
    x = Random.float 100.0;
    y = Random.float 100.0;
    vx = Random.float 10.0;
    vy = Random.float 10.0;
    mass = 1.0;
  }
done
```

### Memory Buffer Management

```ocaml
(* Allocate buffer *)
let dev = Device.get_default () in
let buf = Memory.alloc_scalar dev 1024 Vector.Float32 in

(* Transfer data *)
let v = Vector.create_float32 1024 in
Vector.init_float32 v (fun i -> float_of_int i);
Memory.to_device buf v;

(* Use in kernel *)
let module B = (val buf : Memory.BUFFER) in
Printf.printf "Device pointer: %Ld\n" (Int64.of_nativeint B.device_ptr);
B.bind_to_kargs kargs 0;

(* Transfer back *)
Memory.from_device buf v;

(* Free *)
let module B = (val buf : Memory.BUFFER) in
B.free ()
```

### Subvectors and Partitioning

```ocaml
(* Create large vector *)
let v = Vector.create_float32 10000 in
Vector.init_float32 v (fun i -> float_of_int i);

(* Create subvector (view, shares storage) *)
let sub = Vector.sub_vector v ~start:1000 ~len:100 () in
Printf.printf "Subvector length: %d\n" (Vector.length sub);

(* Modifications to sub affect parent *)
Vector.set_float32 sub 0 999.0;
assert (Vector.get_float32 v 1000 = 999.0);

(* Partition into chunks for multi-device *)
let chunks = Vector.partition v 4 in
Array.iteri (fun i chunk ->
  Printf.printf "Chunk %d: length=%d\n" i (Vector.length chunk)
) chunks;

(* Process chunks on different devices *)
let devices = Device.all () in
Array.iteri (fun i chunk ->
  if i < Array.length devices then begin
    let dev = devices.(i) in
    Transfer.to_device chunk dev;
    (* Run kernel on chunk *)
  end
) chunks
```

## Design Principles

### 1. Type Safety Without Obj.t

The core uses GADTs, first-class modules, and existential types instead of `Obj.t`:

```ocaml
(* Good: GADT preserves element type *)
type _ kind =
  | Scalar : ('a, 'b) scalar_kind -> ('a, 'b) kind
  | Custom : 'a custom_type -> ('a, unit) kind

type (_, _) t = {
  kind : ('a, 'b) kind;
  ...
}

(* Good: First-class module for buffers *)
module type DEVICE_BUFFER = sig ... end
type device_buffer = (module DEVICE_BUFFER)

(* Bad: Loses type information *)
type buffer = Obj.t
```

### 2. Backend-Agnostic Core

Core modules never import backend-specific code:

```ocaml
(* Good: Uses plugin registry *)
match Framework_registry.find_backend "CUDA" with
| Some (module B : Framework_sig.BACKEND) ->
    B.Device.init ()
| None -> (* fallback *)

(* Bad: Direct backend import *)
open Cuda_backend  (* Never do this in core *)
```

All backend interaction goes through:
- `Framework_registry` - Plugin discovery
- `Framework_sig.BACKEND` - Unified interface

### 3. Location Tracking

Vectors track where data lives to optimize transfers:

```ocaml
match vec.location with
| CPU -> (* Need to transfer to device *)
    Transfer.to_device vec dev
| GPU d when d = dev -> (* Already there *)
    ()
| GPU d -> (* On wrong device *)
    Transfer.to_cpu vec;
    Transfer.to_device vec dev
| Both d when d = dev -> (* Synchronized *)
    ()
| Stale_CPU d when d = dev -> (* Device is newer *)
    if auto_sync then Transfer.to_cpu vec
| Stale_GPU d -> (* CPU is newer *)
    Transfer.to_device vec d
```

### 4. Lazy Allocation

Device buffers allocated on first use:

```ocaml
let to_device (vec : ('a, 'b) t) (dev : Device.t) =
  (* Check if buffer exists *)
  let buf = match find_buffer vec dev with
    | Some b -> b  (* Reuse existing *)
    | None -> 
        (* Allocate on first use *)
        let b = alloc_buffer dev vec.length vec.kind in
        vec.buffers <- b :: vec.buffers;
        b
  in
  (* Transfer data *)
  transfer_to_buffer vec buf
```

### 5. Zero-Copy When Possible

CPU backends use shared memory:

```ocaml
let alloc_scalar_buffer dev len kind =
  match Framework_registry.find_backend dev.framework with
  | Some (module B) ->
      (* Try zero-copy first *)
      match B.Memory.alloc_zero_copy dev len kind with
      | Some buf -> buf  (* Host and device share memory *)
      | None -> B.Memory.alloc dev len kind  (* Fallback *)
```

Transfers check zero-copy flag:

```ocaml
let host_ptr_to_device src_ptr ~byte_size =
  if B.Memory.is_zero_copy buf then () (* Skip transfer *)
  else B.Memory.host_ptr_to_device ~src_ptr ~byte_size ~dst:buf
```

## Testing

The core has comprehensive unit tests covering all modules:

```bash
# Run all core tests
dune runtest sarek/core/test

# Run specific test module
dune exec sarek/core/test/test_device.exe
dune exec sarek/core/test/test_vector.exe
dune exec sarek/core/test/test_transfer.exe
```

**Test coverage:**
- `test_device.ml` - Device enumeration, predicates, capabilities
- `test_vector.ml` - Creation, access, subvectors, partitioning
- `test_vector_storage.ml` - Storage allocation, copying
- `test_vector_transfer.ml` - Sync callbacks, host pointers
- `test_memory.ml` - Buffer allocation, transfers
- `test_kernel.ml` - Kernel compilation, argument binding
- `test_kernel_arg.ml` - GADT argument handling

Run with coverage:

```bash
BISECT_FILE=_coverage/core dune runtest sarek/core/test --force
bisect-ppx-report html -o _coverage/core-report
```

## Debugging

### Environment Variables

**`SAREK_DEBUG`** - Enable debug logging:

```bash
# Single component
SAREK_DEBUG=transfer ./my_program

# Multiple components
SAREK_DEBUG=transfer,kernel,device ./my_program

# All components
SAREK_DEBUG=all ./my_program
```

Components: `device`, `memory`, `transfer`, `kernel`, `execute`

### Common Debugging Patterns

**Check device capabilities:**
```ocaml
let dev = Device.get_default () in
let caps = dev.capabilities in
Printf.printf "Device: %s\n" dev.name;
Printf.printf "  Framework: %s\n" dev.framework;
Printf.printf "  Is CPU: %b\n" caps.is_cpu;
Printf.printf "  FP64: %b\n" caps.supports_fp64;
Printf.printf "  Atomics: %b\n" caps.supports_atomics;
Printf.printf "  Warp size: %d\n" caps.warp_size;
Printf.printf "  Max work group: %d\n" caps.max_work_group_size
```

**Track vector location:**
```ocaml
let print_location vec =
  match Vector.location vec with
  | CPU -> Printf.printf "CPU\n"
  | GPU d -> Printf.printf "GPU (device %d)\n" d.id
  | Both d -> Printf.printf "Both (device %d)\n" d.id
  | Stale_CPU d -> Printf.printf "Stale CPU (device %d has newer)\n" d.id
  | Stale_GPU d -> Printf.printf "Stale GPU (CPU has newer, device %d)\n" d.id
```

**Monitor transfers:**
```bash
SAREK_DEBUG=transfer ./my_program
```

Output:
```
[Transfer] host_ptr_to_device: ptr=139824567890432 size=4096
[Transfer] to_device: vector 0 → device 0 (CUDA)
[Transfer] Zero-copy enabled for device 1 (Native)
```

**Check buffer allocation:**
```bash
SAREK_DEBUG=memory ./my_program
```

## See Also

- [../sarek/README.md](../sarek/README.md) - Sarek runtime execution layer
- [../ppx/README.md](../ppx/README.md) - Sarek PPX compiler
- [../../spoc/framework/README.md](../../spoc/framework/README.md) - Plugin interface
- [../../spoc/ir/README.md](../../spoc/ir/README.md) - IR types and transformations
- [../../COVERAGE.md](../../COVERAGE.md) - Coverage infrastructure
- [../../README.md](../../README.md) - Main project documentation
