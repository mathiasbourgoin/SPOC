(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * CUDA API - High-Level Wrappers
 *
 * Provides a safe, OCaml-friendly interface to CUDA functionality.
 * Handles error checking, resource management, and type conversions.
 ******************************************************************************)

open Ctypes
open Cuda_types
open Cuda_bindings

(** {1 Constants} *)

(** Maximum device name length in characters *)
let max_device_name_length = 256

(** Maximum PTX header preview length for error messages *)
let max_ptx_header_preview = 200

(** {1 Exceptions} *)

(** Deprecated: no longer raised. [check] below now raises the canonical
    {!Cuda_error.Cuda_error} (a [Backend_error] alias) via the shared
    {!Sarek_backend_error.Backend_error.Make.check} funnel, so every handler in
    the codebase can catch one exception shape across backends. This declaration
    is kept only so that out-of-tree code matching on
    [Cuda_api.Cuda_error (code, ctx)] still compiles (this library is
    opam-published). *)
exception
  Cuda_error of cu_result * string
      [@ocaml.deprecated
        "no longer raised; Cuda_api.check now raises Cuda_error.Cuda_error \
         (Backend_error) - catch that instead"]

(** Check CUDA result and raise a canonical {!Backend_error} on failure. *)
let check (ctx : string) (result : cu_result) : unit =
  Cuda_error.check
    ~is_success:(fun r -> r = CUDA_SUCCESS)
    ~to_string:string_of_cu_result
    ctx
    result

(** One launch's argument blocks: [slots] holds the argument values, one 8-byte
    slot each, and [params] the array of pointers into it that [cuLaunchKernel]
    receives. Both are ctypes allocations owned by the GC, so they must stay
    reachable (see [retain_kernargs]) until the launch has been consumed. *)
type kernargs = {params : unit ptr ptr; slots : int64 ptr}

(** Host-side kernel-argument buffers (the array of argument pointers plus the
    value slots) for launches that may still be in flight, keyed by device id.

    [cuLaunchKernel] is asynchronous. The NVIDIA driver snapshots the kernel
    parameters synchronously before returning, so a caller may free them right
    after the call — but the ZLUDA/HIP stack reads them later, at dispatch time.
    Freeing them when [Kernel.launch] returns is therefore a use-after-free once
    the OCaml GC reclaims the blocks: the GPU then reads recycled host memory as
    the kernel's arguments (observed as an intermittent GPU page fault at a
    host-range address on ZLUDA).

    We retain the buffers here until the stream that ran the launch is drained.
    Each entry is keyed by [(device_id, stream_key)] where [stream_key] is the
    raw address of the stream handle (0 for the default/null stream, which is
    what SPOC uses). The buffers are held purely for liveness and are never
    inspected. A mutex guards the list so launches/retires from different OCaml
    domains cannot lose an entry (drop = use-after-free) or a removal. *)
let pending_kernargs : (int * nativeint * kernargs) list ref = ref []

let pending_lock = Mutex.create ()

let retain_kernargs device_id stream_key keepalive =
  Mutex.protect pending_lock (fun () ->
      pending_kernargs :=
        (device_id, stream_key, keepalive) :: !pending_kernargs)

(** Release buffers for one stream — use after draining exactly that stream (its
    synchronize, or a blocking op known to run on it). *)
let retire_stream device_id stream_key =
  Mutex.protect pending_lock (fun () ->
      pending_kernargs :=
        List.filter
          (fun (id, sk, _) -> id <> device_id || sk <> stream_key)
          !pending_kernargs)

(** Release all buffers for a device — use only after a full-context synchronize
    (drains every stream) or context destruction. *)
let retire_device device_id =
  Mutex.protect pending_lock (fun () ->
      pending_kernargs :=
        List.filter (fun (id, _, _) -> id <> device_id) !pending_kernargs)

(** TEST INSTRUMENTATION: number of launches whose argument blocks are still
    retained for [(device_id, stream_key)]. *)
let pending_kernarg_count device_id stream_key =
  Mutex.protect pending_lock (fun () ->
      List.length
        (List.filter
           (fun (id, sk, _) -> id = device_id && sk = stream_key)
           !pending_kernargs))

(** Raw address of a stream handle; the default (null) stream is 0. *)
let stream_key_of_ptr (handle : _ ptr) = raw_address_of_ptr (to_voidp handle)

(** Key of the default/null stream (what SPOC's blocking memcpys run on). *)
let default_stream_key = 0n

(** {1 Device Management} *)

module Device = struct
  type t = {
    id : int;
    handle : cu_device;
    context : cu_context structure ptr;
    name : string;
    total_mem : int64;
    compute_capability : int * int;
    max_threads_per_block : int;
    max_block_dims : int * int * int;
    max_grid_dims : int * int * int;
    shared_mem_per_block : int;
    warp_size : int;
    multiprocessor_count : int;
  }

  let initialized = ref false

  (* Device cache - reuse the same device/context to keep kernel handles valid *)
  let device_cache : (int, t) Hashtbl.t = Hashtbl.create 4

  let init () =
    if not !initialized then begin
      check "cuInit" (cuInit 0) ;
      initialized := true
    end

  let count () =
    init () ;
    let n = allocate int 0 in
    check "cuDeviceGetCount" (cuDeviceGetCount n) ;
    !@n

  let get_attribute dev attr =
    let v = allocate int 0 in
    check
      "cuDeviceGetAttribute"
      (cuDeviceGetAttribute v (int_of_device_attribute attr) dev) ;
    !@v

  (* Tracks the device whose context is current, so a null-stream
     [Kernel.launch] attributes retained kernargs to the right device. Kept in
     sync at every context switch: [set_current] and [create_device] (which
     leaves its new context current). All SPOC context switches go through
     these two, so the tracker never disagrees with the live CUDA context.

     Domain-local: a CUDA current context is per-OS-thread (cuCtxSetCurrent is
     thread-local), and OCaml 5 domains are threads, so each domain must track
     its own current device — a shared ref would let one domain's context
     switch mis-attribute another domain's launches. *)
  let current_id_key = Domain.DLS.new_key (fun () -> -1)

  let current_id () = Domain.DLS.get current_id_key

  let set_current_id id = Domain.DLS.set current_id_key id

  (* Create a new device with context - internal, use get for cached version *)
  let create_device idx =
    init () ;
    let dev = allocate cu_device 0 in
    check "cuDeviceGet" (cuDeviceGet dev idx) ;
    let handle = !@dev in

    (* Get name *)
    let name_buf = allocate_n char ~count:max_device_name_length in
    check
      "cuDeviceGetName"
      (cuDeviceGetName name_buf max_device_name_length handle) ;
    let name = string_from_ptr name_buf ~length:(max_device_name_length - 1) in
    let name =
      String.sub
        name
        0
        (try String.index name '\000'
         with Not_found -> max_device_name_length - 1)
    in

    (* Get total memory *)
    let mem = allocate size_t Unsigned.Size_t.zero in
    check "cuDeviceTotalMem" (cuDeviceTotalMem mem handle) ;
    let total_mem = Unsigned.Size_t.to_int64 !@mem in

    (* Get attributes *)
    let major =
      get_attribute handle CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    in
    let minor =
      get_attribute handle CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    in
    let max_threads =
      get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    in
    let max_block_x =
      get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
    in
    let max_block_y =
      get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
    in
    let max_block_z =
      get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
    in
    let max_grid_x = get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X in
    let max_grid_y = get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y in
    let max_grid_z = get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z in
    let shared_mem =
      get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    in
    let warp = get_attribute handle CU_DEVICE_ATTRIBUTE_WARP_SIZE in
    let mp_count =
      get_attribute handle CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
    in

    (* Create context *)
    let ctx = allocate cu_context_ptr (from_voidp cu_context null) in
    check
      "cuCtxCreate"
      (cuCtxCreate ctx (Unsigned.UInt.of_int cu_ctx_sched_auto) handle) ;

    let dev =
      {
        id = idx;
        handle;
        context = !@ctx;
        name;
        total_mem;
        compute_capability = (major, minor);
        max_threads_per_block = max_threads;
        max_block_dims = (max_block_x, max_block_y, max_block_z);
        max_grid_dims = (max_grid_x, max_grid_y, max_grid_z);
        shared_mem_per_block = shared_mem;
        warp_size = warp;
        multiprocessor_count = mp_count;
      }
    in
    Spoc_core.Log.debugf
      Spoc_core.Log.Device
      "CUDA device %d: %s (cc %d.%d, %Ld MB)"
      idx
      name
      major
      minor
      (Int64.div total_mem (Int64.of_int (1024 * 1024))) ;
    (* cuCtxCreate leaves the new context current. *)
    set_current_id idx ;
    dev

  (* Get a device, reusing cached context to keep kernel handles valid *)
  let get idx =
    match Hashtbl.find_opt device_cache idx with
    | Some dev -> dev
    | None ->
        let dev = create_device idx in
        Hashtbl.add device_cache idx dev ;
        dev

  let set_current dev =
    (* Tracker updated only once the driver accepted the switch, so the two
       never disagree. *)
    check "cuCtxSetCurrent" (cuCtxSetCurrent dev.context) ;
    set_current_id dev.id

  (* The id of the cached device whose context the driver reports as current
     for this thread (not the OCaml-side tracker); -1 when none matches. *)
  let driver_current () =
    let p = allocate cu_context_ptr (from_voidp cu_context null) in
    check "cuCtxGetCurrent" (cuCtxGetCurrent p) ;
    let cur = !@p in
    Hashtbl.fold
      (fun id dev found ->
        if ptr_compare dev.context cur = 0 then id else found)
      device_cache
      (-1)

  let synchronize dev =
    set_current dev ;
    check "cuCtxSynchronize" (cuCtxSynchronize ()) ;
    (* A full-context synchronize drains every stream on the device. *)
    retire_device dev.id

  let destroy dev =
    (* Deferred launches on this device must reach it before it is evicted:
       issuing them afterwards would re-create the device
       (specs/launch-window.md). *)
    Spoc_framework.Launch_hooks.flush_pending () ;
    (* Evict from device_cache first: leaving a stale entry means a later
       [get idx] returns a handle whose context has already been destroyed
       (mirrors Vulkan_api_device.destroy). *)
    Hashtbl.remove device_cache dev.id ;
    (* Notify BEFORE anything is released, while the context is still current:
       the listeners are the cache layers — this backend's own [Kernel.cache]
       (which unloads this device's modules) and, above it, memos closing over
       exactly those handles. Doing it here is what stops a stale
       module/function handle being returned by [Kernel.compile_cached] after
       the context is recreated under the same index.
       [notify_device_destroy] re-raises the first failing listener, and
       [device_cache] has already been emptied, so letting it escape here would
       leave the device unreachable with its context still alive and its modules
       still loaded. Capture it, finish the teardown, re-raise at the end. A
       failure in the teardown itself (cuCtxDestroy) legitimately wins over a
       memoization-drop failure. *)
    let listener_exn =
      match
        Spoc_framework.Cache_hooks.notify_device_destroy ~backend:"CUDA" dev.id
      with
      | () -> None
      | exception e -> Some e
    in
    retire_device dev.id ;
    check "cuCtxDestroy" (cuCtxDestroy dev.context) ;
    Option.iter raise listener_exn
end

(** {1 Memory Management} *)

module Memory = struct
  type 'a buffer = {
    ptr : cu_deviceptr;
    size : int;
    elem_size : int;
    device : Device.t;
  }

  let alloc device size kind =
    Device.set_current device ;
    (* NOT Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind): ctypes'
       kind GADT has no Float16 arm and raises Failure "Unsupported bigarray
       kind" there, so [Vector.create Vector.float16 n] died on this line with
       an opaque ctypes error (#57 slice 1 review, MF2). Spoc_core's pure table
       knows f16 is 2 bytes; see Spoc_core.Memory.bigarray_elem_size. *)
    let elem_size = Spoc_core.Memory.bigarray_elem_size kind in
    let bytes = Unsigned.Size_t.of_int (size * elem_size) in
    let ptr = allocate cu_deviceptr Unsigned.UInt64.zero in
    check "cuMemAlloc" (cuMemAlloc ptr bytes) ;
    {ptr = !@ptr; size; elem_size; device}

  (** Allocate buffer for custom types with explicit element size in bytes *)
  let alloc_custom device ~size ~elem_size =
    Device.set_current device ;
    let bytes = Unsigned.Size_t.of_int (size * elem_size) in
    let ptr = allocate cu_deviceptr Unsigned.UInt64.zero in
    check "cuMemAlloc (custom)" (cuMemAlloc ptr bytes) ;
    {ptr = !@ptr; size; elem_size; device}

  (* [free] runs from GC finalisers ([Gpu_memory.register_finalizer]), i.e. at
     an arbitrary point of whatever code it interrupts, possibly between a
     [Device.set_current] and the launch that relies on it for another device.
     Invariant: [free] leaves the selected context as it found it, so a
     finaliser leaves no trace on the code it interrupted. Same contract as
     [Hip_api.Memory.free]. *)
  let free buf =
    let prev = Device.current_id () in
    Fun.protect
      ~finally:(fun () ->
        if prev >= 0 && prev <> buf.device.Device.id then
          match Hashtbl.find_opt Device.device_cache prev with
          | None -> ()
          | Some prev_dev -> (
              (* Cannot raise from [finally] (and the caller is usually a
                 finaliser that swallows exceptions): say so instead. *)
              try Device.set_current prev_dev
              with e ->
                Spoc_core.Log.errorf
                  Spoc_core.Log.Device
                  "cuCtxSetCurrent %d (restoring the context selected before a \
                   free) failed: %s"
                  prev
                  (Printexc.to_string e)))
      (fun () ->
        Device.set_current buf.device ;
        check "cuMemFree" (cuMemFree buf.ptr))

  (* A blocking memcpy on the default stream drains all prior launches on that
     stream, so it is a safe point to release the device's retained kernargs. *)
  (* Both transfers acquire the host pointer through
     [Spoc_core.Memory.borrow_bigarray_ptr] rather than [Ctypes.bigarray_start]:
     the latter raises Failure "Unsupported bigarray kind" for Float16. The
     pointer does not root the bigarray; the [Sys.opaque_identity] after the
     memcpy keeps it reachable across the call. *)
  let host_to_device ~src ~dst =
    Device.set_current dst.device ;
    let src_ptr = Spoc_core.Memory.borrow_bigarray_ptr src in
    let bytes = Unsigned.Size_t.of_int (Bigarray.Array1.size_in_bytes src) in
    check "cuMemcpyHtoD" (cuMemcpyHtoD dst.ptr src_ptr bytes) ;
    ignore (Sys.opaque_identity src) ;
    retire_stream dst.device.id default_stream_key

  let device_to_host ~src ~dst =
    Device.set_current src.device ;
    let dst_ptr = Spoc_core.Memory.borrow_bigarray_ptr dst in
    let bytes = Unsigned.Size_t.of_int (Bigarray.Array1.size_in_bytes dst) in
    check "cuMemcpyDtoH" (cuMemcpyDtoH dst_ptr src.ptr bytes) ;
    ignore (Sys.opaque_identity dst) ;
    retire_stream src.device.id default_stream_key

  (** Transfer from raw pointer to device buffer (for custom types) *)
  let host_ptr_to_device ~src_ptr ~byte_size ~dst =
    Device.set_current dst.device ;
    let bytes = Unsigned.Size_t.of_int byte_size in
    check "cuMemcpyHtoD (ptr)" (cuMemcpyHtoD dst.ptr src_ptr bytes) ;
    retire_stream dst.device.id default_stream_key

  (** Transfer from device buffer to raw pointer (for custom types) *)
  let device_to_host_ptr ~src ~dst_ptr ~byte_size =
    Device.set_current src.device ;
    let bytes = Unsigned.Size_t.of_int byte_size in
    check "cuMemcpyDtoH (ptr)" (cuMemcpyDtoH dst_ptr src.ptr bytes) ;
    retire_stream src.device.id default_stream_key

  let device_to_device ~src ~dst =
    Device.set_current src.device ;
    let bytes = Unsigned.Size_t.of_int (src.size * src.elem_size) in
    check "cuMemcpyDtoD" (cuMemcpyDtoD dst.ptr src.ptr bytes) ;
    retire_stream src.device.id default_stream_key

  let memset buf value =
    Device.set_current buf.device ;
    let bytes = Unsigned.Size_t.of_int (buf.size * buf.elem_size) in
    check "cuMemsetD8" (cuMemsetD8 buf.ptr (Unsigned.UChar.of_int value) bytes)

  (** {2 Pinned (page-locked) host memory}

      Page-locked host buffers let the driver DMA straight to/from the device
      without staging through an internal pageable bounce buffer, roughly
      doubling H2D/D2H bandwidth on PCIe-class links, and are the hard
      prerequisite for true async transfers (a pageable [cuMemcpy*Async]
      silently degrades to synchronous). Two shapes are exposed:

      - {!alloc_host}/{!free_host} — driver-allocated page-locked memory
        ([cuMemAllocHost]); the returned raw pointer is fed to
        {!host_ptr_to_device}/{!device_to_host_ptr} exactly like any other host
        pointer.
      - {!register_host}/{!unregister_host} — page-lock an {e existing} host
        allocation in place ([cuMemHostRegister]); no allocation-path change,
        but the caller owns page-alignment and the register/unregister cost. *)

  type pinned_host = {host_ptr : unit ptr; bytes : int}

  (** Allocate [bytes] of page-locked host memory. Must be released with
      {!free_host}, never [Stdlib]/[free]. *)
  let alloc_host bytes =
    let pp = allocate (ptr void) null in
    check "cuMemAllocHost" (cuMemAllocHost pp (Unsigned.Size_t.of_int bytes)) ;
    {host_ptr = !@pp; bytes}

  let free_host ph = check "cuMemFreeHost" (cuMemFreeHost ph.host_ptr)

  (** Page-lock [bytes] at an existing host pointer (flags = 0: portable,
      non-mapped). Pair with {!unregister_host}. *)
  let register_host ptr bytes =
    check
      "cuMemHostRegister"
      (cuMemHostRegister
         (to_voidp ptr)
         (Unsigned.Size_t.of_int bytes)
         (Unsigned.UInt.of_int 0))

  let unregister_host ptr =
    check "cuMemHostUnregister" (cuMemHostUnregister (to_voidp ptr))
end

(** {1 Stream Management} *)

module Stream = struct
  type t = {handle : cu_stream structure ptr; device : Device.t}

  let create device =
    Device.set_current device ;
    let stream = allocate cu_stream_ptr (from_voidp cu_stream null) in
    check
      "cuStreamCreate"
      (cuStreamCreate stream (Unsigned.UInt.of_int cu_stream_default)) ;
    {handle = !@stream; device}

  let destroy stream =
    Device.set_current stream.device ;
    check "cuStreamDestroy" (cuStreamDestroy stream.handle)

  let synchronize stream =
    (* backlog-225 review: this was the only function in the module omitting
       Device.set_current before touching the stream/device. [Stream.default]
       is a NULL handle (the current context's null stream, not necessarily
       [stream.device]'s), so without this call, with device A current and a
       [Stream.default B] in hand, this would drain A and then retire B's
       kernarg keepalives under (B, 0) below — a use-after-free once the GC
       reclaims the argument CArray before B's kernel has actually launched.
       Filed as a real hazard (T1, code-reading only, not observed at
       runtime); see CHANGES.md and Cuda_plugin_base.ml:21 for the sibling
       finding. Not implicated in backlog-225 itself: both of that
       investigation's failing legs drain via [Device.synchronize], which
       already calls [set_current] correctly. *)
    Device.set_current stream.device ;
    check "cuStreamSynchronize" (cuStreamSynchronize stream.handle) ;
    (* Draining one stream retires only that stream's kernargs. *)
    retire_stream stream.device.id (stream_key_of_ptr stream.handle)

  let default device = {handle = from_voidp cu_stream null; device}
end

(** {1 Event Management} *)

module Event = struct
  type t = {handle : cu_event structure ptr}

  let create () =
    let event = allocate cu_event_ptr (from_voidp cu_event null) in
    check
      "cuEventCreate"
      (cuEventCreate event (Unsigned.UInt.of_int cu_event_default)) ;
    {handle = !@event}

  let destroy event = check "cuEventDestroy" (cuEventDestroy event.handle)

  let record event stream =
    check "cuEventRecord" (cuEventRecord event.handle stream.Stream.handle)

  let synchronize event =
    check "cuEventSynchronize" (cuEventSynchronize event.handle)

  let elapsed ~start ~stop =
    let ms = allocate float 0.0 in
    check "cuEventElapsedTime" (cuEventElapsedTime ms start.handle stop.handle) ;
    !@ms
end

(** {1 Ranged / asynchronous copies, pinned memory, stream ordering} *)

module Async = struct
  let offset_dptr (p : cu_deviceptr) off =
    Unsigned.UInt64.add p (Unsigned.UInt64.of_int off)

  let check_range what (buf : _ Memory.buffer) ~off ~byte_size =
    let total = buf.Memory.size * buf.Memory.elem_size in
    if off < 0 || byte_size < 0 || off + byte_size > total then
      invalid_arg
        (Printf.sprintf
           "Cuda_api.Async.%s: byte range [%d, %d) outside buffer of %d bytes"
           what
           off
           (off + byte_size)
           total)

  let check_stream_device what (s : Stream.t) (d : Device.t) =
    if s.Stream.device.Device.id <> d.Device.id then
      invalid_arg
        (Printf.sprintf
           "Cuda_api.Async.%s: stream belongs to CUDA device %d, buffer to %d"
           what
           s.Stream.device.Device.id
           d.Device.id)

  let host_to_device ~src_ptr ~(dst : _ Memory.buffer) ~dst_offset ~byte_size
      ~stream =
    check_range "host_to_device" dst ~off:dst_offset ~byte_size ;
    if byte_size > 0 then begin
      let d = offset_dptr dst.Memory.ptr dst_offset in
      let src = ptr_of_raw_address src_ptr in
      let n = Unsigned.Size_t.of_int byte_size in
      match stream with
      | None ->
          Device.set_current dst.Memory.device ;
          check "cuMemcpyHtoD (range)" (cuMemcpyHtoD d src n) ;
          retire_stream dst.Memory.device.Device.id default_stream_key
      | Some s ->
          check_stream_device "host_to_device" s dst.Memory.device ;
          Device.set_current s.Stream.device ;
          check "cuMemcpyHtoDAsync" (cuMemcpyHtoDAsync d src n s.Stream.handle)
    end

  let device_to_host ~(src : _ Memory.buffer) ~src_offset ~dst_ptr ~byte_size
      ~stream =
    check_range "device_to_host" src ~off:src_offset ~byte_size ;
    if byte_size > 0 then begin
      let sp = offset_dptr src.Memory.ptr src_offset in
      let dst = ptr_of_raw_address dst_ptr in
      let n = Unsigned.Size_t.of_int byte_size in
      match stream with
      | None ->
          Device.set_current src.Memory.device ;
          check "cuMemcpyDtoH (range)" (cuMemcpyDtoH dst sp n) ;
          retire_stream src.Memory.device.Device.id default_stream_key
      | Some s ->
          check_stream_device "device_to_host" s src.Memory.device ;
          Device.set_current s.Stream.device ;
          check "cuMemcpyDtoHAsync" (cuMemcpyDtoHAsync dst sp n s.Stream.handle)
    end

  let stream_wait_event (s : Stream.t) (e : Event.t) =
    Device.set_current s.Stream.device ;
    check
      "cuStreamWaitEvent"
      (cuStreamWaitEvent s.Stream.handle e.Event.handle Unsigned.UInt.zero)

  let event_query (e : Event.t) =
    match cuEventQuery e.Event.handle with
    | CUDA_SUCCESS -> true
    | CUDA_ERROR_NOT_READY -> false
    | r ->
        (* Any other code is an error: raise it rather than report "not
           ready", which a poller would wait on forever. *)
        check "cuEventQuery" r ;
        failwith "cuEventQuery: unexpected result accepted by check"

  let alloc_pinned (d : Device.t) ~byte_size =
    if byte_size <= 0 then
      invalid_arg "Cuda_api.Async.alloc_pinned: byte_size must be positive" ;
    Device.set_current d ;
    let pp = allocate (ptr void) null in
    (* CU_MEMHOSTALLOC_PORTABLE (0x1): pinned for every context, so a copy
       issued from another device's stream does not silently degrade to a
       staged copy. *)
    check
      "cuMemHostAlloc"
      (cuMemHostAlloc pp (Unsigned.Size_t.of_int byte_size) Unsigned.UInt.one) ;
    raw_address_of_ptr !@pp

  let free_pinned (d : Device.t) addr =
    Device.set_current d ;
    check "cuMemFreeHost" (cuMemFreeHost (ptr_of_raw_address addr))

  let register_host (d : Device.t) addr ~byte_size =
    Device.set_current d ;
    (* CU_MEMHOSTREGISTER_PORTABLE (0x1), for the same reason. *)
    check
      "cuMemHostRegister"
      (cuMemHostRegister
         (ptr_of_raw_address addr)
         (Unsigned.Size_t.of_int byte_size)
         Unsigned.UInt.one)

  let unregister_host (d : Device.t) addr =
    Device.set_current d ;
    Memory.unregister_host (ptr_of_raw_address addr)
end

(** {1 Kernel Management} *)

module Kernel = struct
  type t = {
    module_ : cu_module structure ptr;
    function_ : cu_function structure ptr;
    name : string;
  }

  type arg =
    | ArgBuffer : _ Memory.buffer -> arg
    | ArgInt32 : int32 -> arg
    | ArgInt64 : int64 -> arg
    | ArgFloat32 : float -> arg
    | ArgFloat64 : float -> arg
    | ArgPtr : nativeint -> arg

  (* Compilation cache. Guarded against concurrent multi-domain access by
     [Spoc_framework.Guarded_cache]: cache lookup/insert, per-device key
     tracking, eviction and clearing are all atomic, while the NVRTC compile
     runs outside the lock. Keys are grouped by device id (via
     [find_or_build ~device_id]) so a device destroy/recreate cycle can evict
     exactly its own stale module/function handles without reversing the
     (digested, opaque) cache key. *)
  let cache : (string, t) Spoc_framework.Guarded_cache.t =
    Spoc_framework.Guarded_cache.create
      ~destroy:(fun k -> ignore (cuModuleUnload k.module_))
      ()

  (* Evict every cached kernel compiled for [device_id]. Registered on the
     shared [Cache_hooks] registry — the one mechanism every backend uses (see
     Cache_hooks.mli) rather than a CUDA-private hook list — so that
     [Device.destroy], which fires the notification before it destroys
     anything, retires these module handles while the context is still alive.
     Match on the family name, never on the index alone: backend-local indices
     collide across backends, and [evict_device] does not merely drop
     memoization, it aborts in-flight builds for that index. *)
  let () =
    Spoc_framework.Cache_hooks.on_device_destroy (fun ~backend index ->
        if String.equal backend "CUDA" then
          Spoc_framework.Guarded_cache.evict_device cache index)

  (* Replace the .target directive in a PTX string to match the given SM
     version, and raise .version to the PTX ISA that SM needs (never lower it).
     This makes a static PTX string portable: PTX written for sm_86 loads fine
     on sm_61 as long as it doesn't use sm_86-specific instructions, and on
     sm_120 once the ISA is 8.7 (issue #506). *)
  let with_sm_target = Cuda_ptx_isa.with_sm_target

  (* Fail with both versions named when the driver cannot accept the PTX ISA
     version the (rewritten) PTX declares, instead of an opaque
     CUDA_ERROR_INVALID_PTX. Skipped on ZLUDA: it parses PTX without enforcing
     .version, and reports unlisted SMs (sm_88) that the table maps to 9.0. *)
  let check_driver_accepts ~device_name ~major ~minor ptx =
    let is_zluda =
      let n = String.length "ZLUDA" in
      let rec go i =
        i + n <= String.length device_name
        && (String.sub device_name i n = "ZLUDA" || go (i + 1))
      in
      go 0
    in
    match Cuda_ptx_isa.declared_version ptx with
    | Some declared when not is_zluda -> (
        let v = allocate int 0 in
        check "cuDriverGetVersion" (cuDriverGetVersion v) ;
        let ver = !@v in
        match
          Cuda_ptx_isa.check_driver
            ~driver:(ver / 1000, ver mod 1000 / 10)
            ~required:declared
        with
        | Ok () -> ()
        | Error msg ->
            Cuda_error.raise_error
              (Cuda_error.module_load_failed
                 (String.length ptx)
                 (Printf.sprintf "sm_%d%d: %s" major minor msg)))
    | _ -> ()

  (* Load a PTX string into a CUDA module and retrieve [name] as a function.
     The device context must already be current when this is called. *)
  let load_module_from_ptx ~name ptx =
    let module_ = allocate cu_module_ptr (from_voidp cu_module null) in
    let ptx_len = String.length ptx in
    let ptx_ba =
      Bigarray.Array1.create Bigarray.char Bigarray.c_layout (ptx_len + 1)
    in
    for i = 0 to ptx_len - 1 do
      Bigarray.Array1.set ptx_ba i ptx.[i]
    done ;
    Bigarray.Array1.set ptx_ba ptx_len '\000' ;
    let ptx_ptr = bigarray_start array1 ptx_ba |> to_voidp in
    let opt_arr =
      CArray.of_list int [int_of_jit_option CU_JIT_TARGET_FROM_CUCONTEXT]
    in
    let opt_vals = CArray.of_list (ptr void) [from_voidp void null] in
    let load_result =
      cuModuleLoadDataEx
        module_
        ptx_ptr
        (Unsigned.UInt.of_int (CArray.length opt_arr))
        (CArray.start opt_arr)
        (CArray.start opt_vals)
    in
    let load_result =
      match load_result with
      | CUDA_SUCCESS -> load_result
      | _ -> cuModuleLoadData module_ ptx_ptr
    in
    ignore (Sys.opaque_identity ptx_ba) ;
    (match load_result with
    | CUDA_SUCCESS ->
        Spoc_core.Log.debug Spoc_core.Log.Kernel "PTX module load succeeded"
    | err ->
        let ptx_header =
          String.sub ptx 0 (min max_ptx_header_preview (String.length ptx))
        in
        Spoc_core.Log.errorf
          Spoc_core.Log.Kernel
          "cuModuleLoadData failed: %s\nPTX header: %s"
          (string_of_cu_result err)
          ptx_header ;
        Cuda_error.raise_error
          (Cuda_error.module_load_failed
             (String.length ptx)
             (Printf.sprintf "cuModuleLoadData: %s" (string_of_cu_result err)))) ;
    let func = allocate cu_function_ptr (from_voidp cu_function null) in
    check "cuModuleGetFunction" (cuModuleGetFunction func !@module_ name) ;
    {module_ = !@module_; function_ = !@func; name}

  let compile device ~name ~source =
    Device.set_current device ;

    (* Compile to PTX - clamp architecture to what NVRTC likely supports.
       CUDA 13.x NVRTC supports up to compute_90. Newer devices will use
       the highest supported arch and rely on driver JIT. *)
    let major, minor = device.Device.compute_capability in
    let cc_num = (major * 10) + minor in
    let arch =
      if cc_num >= 90 then "compute_90"
        (* Clamp to compute_90 for Hopper and newer *)
      else Printf.sprintf "compute_%d%d" major minor
    in
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "CUDA compile: kernel='%s' arch=%s (cc %d.%d) device=%d"
      name
      arch
      major
      minor
      device.Device.id ;
    let ptx = Cuda_nvrtc.compile_to_ptx ~name ~arch source in
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "CUDA PTX generated (%d bytes)"
      (String.length ptx) ;
    load_module_from_ptx ~name ptx

  (** Load a pre-assembled PTX string directly, bypassing NVRTC. The .target
      directive in the PTX is automatically rewritten to match the device's
      actual SM, so a PTX built for sm_86 loads cleanly on sm_61 as long as it
      uses no sm_86-specific instructions. *)
  let load_from_ptx device ~name ~ptx =
    Device.set_current device ;
    let major, minor = device.Device.compute_capability in
    let ptx = with_sm_target ~major ~minor ptx in
    check_driver_accepts ~device_name:device.Device.name ~major ~minor ptx ;
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "PTX load_from_ptx: kernel='%s' sm_%d%d (%d bytes)"
      name
      major
      minor
      (String.length ptx) ;
    load_module_from_ptx ~name ptx

  (** Load a pre-assembled PTX string using the already-current CUDA context.
      The caller must have already set the device context via
      Device.set_current. *)
  let load_from_ptx_current ~name ~ptx =
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "PTX load_from_ptx_current: kernel='%s' (%d bytes)"
      name
      (String.length ptx) ;
    load_module_from_ptx ~name ptx

  (* Shared memoization for compiled/loaded kernels. The cache key must
     include device ID and the kernel name - a source file may define more
     than one kernel, and a resolved kernel handle for one name must never be
     returned for another (see Compile_cache.mli). Passing [~device_id] keeps
     the device-destroy eviction hook working for every entry. *)
  let with_cache device ~name ~source build =
    let key =
      Spoc_framework.Compile_cache.make_key
        ~device:(string_of_int device.Device.id)
        ~name
        ~source
        ()
    in
    Spoc_framework.Guarded_cache.find_or_build
      cache
      ~key
      ~device_id:device.Device.id
      build

  (** Cached variant of [load_from_ptx] — same cache as [compile_cached].
      Without it, every launch reloads (and re-JITs) the PTX module, which
      dominates kernel time on drivers that compile at module-load (NVIDIA JIT,
      ZLUDA). *)
  let load_from_ptx_cached device ~name ~ptx =
    with_cache device ~name ~source:ptx (fun () ->
        load_from_ptx device ~name ~ptx)

  let compile_cached device ~name ~source =
    with_cache device ~name ~source (fun () -> compile device ~name ~source)

  let clear_cache () =
    Spoc_framework.Cache_hooks.around_clear (fun () ->
        Spoc_framework.Guarded_cache.clear cache)

  (* TEST INSTRUMENTATION: counts the host blocks [marshal_args] allocates. *)
  let kernarg_blocks = Atomic.make 0

  let kernarg_blocks_allocated () = Atomic.get kernarg_blocks

  let alloc_block typ ~count =
    Atomic.incr kernarg_blocks ;
    allocate_n typ ~count

  (* Argument marshalling (the HIP twin is [Hip_api.Kernel.launch]; keep the
     two layouts identical): ONE 8-byte-slot block holding every argument value
     and ONE array of pointers into it, i.e. two host allocations per launch
     whatever the argument count (a ctypes [allocate] per argument is a calloc
     plus a GC finaliser each).

     Every value is written with its OWN C type at the start of its slot, so a
     4-byte argument occupies the low-addressed 4 bytes of its slot and the
     driver, which reads each parameter at the width the compiled kernel
     declares, sees the same bytes as with one cell per argument - no
     endianness assumption. Slots are 8 bytes and the block is [int64_t]-aligned,
     which covers every argument kind (pointers, int64, double). A zero-argument
     launch still allocates one slot so the pointer handed to the driver is
     never dangling. Both blocks are retained by [launch] until the stream is
     retired. *)
  let marshal_args args : kernargs =
    let n = List.length args in
    let slots = alloc_block int64_t ~count:(max 1 n) in
    let params = alloc_block (ptr void) ~count:(max 1 n) in
    List.iteri
      (fun i arg ->
        let slot = to_voidp (slots +@ i) in
        (match arg with
        | ArgBuffer buf -> from_voidp cu_deviceptr slot <-@ buf.Memory.ptr
        | ArgInt32 v -> from_voidp int32_t slot <-@ v
        | ArgInt64 v -> from_voidp int64_t slot <-@ v
        | ArgFloat32 v -> from_voidp float slot <-@ v
        | ArgFloat64 v -> from_voidp double slot <-@ v
        | ArgPtr v -> from_voidp nativeint slot <-@ v) ;
        params +@ i <-@ slot)
      args ;
    {params; slots}

  (* TEST INSTRUMENTATION: the pointer handed to the driver for argument [i]. *)
  let kernarg_pointer (k : kernargs) i = !@(k.params +@ i)

  let launch kernel ~args ~grid ~block ~shared_mem ~stream =
    let ({params; _} as kernargs) = marshal_args args in

    let stream_ptr =
      match stream with
      | Some s ->
          (* The module was loaded for the CURRENT device; enqueueing it on a
             stream of another device is refused here rather than left to an
             opaque driver error (reachable through run_source_on_stream). *)
          let cur = Device.current_id () in
          if cur >= 0 && s.Stream.device.Device.id <> cur then
            invalid_arg
              (Printf.sprintf
                 "Kernel.launch: stream of device %d, current device %d"
                 s.Stream.device.Device.id
                 cur) ;
          s.Stream.handle
      | None -> from_voidp cu_stream null
    in

    let gx, gy, gz = grid in
    let bx, by, bz = block in

    check
      "cuLaunchKernel"
      (cuLaunchKernel
         kernel.function_
         (Unsigned.UInt.of_int gx)
         (Unsigned.UInt.of_int gy)
         (Unsigned.UInt.of_int gz)
         (Unsigned.UInt.of_int bx)
         (Unsigned.UInt.of_int by)
         (Unsigned.UInt.of_int bz)
         (Unsigned.UInt.of_int shared_mem)
         stream_ptr
         params
         (from_voidp (ptr void) null)) ;
    (* The launch is asynchronous and the argument blocks (params + slots) may
       be read after this returns. Keep them alive until the stream that ran
       the launch is drained. Attribute to that stream (and its device); a
       null/default stream keys as [default_stream_key] on the current
       context's device. *)
    let device_id, stream_key =
      match stream with
      | Some s -> (s.Stream.device.id, stream_key_of_ptr s.Stream.handle)
      | None -> (Device.current_id (), default_stream_key)
    in
    retain_kernargs device_id stream_key kernargs
end

(** {1 Utility Functions} *)

let driver_version () =
  let v = allocate int 0 in
  check "cuDriverGetVersion" (cuDriverGetVersion v) ;
  let ver = !@v in
  (ver / 1000, ver mod 1000 / 10)

(** Driver-only availability: libcuda is loadable and reports at least one
    device. Sufficient for the PTX backend, which never calls NVRTC — this is
    what makes SPOC work on non-NVIDIA CUDA implementations such as ZLUDA, which
    ship the driver API without libnvrtc. *)
let is_driver_available () =
  if not (Cuda_bindings.is_available ()) then false
  else
    try
      Device.init () ;
      Device.count () > 0
    with _ -> false

(* NVRTC is probed before the driver so that a host without libnvrtc (e.g.
   ZLUDA) returns false without cuInit-ing the driver as a side effect. *)
let is_available () =
  Cuda_bindings.is_available ()
  && Cuda_nvrtc.is_available () && is_driver_available ()

let memory_info device =
  Device.set_current device ;
  let free = allocate size_t Unsigned.Size_t.zero in
  let total = allocate size_t Unsigned.Size_t.zero in
  check "cuMemGetInfo" (cuMemGetInfo free total) ;
  (Unsigned.Size_t.to_int64 !@free, Unsigned.Size_t.to_int64 !@total)
