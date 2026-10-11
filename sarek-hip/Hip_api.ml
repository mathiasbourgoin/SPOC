(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * HIP API - High-Level Wrappers
 *
 * Safe, OCaml-friendly interface to HIP. HIP analog of Cuda_api. Uses HIP's
 * runtime device model (hipSetDevice) instead of the deprecated hipCtx* API;
 * module load/launch go through the driver-style hipModule* calls. Kernels are
 * JIT-compiled with hiprtc to a finalized code object and loaded via
 * hipModuleLoadData (one fewer stage than the CUDA NVRTC->ptxas->module path).
 ******************************************************************************)

open Ctypes
open Hip_types
open Hip_bindings

(** {1 Constants} *)

let max_device_name_length = 256

(** {1 Error checking} *)

let check (ctx : string) (result : hip_result) : unit =
  Hip_error.check
    ~is_success:(fun r -> r = HIP_SUCCESS)
    ~to_string:string_of_hip_result
    ctx
    result

(** hipModuleLaunchKernel is asynchronous and the HIP stack snapshots the kernel
    parameters at dispatch time, not at the call site (see the CUDA sibling's
    note - it explicitly calls out the HIP behaviour). Freeing the arg cells
    when [Kernel.launch] returns would be a use-after-free once the GC reclaims
    them. We retain them until the stream that ran the launch is drained, keyed
    by [(device_id, stream_key)]. The cells are held for liveness only and never
    inspected. A mutex guards the list against concurrent domains. *)
type kernargs = {params : unit ptr ptr; slots : int64 ptr}

let pending_kernargs : (int * nativeint * kernargs) list ref = ref []

let pending_lock = Mutex.create ()

let retain_kernargs device_id stream_key keepalive =
  Mutex.protect pending_lock (fun () ->
      pending_kernargs :=
        (device_id, stream_key, keepalive) :: !pending_kernargs)

let retire_stream device_id stream_key =
  Mutex.protect pending_lock (fun () ->
      pending_kernargs :=
        List.filter
          (fun (id, sk, _) -> id <> device_id || sk <> stream_key)
          !pending_kernargs)

let retire_device device_id =
  Mutex.protect pending_lock (fun () ->
      pending_kernargs :=
        List.filter (fun (id, _, _) -> id <> device_id) !pending_kernargs)

let stream_key_of_ptr (handle : _ ptr) = raw_address_of_ptr (to_voidp handle)

let default_stream_key = 0n

(** {1 Device Management} *)

module Device = struct
  type t = {
    id : int;
    handle : hip_device;
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

  let device_cache : (int, t) Hashtbl.t = Hashtbl.create 4

  let init () =
    if not !initialized then begin
      check "hipInit" (hipInit (Unsigned.UInt.of_int 0)) ;
      initialized := true
    end

  let count () =
    init () ;
    let n = allocate int 0 in
    check "hipGetDeviceCount" (hipGetDeviceCount n) ;
    !@n

  let get_attribute dev attr =
    let v = allocate int 0 in
    check
      "hipDeviceGetAttribute"
      (hipDeviceGetAttribute v (int_of_device_attribute attr) dev) ;
    !@v

  (* Domain-local current-device tracker: hipSetDevice is per-OS-thread and
     OCaml 5 domains are threads, so each domain tracks its own current device
     (mirrors the CUDA backend). *)
  let current_id_key = Domain.DLS.new_key (fun () -> -1)

  let current_id () = Domain.DLS.get current_id_key

  let set_current_id id = Domain.DLS.set current_id_key id

  let create_device idx =
    init () ;
    let dev = allocate hip_device 0 in
    check "hipDeviceGet" (hipDeviceGet dev idx) ;
    let handle = !@dev in

    let name_buf = allocate_n char ~count:max_device_name_length in
    check
      "hipDeviceGetName"
      (hipDeviceGetName name_buf max_device_name_length handle) ;
    let name = string_from_ptr name_buf ~length:(max_device_name_length - 1) in
    let name =
      String.sub
        name
        0
        (try String.index name '\000'
         with Not_found -> max_device_name_length - 1)
    in

    let mem = allocate size_t Unsigned.Size_t.zero in
    check "hipDeviceTotalMem" (hipDeviceTotalMem mem handle) ;
    let total_mem = Unsigned.Size_t.to_int64 !@mem in

    let major = allocate int 0 and minor = allocate int 0 in
    check
      "hipDeviceComputeCapability"
      (hipDeviceComputeCapability major minor handle) ;
    let major = !@major and minor = !@minor in

    let max_threads =
      get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    in
    let max_block_x =
      get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
    in
    let max_block_y =
      get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
    in
    let max_block_z =
      get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
    in
    let max_grid_x = get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X in
    let max_grid_y = get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y in
    let max_grid_z = get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z in
    let shared_mem =
      get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    in
    let warp = get_attribute handle HIP_DEVICE_ATTRIBUTE_WARP_SIZE in
    let mp_count =
      get_attribute handle HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
    in

    let dev =
      {
        id = idx;
        handle;
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
      "HIP device %d: %s (cc %d.%d, %Ld MB, warp %d)"
      idx
      name
      major
      minor
      (Int64.div total_mem (Int64.of_int (1024 * 1024)))
      warp ;
    dev

  let get idx =
    match Hashtbl.find_opt device_cache idx with
    | Some dev -> dev
    | None ->
        let dev = create_device idx in
        Hashtbl.add device_cache idx dev ;
        dev

  let set_current dev =
    check "hipSetDevice" (hipSetDevice dev.id) ;
    set_current_id dev.id

  (* The device the HIP runtime has selected for this thread, as the driver
     reports it (not the OCaml-side tracker). *)
  let driver_current () =
    let d = allocate int (-1) in
    check "hipGetDevice" (hipGetDevice d) ;
    !@d

  let synchronize dev =
    set_current dev ;
    check "hipDeviceSynchronize" (hipDeviceSynchronize ()) ;
    retire_device dev.id

  let destroy dev =
    (* Deferred launches on this device must reach it before it is evicted:
       issuing them afterwards would re-create the device (specs/launch-window.md). *)
    Spoc_framework.Launch_hooks.flush_pending () ;
    (* Evict from device_cache, then notify while the device is still
       selected, mirroring the CUDA backend. HIP has no explicit context to
       destroy under the runtime model, so there is no hipCtxDestroy analog. *)
    Hashtbl.remove device_cache dev.id ;
    (* Notify BEFORE anything is released: the listeners are the cache layers —
       this backend's own [Kernel.cache] (which unloads this device's modules)
       and, above it, memos closing over exactly those handles.
       [notify_device_destroy] re-raises the first failing listener; since
       [device_cache] has already been emptied, letting it escape here would
       leave the device unreachable with its modules still loaded. Capture it,
       finish the teardown, re-raise at the end (mirrors the CUDA backend). *)
    let listener_exn =
      match
        Spoc_framework.Cache_hooks.notify_device_destroy ~backend:"HIP" dev.id
      with
      | () -> None
      | exception e -> Some e
    in
    retire_device dev.id ;
    Option.iter raise listener_exn
end

(** {1 Memory Management} *)

module Memory = struct
  type 'a buffer = {
    ptr : hip_deviceptr;
    size : int;
    elem_size : int;
    device : Device.t;
  }

  let alloc device size kind =
    Device.set_current device ;
    (* NOT Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind): ctypes has
       no Float16 kind and raises "Unsupported bigarray kind" there. Spoc_core's
       pure table knows f16 is 2 bytes. See Spoc_core.Memory.borrow_bigarray_ptr
       for the full explanation. *)
    let elem_size = Spoc_core.Memory.bigarray_elem_size kind in
    let bytes = Unsigned.Size_t.of_int (size * elem_size) in
    let pp = allocate (ptr void) null in
    check "hipMalloc" (hipMalloc pp bytes) ;
    {ptr = !@pp; size; elem_size; device}

  let alloc_custom device ~size ~elem_size =
    Device.set_current device ;
    let bytes = Unsigned.Size_t.of_int (size * elem_size) in
    let pp = allocate (ptr void) null in
    check "hipMalloc (custom)" (hipMalloc pp bytes) ;
    {ptr = !@pp; size; elem_size; device}

  (* [free] is called from GC finalisers ([Gpu_memory.register_finalizer]),
     i.e. at an arbitrary allocation point of whatever code is running — in
     particular between [Execute]'s [set_current] and [hipModuleLaunchKernel]
     of a launch on ANOTHER device. Selecting the buffer's device and leaving
     it selected then made that launch run a module of device B with device A
     current: hipErrorInvalidDevice. Observed with two HIP devices (RX 7900 XTX
     + the 7950X iGPU) once launches allocated less and the collector's timing
     moved (cheap-repeated-launches). So the previously selected device is
     restored: a finaliser must leave no trace on the code it interrupted. *)
  let free buf =
    let prev = Device.current_id () in
    Fun.protect
      ~finally:(fun () ->
        if prev >= 0 && prev <> buf.device.Device.id then begin
          match hipSetDevice prev with
          | HIP_SUCCESS -> Device.set_current_id prev
          | err ->
              (* Cannot raise from [finally] (and the caller is usually a
                 finaliser that swallows exceptions): say so instead. *)
              Spoc_core.Log.errorf
                Spoc_core.Log.Device
                "hipSetDevice %d (restoring the device selected before a free) \
                 failed: %s"
                prev
                (string_of_hip_result err)
        end)
      (fun () ->
        Device.set_current buf.device ;
        check "hipFree" (hipFree buf.ptr))

  (* The [Sys.opaque_identity] here is the same keep-alive the sibling
     [load_module_from_code_object] below applies to its code bigarray: the
     bigarray must stay reachable until hipMemcpy has consumed its address.
     [Spoc_core.Memory.borrow_bigarray_ptr] does not root it (its f16 pointer
     is unmanaged), so this keep-alive is required on every kind. *)
  let host_to_device ~src ~dst =
    Device.set_current dst.device ;
    let src_ptr = Spoc_core.Memory.borrow_bigarray_ptr src in
    let bytes = Unsigned.Size_t.of_int (Bigarray.Array1.size_in_bytes src) in
    check "hipMemcpyHtoD" (hipMemcpyHtoD dst.ptr src_ptr bytes) ;
    ignore (Sys.opaque_identity src) ;
    retire_stream dst.device.id default_stream_key

  let device_to_host ~src ~dst =
    Device.set_current src.device ;
    let dst_ptr = Spoc_core.Memory.borrow_bigarray_ptr dst in
    let bytes = Unsigned.Size_t.of_int (Bigarray.Array1.size_in_bytes dst) in
    check "hipMemcpyDtoH" (hipMemcpyDtoH dst_ptr src.ptr bytes) ;
    ignore (Sys.opaque_identity dst) ;
    retire_stream src.device.id default_stream_key

  let host_ptr_to_device ~src_ptr ~byte_size ~dst =
    Device.set_current dst.device ;
    let bytes = Unsigned.Size_t.of_int byte_size in
    check "hipMemcpyHtoD (ptr)" (hipMemcpyHtoD dst.ptr src_ptr bytes) ;
    retire_stream dst.device.id default_stream_key

  let device_to_host_ptr ~src ~dst_ptr ~byte_size =
    Device.set_current src.device ;
    let bytes = Unsigned.Size_t.of_int byte_size in
    check "hipMemcpyDtoH (ptr)" (hipMemcpyDtoH dst_ptr src.ptr bytes) ;
    retire_stream src.device.id default_stream_key

  let device_to_device ~src ~dst =
    Device.set_current src.device ;
    let bytes = Unsigned.Size_t.of_int (src.size * src.elem_size) in
    check "hipMemcpyDtoD" (hipMemcpyDtoD dst.ptr src.ptr bytes) ;
    retire_stream src.device.id default_stream_key

  let memset buf value =
    Device.set_current buf.device ;
    let bytes = Unsigned.Size_t.of_int (buf.size * buf.elem_size) in
    check
      "hipMemsetD8"
      (hipMemsetD8 buf.ptr (Unsigned.UChar.of_int value) bytes)
end

(** {1 Stream Management} *)

module Stream = struct
  type t = {handle : hip_stream structure ptr; device : Device.t}

  let create device =
    Device.set_current device ;
    let stream = allocate hip_stream_ptr (from_voidp hip_stream null) in
    check "hipStreamCreate" (hipStreamCreate stream) ;
    {handle = !@stream; device}

  let destroy stream =
    Device.set_current stream.device ;
    check "hipStreamDestroy" (hipStreamDestroy stream.handle)

  let synchronize stream =
    (* Select the stream's device first, as Cuda_api.Stream.synchronize does
       since backlog-225: [default] is the NULL handle, i.e. the CURRENT
       device's null stream, so without this a [default d2] synchronised while
       d1 is current drains d1 and then retires d2's keepalives. *)
    Device.set_current stream.device ;
    check "hipStreamSynchronize" (hipStreamSynchronize stream.handle) ;
    retire_stream stream.device.id (stream_key_of_ptr stream.handle)

  let default device = {handle = from_voidp hip_stream null; device}
end

(** {1 Event Management} *)

module Event = struct
  type t = {handle : hip_event structure ptr}

  let create () =
    let event = allocate hip_event_ptr (from_voidp hip_event null) in
    check "hipEventCreate" (hipEventCreate event) ;
    {handle = !@event}

  let destroy event = check "hipEventDestroy" (hipEventDestroy event.handle)

  let record event stream =
    check "hipEventRecord" (hipEventRecord event.handle stream.Stream.handle)

  let synchronize event =
    check "hipEventSynchronize" (hipEventSynchronize event.handle)

  let elapsed ~start ~stop =
    let ms = allocate float 0.0 in
    check
      "hipEventElapsedTime"
      (hipEventElapsedTime ms start.handle stop.handle) ;
    !@ms
end

(** {1 Ranged / asynchronous copies, pinned memory, stream ordering} *)

module Async = struct
  let addr_of p = raw_address_of_ptr p

  let offset_ptr (p : unit ptr) off =
    ptr_of_raw_address (Nativeint.add (addr_of p) (Nativeint.of_int off))

  let check_range what (buf : _ Memory.buffer) ~off ~byte_size =
    let total = buf.Memory.size * buf.Memory.elem_size in
    if off < 0 || byte_size < 0 || off + byte_size > total then
      invalid_arg
        (Printf.sprintf
           "Hip_api.Async.%s: byte range [%d, %d) outside buffer of %d bytes"
           what
           off
           (off + byte_size)
           total)

  let check_stream_device what (s : Stream.t) (d : Device.t) =
    if s.Stream.device.Device.id <> d.Device.id then
      invalid_arg
        (Printf.sprintf
           "Hip_api.Async.%s: stream belongs to HIP device %d, buffer to %d"
           what
           s.Stream.device.Device.id
           d.Device.id)

  let host_to_device ~src_ptr ~(dst : _ Memory.buffer) ~dst_offset ~byte_size
      ~stream =
    check_range "host_to_device" dst ~off:dst_offset ~byte_size ;
    if byte_size > 0 then begin
      let d = offset_ptr dst.Memory.ptr dst_offset in
      let src = ptr_of_raw_address src_ptr in
      let n = Unsigned.Size_t.of_int byte_size in
      match stream with
      | None ->
          Device.set_current dst.Memory.device ;
          check "hipMemcpyHtoD (range)" (hipMemcpyHtoD d src n) ;
          retire_stream dst.Memory.device.Device.id default_stream_key
      | Some s ->
          check_stream_device "host_to_device" s dst.Memory.device ;
          Device.set_current s.Stream.device ;
          check
            "hipMemcpyHtoDAsync"
            (hipMemcpyHtoDAsync d src n s.Stream.handle)
    end

  let device_to_host ~(src : _ Memory.buffer) ~src_offset ~dst_ptr ~byte_size
      ~stream =
    check_range "device_to_host" src ~off:src_offset ~byte_size ;
    if byte_size > 0 then begin
      let sp = offset_ptr src.Memory.ptr src_offset in
      let dst = ptr_of_raw_address dst_ptr in
      let n = Unsigned.Size_t.of_int byte_size in
      match stream with
      | None ->
          Device.set_current src.Memory.device ;
          check "hipMemcpyDtoH (range)" (hipMemcpyDtoH dst sp n) ;
          retire_stream src.Memory.device.Device.id default_stream_key
      | Some s ->
          check_stream_device "device_to_host" s src.Memory.device ;
          Device.set_current s.Stream.device ;
          check
            "hipMemcpyDtoHAsync"
            (hipMemcpyDtoHAsync dst sp n s.Stream.handle)
    end

  let stream_wait_event (s : Stream.t) (e : Event.t) =
    Device.set_current s.Stream.device ;
    check
      "hipStreamWaitEvent"
      (hipStreamWaitEvent s.Stream.handle e.Event.handle Unsigned.UInt.zero)

  let event_query (e : Event.t) =
    match hipEventQuery e.Event.handle with
    | HIP_SUCCESS -> true
    | HIP_ERROR_NOT_READY -> false
    | r ->
        (* Any other code is an error: raise it rather than report "not
           ready", which a poller would wait on forever. *)
        check "hipEventQuery" r ;
        failwith ("hipEventQuery: unexpected " ^ string_of_hip_result r)

  let alloc_pinned (d : Device.t) ~byte_size =
    if byte_size <= 0 then
      invalid_arg "Hip_api.Async.alloc_pinned: byte_size must be positive" ;
    Device.set_current d ;
    let pp = allocate (ptr void) null in
    (* hipHostMallocPortable (0x1): page-locked for every device, not only
       the current one, so a copy issued on another device's stream does not
       silently fall back to a staged copy. *)
    check
      "hipHostMalloc"
      (hipHostMalloc pp (Unsigned.Size_t.of_int byte_size) Unsigned.UInt.one) ;
    addr_of !@pp

  let free_pinned (d : Device.t) addr =
    Device.set_current d ;
    check "hipHostFree" (hipHostFree (ptr_of_raw_address addr))

  let register_host (d : Device.t) addr ~byte_size =
    Device.set_current d ;
    check
      "hipHostRegister"
      (* hipHostRegisterPortable (0x1), for the same reason as alloc_pinned. *)
      (hipHostRegister
         (ptr_of_raw_address addr)
         (Unsigned.Size_t.of_int byte_size)
         Unsigned.UInt.one)

  let unregister_host (d : Device.t) addr =
    Device.set_current d ;
    check "hipHostUnregister" (hipHostUnregister (ptr_of_raw_address addr))
end

(** {1 Kernel Management} *)

module Kernel = struct
  type t = {
    module_ : hip_module structure ptr;
    function_ : hip_function structure ptr;
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
     tracking, eviction and clearing are all atomic, while the hiprtc compile
     runs outside the lock. Keys are grouped by device id (via
     [find_or_build ~device_id]) so a device destroy/recreate cycle can evict
     exactly its own stale module/function handles. *)
  let cache : (string, t) Spoc_framework.Guarded_cache.t =
    Spoc_framework.Guarded_cache.create
      ~destroy:(fun k -> ignore (hipModuleUnload k.module_))
      ()

  (* Evict every cached kernel compiled for [device_id]. Registered on the
     shared [Cache_hooks] registry — the one mechanism every backend uses (see
     Cache_hooks.mli) rather than a HIP-private hook list. [Device.destroy]
     fires the notification before it retires anything. Match on the family
     name, never on the index alone: backend-local indices collide across
     backends. *)
  let () =
    Spoc_framework.Cache_hooks.on_device_destroy (fun ~backend index ->
        if String.equal backend "HIP" then
          Spoc_framework.Guarded_cache.evict_device cache index)

  (* Load a finalized HIP code object (binary ELF bytes) into a module and
     resolve [name]. The device must already be selected. The code object is
     self-describing (its size is in the ELF header), so hipModuleLoadData
     needs only the base pointer. *)
  let load_module_from_code_object ~name code =
    let module_ = allocate hip_module_ptr (from_voidp hip_module null) in
    let len = String.length code in
    let code_ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout len in
    for i = 0 to len - 1 do
      Bigarray.Array1.set code_ba i code.[i]
    done ;
    let code_ptr = bigarray_start array1 code_ba |> to_voidp in
    let load_result = hipModuleLoadData module_ code_ptr in
    ignore (Sys.opaque_identity code_ba) ;
    (match load_result with
    | HIP_SUCCESS ->
        Spoc_core.Log.debug
          Spoc_core.Log.Kernel
          "HIP code object load succeeded"
    | err ->
        Spoc_core.Log.errorf
          Spoc_core.Log.Kernel
          "hipModuleLoadData failed: %s"
          (string_of_hip_result err) ;
        Hip_error.raise_error
          (Hip_error.module_load_failed
             len
             (Printf.sprintf "hipModuleLoadData: %s" (string_of_hip_result err)))) ;
    let func = allocate hip_function_ptr (from_voidp hip_function null) in
    check "hipModuleGetFunction" (hipModuleGetFunction func !@module_ name) ;
    {module_ = !@module_; function_ = !@func; name}

  let compile device ~name ~source =
    Device.set_current device ;
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "HIP compile: kernel='%s' device=%d (current-device arch)"
      name
      device.Device.id ;
    (* No explicit --offload-arch: hiprtc targets the just-selected device,
       which is correct for every gfx target incl. the integrated gfx1036. *)
    let code = Hip_rtc.compile_to_code_object ~name source in
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "HIP code object generated (%d bytes)"
      (String.length code) ;
    load_module_from_code_object ~name code

  (* Like [compile] but threads extra hiprtc options (e.g. include dirs such as
     "-I/opt/rocm/include" for rocWMMA headers, or preprocessor defines).
     Targets the current device (no explicit --offload-arch). *)
  let compile_with_options device ~name ~source ~options =
    Device.set_current device ;
    let code = Hip_rtc.compile_to_code_object ~name ~options source in
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "HIP code object generated (%d bytes, %d extra opts)"
      (String.length code)
      (List.length options) ;
    load_module_from_code_object ~name code

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

  let compile_cached device ~name ~source =
    with_cache device ~name ~source (fun () -> compile device ~name ~source)

  let clear_cache () =
    Spoc_framework.Cache_hooks.around_clear (fun () ->
        Spoc_framework.Guarded_cache.clear cache)

  (* Argument marshalling: ONE 8-byte-slot block holding every argument value
     and ONE array of pointers into it, i.e. two host allocations per launch
     whatever the argument count. The previous shape did one ctypes [allocate]
     per argument; each is a calloc plus a GC finaliser, and on a 10-argument
     kernel that was ~3 us of a ~6 us launch (RX 7900 XTX, measured).

     Every value is written with its OWN C type at the start of its slot, so a
     4-byte argument occupies the low-addressed 4 bytes of its slot and the
     driver, which reads each parameter at the width the compiled kernel
     declares, sees exactly the bytes it saw before — no endianness assumption.
     Slots are 8 bytes and the block is [int64_t]-aligned, which covers every
     argument kind (pointers, int64, double). A zero-argument launch still
     allocates one slot so the pointer handed to the driver is never dangling.
     Both blocks are retained below exactly as the per-argument values were. *)
  let launch kernel ~args ~grid ~block ~shared_mem ~stream =
    let n = List.length args in
    let slots = allocate_n int64_t ~count:(max 1 n) in
    let params = allocate_n (ptr void) ~count:(max 1 n) in
    List.iteri
      (fun i arg ->
        let slot = to_voidp (slots +@ i) in
        (match arg with
        | ArgBuffer buf -> from_voidp (ptr void) slot <-@ buf.Memory.ptr
        | ArgInt32 v -> from_voidp int32_t slot <-@ v
        | ArgInt64 v -> from_voidp int64_t slot <-@ v
        | ArgFloat32 v -> from_voidp float slot <-@ v
        | ArgFloat64 v -> from_voidp double slot <-@ v
        | ArgPtr v -> from_voidp nativeint slot <-@ v) ;
        params +@ i <-@ slot)
      args ;

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
      | None -> from_voidp hip_stream null
    in

    let gx, gy, gz = grid in
    let bx, by, bz = block in

    check
      "hipModuleLaunchKernel"
      (hipModuleLaunchKernel
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
    let device_id, stream_key =
      match stream with
      | Some s -> (s.Stream.device.id, stream_key_of_ptr s.Stream.handle)
      | None -> (Device.current_id (), default_stream_key)
    in
    retain_kernargs device_id stream_key {params; slots}
end

(** {1 Utility Functions} *)

let runtime_version () =
  let v = allocate int 0 in
  check "hipRuntimeGetVersion" (hipRuntimeGetVersion v) ;
  !@v

(** Available if libamdhip64 loads, libhiprtc loads, and at least one device is
    present. hiprtc is required because this backend is JIT-only (HIP C++ ->
    code object at runtime). *)
let is_available () =
  if not (Hip_bindings.is_available () && Hip_rtc.is_available ()) then false
  else
    try
      Device.init () ;
      Device.count () > 0
    with _ -> false

let memory_info device =
  Device.set_current device ;
  let free = allocate size_t Unsigned.Size_t.zero in
  let total = allocate size_t Unsigned.Size_t.zero in
  check "hipMemGetInfo" (hipMemGetInfo free total) ;
  (Unsigned.Size_t.to_int64 !@free, Unsigned.Size_t.to_int64 !@total)
