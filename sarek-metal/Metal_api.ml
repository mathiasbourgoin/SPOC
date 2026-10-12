(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Metal API - High-Level Wrappers
 *
 * Provides a safe, OCaml-friendly interface to Metal functionality.
 * Handles error checking, resource management, and type conversions.
 ******************************************************************************)

open Ctypes
open Foreign
open Metal_types
open Metal_bindings

(** memcpy from libc for memory transfers, with the OCaml runtime lock released
    (a large copy would otherwise stop every other domain's minor collections).
    Both arguments are raw addresses, which the call does not root: the caller
    keeps the memory behind them allocated until [memcpy] returns (bigarrays end
    with [ignore (Sys.opaque_identity ba)]; MTLBuffer [contents] stays valid
    while the buffer is retained). See {!Metal_bindings.msg_void_blocking_lazy}.
*)
let memcpy_lazy =
  lazy
    (foreign
       ~release_runtime_lock:true
       "memcpy"
       (ptr void @-> ptr void @-> size_t @-> returning (ptr void)))

let memcpy ~dst ~src ~size =
  let _ = Lazy.force memcpy_lazy dst src (Unsigned.Size_t.of_int size) in
  ()

(** {1 Exceptions} *)

(** Deprecated: no longer raised. [check] below now raises the canonical
    {!Metal_error.Metal_error} (a [Backend_error] alias) via the shared
    {!Sarek_backend_error.Backend_error.Make.check} funnel. Kept only so
    out-of-tree code matching on [Metal_api.Metal_error msg] still compiles
    (this library is opam-published). *)
exception
  Metal_error of string
      [@ocaml.deprecated
        "no longer raised; Metal_api.check now raises Metal_error.Metal_error \
         (Backend_error) - catch that instead"]

(** Check Metal result and raise a canonical {!Backend_error} on failure. *)
let check (ctx : string) (result : mtl_error) : unit =
  Metal_error.check
    ~is_success:(fun r -> r = MTL_SUCCESS)
    ~to_string:string_of_mtl_error
    ctx
    result

(** {1 Device Management} *)

module Device = struct
  type t = {
    id : int;
    handle : mtl_device;
    name : string;
    max_threads_per_threadgroup : mtl_size structure;
    max_threadgroup_memory : int;
    (* Always [false]. The Metal Shading Language has no `double` type on any
       Apple GPU, so this is a property of MSL rather than of the device, and
       [get] below sets it unconditionally.

       This field carried the comment "Metal always supports FP64 on macOS"
       until #141 — the exact opposite of the value set eleven lines below it,
       and of the truth. Sarek_real64 reads this field and selects
       [Fallback_df64] (double-float: an unevaluated pair of binary32, ~2^-46,
       needing no hardware fp64), which is the supported route to extra
       precision on Metal and is measured coherent on an Apple M4. The Metal
       codegen refuses [TFloat64] outright — see
       [Sarek_capability.float64_absent_metal], the [Backend_structural] entry
       this fact is now stated in exactly once. *)
    supports_fp64 : bool;
    is_cpu : bool; (* Metal doesn't distinguish CPU/GPU - always false *)
  }

  let init () =
    (* Metal doesn't require explicit initialization *)
    ()

  let count () =
    let devices_array = mtl_copy_all_devices () in
    let n =
      if is_null devices_array then 0
      else
        (* Get count from NSArray *)
        let sel = sel_registerName "count" in
        let fn =
          foreign
            ~from:(get_objc_lib ())
            "objc_msgSend"
            (ptr void @-> ptr void @-> returning uint64_t)
        in
        Unsigned.UInt64.to_int (fn devices_array sel)
    in
    if n > 0 then n
    else
      (* Fallback: Check for system default device if enumeration failed *)
      let default_dev = mtl_create_system_default_device () in
      if is_null default_dev then 0 else 1

  let get_all_handles () =
    let devices_array = mtl_copy_all_devices () in
    if is_null devices_array then [||]
    else
      let sel_count = sel_registerName "count" in
      let fn_count =
        foreign
          ~from:(get_objc_lib ())
          "objc_msgSend"
          (ptr void @-> ptr void @-> returning uint64_t)
      in
      let count = Unsigned.UInt64.to_int (fn_count devices_array sel_count) in

      let sel_obj = sel_registerName "objectAtIndex:" in
      let fn_obj =
        foreign
          ~from:(get_objc_lib ())
          "objc_msgSend"
          (ptr void @-> ptr void @-> uint64_t @-> returning mtl_device)
      in

      Array.init count (fun i ->
          fn_obj devices_array sel_obj (Unsigned.UInt64.of_int i))

  let make_device idx handle =
    let name = mtl_device_name handle in

    let max_threads =
      try mtl_device_max_threads_per_threadgroup handle
      with _ -> make_mtl_size ~width:1024 ~height:1024 ~depth:64
    in

    let max_threadgroup_memory =
      try mtl_device_max_threadgroup_memory_length handle with _ -> 32768
    in

    {
      id = idx;
      handle;
      name;
      max_threads_per_threadgroup = max_threads;
      max_threadgroup_memory;
      supports_fp64 = false;
      (* Metal shading language does NOT support double precision *)
      is_cpu = false;
      (* Metal devices are GPU-like *)
    }

  let get_uncached idx =
    if idx = 0 then
      (* Get default device *)
      let dev = mtl_create_system_default_device () in
      if is_null dev then
        (* Fallback: Try to get from list *)
        let devices = get_all_handles () in
        if Array.length devices > 0 then make_device 0 devices.(0)
        else
          Metal_error.raise_error
            (Metal_error.backend_unavailable "No Metal device found")
      else make_device 0 dev
    else
      (* Get from device list *)
      let devices = get_all_handles () in
      if idx >= Array.length devices then
        Metal_error.raise_error
          (Metal_error.device_not_found idx (Array.length devices))
      else make_device idx devices.(idx)

  (* Device records are immutable and an MTLDevice outlives the process, so
     [get] is memoised: Execute and the plugin look a device up on every
     launch, and building the record costs several Objective-C round trips. *)
  let cache : (int, t) Hashtbl.t = Hashtbl.create 4

  let cache_lock = Mutex.create ()

  let get idx =
    match Mutex.protect cache_lock (fun () -> Hashtbl.find_opt cache idx) with
    | Some d -> d
    | None ->
        let d = get_uncached idx in
        Mutex.protect cache_lock (fun () ->
            match Hashtbl.find_opt cache idx with
            | Some d' -> d'
            | None ->
                Hashtbl.replace cache idx d ;
                d)

  let id dev = dev.id

  let name dev = dev.name

  let handle dev = dev.handle

  (** The largest MTLBuffer this device allocates, in bytes ([maxBufferLength]).
  *)
  let max_buffer_length dev = mtl_device_max_buffer_length dev.handle

  (** [recommendedMaxWorkingSetSize] in bytes: on Apple silicon, the share of
      unified memory the GPU may keep resident without degrading. *)
  let recommended_working_set dev =
    mtl_device_recommended_max_working_set_size dev.handle
end

(** {1 Host memory} *)

(** The OS page size in bytes (16384 on Apple silicon). *)
let page_size =
  let getpagesize = lazy (foreign "getpagesize" (void @-> returning int)) in
  fun () -> (Lazy.force getpagesize) ()

(** [byte_size] bytes of host memory aligned to the OS page size
    ([posix_memalign]), uninitialised. Released only by {!free_aligned}. *)
let alloc_aligned =
  let posix_memalign =
    lazy
      (foreign
         "posix_memalign"
         (ptr (ptr void) @-> size_t @-> size_t @-> returning int))
  in
  fun ~byte_size ->
    let out = allocate (ptr void) null in
    let rc =
      (Lazy.force posix_memalign)
        out
        (Unsigned.Size_t.of_int (page_size ()))
        (Unsigned.Size_t.of_int byte_size)
    in
    if rc <> 0 || is_null !@out then
      Metal_error.raise_error
        (Metal_error.memory_allocation_failed
           (Int64.of_int byte_size)
           (Printf.sprintf "posix_memalign failed (error %d)" rc))
    else raw_address_of_ptr !@out

let free_aligned =
  let free_c = lazy (foreign "free" (ptr void @-> returning void)) in
  fun addr -> (Lazy.force free_c) (ptr_of_raw_address addr)

(** {1 Command Queue Management} *)

module CommandQueue = struct
  type t = {handle : mtl_command_queue; device : Device.t}

  let create device =
    let queue = mtl_device_new_command_queue device.Device.handle in
    if is_null queue then
      Metal_error.raise_error
        (Metal_error.context_error
           "mtl_device_new_command_queue"
           "Failed to create command queue")
    else {handle = queue; device}

  let release queue = release queue.handle

  let device queue = queue.device
end

(** Force every lazily built FFI stub on the launch, copy and release paths (see
    {!Metal_bindings.force_all}). *)
let force_all () =
  Metal_bindings.force_all () ;
  memcpy
    ~dst:(to_voidp (allocate char 'a'))
    ~src:(to_voidp (allocate char 'b'))
    ~size:1 ;
  ignore (page_size ()) ;
  free_aligned (alloc_aligned ~byte_size:1)

(** {1 Pending GPU work}

    Launches are asynchronous: {!Kernel.execute} appends a dispatch to its
    queue's open command buffer and returns. At each launch the open command
    buffer is committed if the queue's previous one has completed (the GPU would
    otherwise idle) or once it holds [SPOC_METAL_BATCH] dispatches; otherwise it
    stays open until the next launch, wait, event record or synchronisation
    commits it. This table remembers every open or committed command buffer that
    nobody has waited for yet, per command queue, with the MTLBuffers it binds,
    so that every point where the host can observe or modify device memory waits
    for exactly the work that touches it:

    - a host read or write of a buffer (whole or ranged copy, read-back) waits
      for the pending command buffers that bind that buffer ({!wait_buffers});
    - a launch on one queue first waits for pending work on OTHER queues that
      binds any of its buffers ({!wait_buffers_on_other_queues}). Within one
      queue Metal orders command buffers itself: buffers are allocated with the
      default tracked hazard mode, so a command buffer reading what an earlier
      one on the same queue wrote waits for it on the GPU;
    - device synchronisation waits for everything ({!wait_all}), stream
      synchronisation for its queue ({!wait_queue}).

    The two waits run before, not atomically with, the copy or launch they
    guard: two domains that launch on different queues, or copy and launch, on
    the same buffers at the same moment must order themselves, as on every
    backend.

    Freeing an MTLBuffer needs no wait: a command buffer retains the buffers it
    binds until it completes, so a release only drops SPOC's reference.

    A command buffer that ends in [MTLCommandBufferStatusError] raises
    [kernel_launch_failed] at the first wait that observes it, naming every
    kernel the command buffer ran: the command buffer, not the dispatch, is the
    unit of failure, and that wait may be for an unrelated buffer of the same
    batch. Failures still pending at exit are waited for and reported on stderr
    ([at_exit]). *)
module Pending = struct
  (** A committed command buffer nobody has waited for yet. *)
  type entry = {
    cmd : unit ptr;  (** Retained by the table until removed. *)
    names : string list;  (** Kernels it runs (first few), for errors. *)
    buffers : nativeint array;  (** MTLBuffer handles its dispatches bind. *)
  }

  (** The open (not yet committed) command buffer of a queue, with its compute
      encoder; launches append dispatches to it. *)
  type batch = {
    bcmd : unit ptr;  (** Retained. *)
    encoder : unit ptr;  (** Retained. *)
    mutable bnames : string list;
    mutable bbufs : nativeint list;
    mutable count : int;
  }

  type queue_state = {
    mutable entries : entry list;  (** Newest first. *)
    mutable open_batch : batch option;
  }

  let lock = Mutex.create ()

  let queues : (nativeint, queue_state) Hashtbl.t = Hashtbl.create 8

  let key (p : unit ptr) = raw_address_of_ptr p

  (** Most dispatches one command buffer collects before it is committed even
      though the GPU is still busy ([SPOC_METAL_BATCH], default 32). *)
  let batch_limit =
    match Option.bind (Sys.getenv_opt "SPOC_METAL_BATCH") int_of_string_opt with
    | Some n when n >= 1 -> n
    | _ -> 32

  (* Queues whose entry list grows past this are pruned of completed command
     buffers when the next batch is committed. *)
  let prune_threshold = 64

  let mtl_status_completed = 4

  let state_locked q =
    match Hashtbl.find_opt queues q with
    | Some st -> st
    | None ->
        let st = {entries = []; open_batch = None} in
        Hashtbl.replace queues q st ;
        st

  (* Every distinct kernel name of a batch (at most [batch_limit] of them). *)
  let add_name n names = if List.mem n names then names else n :: names

  (** Command buffers committed so far (for tests: batching keeps it well below
      the number of launches when the GPU is the bottleneck). *)
  let commits = Atomic.make 0

  (* End the open batch of [st] and commit it. Called with [lock] held. *)
  let commit_locked st =
    match st.open_batch with
    | None -> ()
    | Some b ->
        st.open_batch <- None ;
        Atomic.incr commits ;
        mtl_compute_command_encoder_end_encoding b.encoder ;
        release b.encoder ;
        mtl_command_buffer_commit b.bcmd ;
        st.entries <-
          {cmd = b.bcmd; names = b.bnames; buffers = Array.of_list b.bbufs}
          :: st.entries ;
        if List.compare_length_with st.entries prune_threshold > 0 then
          (* Completed entries are dropped; failed ones stay so that a later
             wait reports them. *)
          st.entries <-
            List.filter
              (fun e ->
                if mtl_command_buffer_status e.cmd = mtl_status_completed then begin
                  release e.cmd ;
                  false
                end
                else true)
              st.entries

  let open_batch_locked q st ~name =
    match st.open_batch with
    | Some b -> b
    | None ->
        let b =
          with_autorelease_pool (fun () ->
              let cmd = mtl_command_queue_command_buffer q in
              if is_null cmd then
                Metal_error.raise_error
                  (Metal_error.kernel_launch_failed
                     name
                     "Failed to create command buffer") ;
              let encoder = mtl_command_buffer_compute_command_encoder cmd in
              if is_null encoder then
                Metal_error.raise_error
                  (Metal_error.kernel_launch_failed
                     name
                     "Failed to create compute encoder") ;
              (* Owned past the pool: released once committed / waited for. *)
              ignore (retain cmd) ;
              ignore (retain encoder) ;
              {bcmd = cmd; encoder; bnames = []; bbufs = []; count = 0})
        in
        st.open_batch <- Some b ;
        b

  (** Append one dispatch to queue [q]'s open command buffer ([encode] receives
      the compute encoder). The batch is committed when it holds {!batch_limit}
      dispatches, or at once when the queue's previous command buffer has
      completed (the GPU would otherwise sit idle). *)
  let encode q ~name ~(buffers : unit ptr list) (encode : unit ptr -> unit) =
    Mutex.protect lock (fun () ->
        let st = state_locked (key q) in
        let b = open_batch_locked q st ~name in
        encode b.encoder ;
        b.count <- b.count + 1 ;
        b.bnames <- add_name name b.bnames ;
        List.iter
          (fun h ->
            let h = key h in
            if not (List.mem h b.bbufs) then b.bbufs <- h :: b.bbufs)
          buffers ;
        let gpu_idle =
          match st.entries with
          | [] -> true
          | e :: _ -> mtl_command_buffer_status e.cmd >= mtl_status_completed
        in
        if gpu_idle || b.count >= batch_limit then commit_locked st)

  (* Commit the open batches [select] picks ([~commit_all]: every open batch),
     then wait for every committed entry it picks. Entries are retained for the
     wait and removed from the table only after they completed, so a
     concurrent waiter never sees a buffer as idle while work on it runs. *)
  let wait_selected ?(commit_all = false) ?(cmd : unit ptr option)
      (select : nativeint -> nativeint array -> bool) =
    (* [~cmd]: pick exactly that committed command buffer (an event's). *)
    let select q bufs e_cmd =
      match cmd with
      | Some c -> ( match e_cmd with Some ec -> ec == c | None -> false)
      | None -> select q bufs
    in
    let picked =
      Mutex.protect lock (fun () ->
          (* A host wait commits every open batch, not only the selected ones,
             so trailing dispatches do not sit uncommitted while the GPU idles.
             The cross-queue check a launch makes does not ([commit_all =
             false]): committing there would end every batch after one
             dispatch. *)
          Hashtbl.iter
            (fun q st ->
              match st.open_batch with
              | Some b when commit_all || select q (Array.of_list b.bbufs) None
                ->
                  commit_locked st
              | _ -> ())
            queues ;
          Hashtbl.fold
            (fun q st acc ->
              List.fold_left
                (fun acc e ->
                  if select q e.buffers (Some e.cmd) then begin
                    ignore (retain e.cmd) ;
                    (q, e) :: acc
                  end
                  else acc)
                acc
                st.entries)
            queues
            [])
    in
    if picked <> [] then begin
      (* Every failed command buffer picked is reported (it is removed from
         the table below, so this is its only report). *)
      let failures = ref [] in
      (* Removes the picked entries still in the table and releases them;
         only the waiter that removes a failed entry reports it, so a failure
         is reported once even when two domains wait on it. *)
      let remove_and_release () =
        Mutex.protect lock (fun () ->
            List.iter
              (fun (q, e) ->
                (match Hashtbl.find_opt queues q with
                | Some st when List.memq e st.entries ->
                    st.entries <- List.filter (fun x -> x != e) st.entries ;
                    if
                      mtl_command_buffer_status e.cmd
                      = mtl_command_buffer_status_error
                    then
                      failures :=
                        ( String.concat ", " (List.rev e.names),
                          mtl_command_buffer_error_description e.cmd )
                        :: !failures ;
                    release e.cmd
                | _ -> ()) ;
                release e.cmd)
              picked)
      in
      Fun.protect ~finally:remove_and_release (fun () ->
          List.iter
            (fun (_, e) -> mtl_command_buffer_wait_until_completed e.cmd)
            picked) ;
      match List.rev !failures with
      | [] -> ()
      | (names, msg) :: rest ->
          let more =
            List.map
              (fun (n, m) -> Printf.sprintf "; also failed: [%s] %s" n m)
              rest
          in
          Metal_error.raise_error
            (Metal_error.kernel_launch_failed
               names
               (Printf.sprintf
                  "command buffer failed (one of its dispatches: the command \
                   buffer, not the kernel, is the unit of failure): %s%s"
                  msg
                  (String.concat "" more)))
    end

  let touches hs bufs = Array.exists (fun b -> List.mem b hs) bufs

  (** Wait for all pending work on every queue. *)
  let wait_all () = wait_selected ~commit_all:true (fun _ _ -> true)

  (** Wait for all pending work on queue [q]. *)
  let wait_queue q =
    let k = key q in
    wait_selected ~commit_all:true (fun q' _ -> Nativeint.equal q' k)

  (** Wait for pending work that binds any of the MTLBuffers [hs]. *)
  let wait_buffers (hs : unit ptr list) =
    let hs = List.map key hs in
    wait_selected ~commit_all:true (fun _ bufs -> touches hs bufs)

  (** Wait for the pending command buffer [c] (an event's), reporting its
      failure through the table so it is reported once. No-op when [c] is no
      longer pending (someone already waited for it and reported it). *)
  let wait_cmd c = wait_selected ~cmd:c (fun _ _ -> false)

  (** Wait for pending work on queues other than [q] that binds any of [hs]. *)
  let wait_buffers_on_other_queues q (hs : unit ptr list) =
    let k = key q and hs = List.map key hs in
    wait_selected (fun q' bufs ->
        (not (Nativeint.equal q' k)) && touches hs bufs)

  (** Commit queue [q]'s open batch, then return its newest command buffer,
      retained for the caller (who releases it), or [None] when the queue has no
      pending work. *)
  let last_on_queue q =
    Mutex.protect lock (fun () ->
        match Hashtbl.find_opt queues (key q) with
        | None -> None
        | Some st -> (
            commit_locked st ;
            match st.entries with
            | e :: _ ->
                ignore (retain e.cmd) ;
                Some e.cmd
            | [] -> None))

  (** Forget queue [q] (after waiting for it): its handle may be reused. *)
  let forget_queue q =
    Mutex.protect lock (fun () ->
        match Hashtbl.find_opt queues (key q) with
        | None -> ()
        | Some st ->
            (* Work enqueued after the caller's wait still runs: commit it.
               Releasing a committed command buffer does not cancel it. *)
            commit_locked st ;
            (* Completed entries go; unfinished or failed ones stay (under the
               old handle) so a later wait or the at-exit report sees them. A
               new queue reusing the handle only over-waits. *)
            st.entries <-
              List.filter
                (fun e ->
                  if mtl_command_buffer_status e.cmd = mtl_status_completed then begin
                    release e.cmd ;
                    false
                  end
                  else true)
                st.entries ;
            if st.entries = [] then Hashtbl.remove queues (key q))

  (** Number of queues with pending (open or unwaited) command buffers. For
      tests: a launch that returned before completing leaves one. *)
  let queues_with_pending () =
    Mutex.protect lock (fun () ->
        Hashtbl.fold
          (fun _ st n ->
            if st.entries <> [] || Option.is_some st.open_batch then n + 1
            else n)
          queues
          0)

  (* A failure nobody waited for would otherwise vanish with the process. *)
  let () =
    at_exit (fun () ->
        try wait_all ()
        with e ->
          Printf.eprintf
            "sarek-metal: pending GPU work failed at exit: %s\n%!"
            (Printexc.to_string e))
end

(** {1 Memory Management} *)

module Memory = struct
  type 'a buffer = {
    handle : mtl_buffer;
    size : int;
    elem_size : int;
    device : Device.t;
    contents : unit ptr; (* Pointer to GPU memory *)
  }

  let alloc device size elem_size =
    let byte_size = size * elem_size in
    (* Refuse above maxBufferLength before asking Metal, whose only answer is
       a null buffer: name the limit so the caller knows to split. *)
    let limit = Device.max_buffer_length device in
    if byte_size > limit then
      Metal_error.raise_error
        (Metal_error.memory_allocation_failed
           (Int64.of_int byte_size)
           (Printf.sprintf
              "%d bytes exceed this device's maxBufferLength (%d bytes); split \
               the data over several buffers"
              byte_size
              limit)) ;
    (* Shared storage: on Apple silicon CPU and GPU address the same physical
       memory, so [contents] is the buffer itself and a host<->device copy is
       one memcpy. Device-only vectors use it too (see Metal_plugin_base). *)
    let buf =
      mtl_device_new_buffer_with_length
        device.Device.handle
        byte_size
        mtl_resource_storage_mode_shared
    in
    if is_null buf then
      Metal_error.raise_error
        (Metal_error.memory_allocation_failed
           (Int64.of_int byte_size)
           "mtl_device_new_buffer_with_length returned null")
    else
      let contents = mtl_buffer_contents buf in
      {handle = buf; size; elem_size; device; contents}

  (* [Spoc_core.Memory.borrow_bigarray_ptr] rather than [Ctypes.bigarray_start]:
     ctypes' kind GADT has no Float16 arm. The pointer does not root the array;
     the [Sys.opaque_identity] after each memcpy does. *)
  let alloc_bigarray device ba elem_size =
    let size = Bigarray.Array1.dim ba in
    let buf = alloc device size elem_size in
    (* Copy data to GPU *)
    let ba_ptr = Spoc_core.Memory.borrow_bigarray_ptr ba in
    let byte_size = size * elem_size in
    memcpy ~dst:buf.contents ~src:ba_ptr ~size:byte_size ;
    ignore (Sys.opaque_identity ba) ;
    buf

  let to_bigarray (type a b) buf (kind : (a, b) Bigarray.kind) =
    Pending.wait_buffers [buf.handle] ;
    let ba = Bigarray.Array1.create kind Bigarray.c_layout buf.size in
    let ba_ptr = Spoc_core.Memory.borrow_bigarray_ptr ba in
    let byte_size = buf.size * buf.elem_size in
    memcpy ~dst:ba_ptr ~src:buf.contents ~size:byte_size ;
    ignore (Sys.opaque_identity ba) ;
    ba

  let release buf = release buf.handle

  let size buf = buf.size

  let handle buf = buf.handle

  let contents buf = buf.contents
end

(** {1 Library and Function Management} *)

module Library = struct
  type t = {handle : mtl_library; device : Device.t}

  (** Compile MSL source.

      Requests Metal's non-fast-math mode (backlog #125). Until this change the
      options argument was hardcoded [None] AND the binding ignored it
      regardless, so every Sarek Metal kernel compiled under Metal's fast-math
      default with no way to turn it off.

      These options are NOT a contraction defence — measured on an Apple M4,
      [a*b+c] is contracted under every setting of them, and only
      [Sarek_ir_metal.metal_fp_contract_pragma] in the generated source stops
      it. What they buy is math-function accuracy, and that much is measured
      (22135 of 65536 results change on [sqrt(a) + 1/a]).

      If [mtl_compile_options_conformant] returns [None] (older OS, missing
      selector, allocation failure) this falls back to null options, i.e.
      exactly the behaviour before backlog #125, and says so in the log rather
      than silently.

      Metal float results are STILL outside the guarantee: [test_df64] and
      [test_real64] have never been run on a Mac, so agreement with the
      interpreter is not established. See docs/fp-contraction-policy.md §10. *)
  let create_from_source device source =
    let options = mtl_compile_options_conformant () in
    (match options with
    | Some _ ->
        Spoc_core.Log.debugf
          Spoc_core.Log.Kernel
          "Metal: compiling with safe math mode (mathMode=Safe + \
           mathFloatingPointFunctions=Precise, or fastMathEnabled=NO on macOS \
           < 15). Contraction is handled separately, by the #pragma METAL fp \
           contract(off) in the generated source."
    | None ->
        Spoc_core.Log.warnf
          Spoc_core.Log.Kernel
          "Metal: could not build MTLCompileOptions, falling back to Metal's \
           DEFAULT compile options, whose math mode is FAST on both knobs \
           (measured on Apple M4). Single-precision math functions will \
           resolve to metal::fast and may disagree with the Sarek interpreter. \
           See docs/fp-contraction-policy.md §10.") ;
    let result =
      mtl_device_new_library_with_source device.Device.handle source options
    in
    (match options with Some o -> release o | None -> ()) ;
    match result with
    | Ok lib -> {handle = lib; device}
    | Error msg ->
        Metal_error.raise_error (Metal_error.compilation_failed source msg)

  let release lib = release lib.handle

  let get_function lib name =
    let func = mtl_library_new_function_with_name lib.handle name in
    if is_null func then
      Metal_error.raise_error
        (Metal_error.context_error
           "mtl_library_new_function_with_name"
           (Printf.sprintf "Function '%s' not found in library" name))
    else func
end

(** {1 Compute Pipeline Management} *)

module ComputePipeline = struct
  type t = {
    handle : mtl_compute_pipeline_state;
    device : Device.t;
    max_threads_per_threadgroup : int;
    thread_execution_width : int;
  }

  let create device func =
    match mtl_device_new_compute_pipeline_state device.Device.handle func with
    | Ok pso ->
        let max_threads =
          mtl_compute_pipeline_state_max_total_threads_per_threadgroup pso
        in
        let thread_width =
          mtl_compute_pipeline_state_threadgroup_memory_length pso
        in
        {
          handle = pso;
          device;
          max_threads_per_threadgroup = max_threads;
          thread_execution_width = thread_width;
        }
    | Error msg ->
        Metal_error.raise_error
          (Metal_error.context_error
             "mtl_device_new_compute_pipeline_state"
             (Printf.sprintf "Pipeline state creation failed: %s" msg))

  let release pipeline = release pipeline.handle

  let handle pipeline = pipeline.handle
end

(** {1 Kernel Execution} *)

module Kernel = struct
  type arg =
    | Buffer of unit ptr * int (* buffer, offset *)
    | Int32 of int32
    | Int64 of int64
    | Float32 of float
    | Float64 of float

  type args = arg list

  type t = {pipeline : ComputePipeline.t; function_name : string}

  let create pipeline function_name = {pipeline; function_name}

  let encode_dispatch encoder kernel ~grid_size ~block_size args =
    (* Set pipeline state *)
    mtl_compute_command_encoder_set_compute_pipeline_state
      encoder
      kernel.pipeline.handle ;

    (* Set arguments *)
    List.iteri
      (fun idx arg ->
        match arg with
        | Buffer (ptr, offset) ->
            (* For Metal buffers, we pass the buffer directly *)
            mtl_compute_command_encoder_set_buffer encoder ptr offset idx
        | Int32 v ->
            let ptr = allocate int32_t v in
            mtl_compute_command_encoder_set_bytes
              encoder
              (to_voidp ptr)
              (sizeof int32_t)
              idx
        | Int64 v ->
            let ptr = allocate int64_t v in
            mtl_compute_command_encoder_set_bytes
              encoder
              (to_voidp ptr)
              (sizeof int64_t)
              idx
        | Float32 v ->
            let ptr = allocate float v in
            mtl_compute_command_encoder_set_bytes
              encoder
              (to_voidp ptr)
              (sizeof float)
              idx
        | Float64 v ->
            let ptr = allocate double v in
            mtl_compute_command_encoder_set_bytes
              encoder
              (to_voidp ptr)
              (sizeof double)
              idx)
      args ;

    (* Dispatch threads *)
    let grid_x, grid_y, grid_z = grid_size in
    let block_x, block_y, block_z = block_size in
    let threads = make_mtl_size ~width:grid_x ~height:grid_y ~depth:grid_z in
    let threadgroup =
      make_mtl_size ~width:block_x ~height:block_y ~depth:block_z
    in
    mtl_compute_command_encoder_dispatch_threads encoder threads threadgroup

  (** Append one dispatch to [queue]'s open command buffer and return without
      waiting (see {!Pending}). Work on other queues that binds the same buffers
      is waited for first. *)
  let execute queue kernel ~grid_size ~block_size args =
    let buffers =
      List.filter_map (function Buffer (b, _) -> Some b | _ -> None) args
    in
    let qh = queue.CommandQueue.handle in
    Pending.wait_buffers_on_other_queues qh buffers ;
    Pending.encode qh ~name:kernel.function_name ~buffers (fun encoder ->
        encode_dispatch encoder kernel ~grid_size ~block_size args)
end

(** {1 Synchronization} *)

let synchronize _device =
  (* Launches are asynchronous: wait for all pending work (see Pending). *)
  Pending.wait_all ()

(** {1 Profiling} *)

module Profiling = struct
  type event = unit (* Metal uses different profiling mechanism *)

  let enabled = ref false

  let enable () = enabled := true

  let disable () = enabled := false
end
