(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Streams and the launch batches they record (specs/vulkan-async-launches.md).

    A launch is appended to the stream's open command buffer and returns. The
    buffer is submitted (not waited for) when it holds {!batch_limit}
    dispatches, when another stream of the device launches, and at any
    synchronisation point; the host blocks only at {!synchronize}, at
    {!Vulkan_api_device.drain_pending} (host access to device memory, buffer
    release, device wait) and when a command buffer is reused. A stream has two
    command buffers, so one can be recorded while the other runs.

    All streams of a device submit to one queue, in launch order, because a
    launch first submits the open batches of the device's other streams;
    dispatches are ordered in memory by a barrier recorded before each one. *)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base
module Device = Vulkan_api_device
module BE = Sarek_backend_error.Backend_error

(** Dispatches one command buffer collects before it is submitted, from
    [SPOC_VULKAN_BATCH] (a positive integer; anything else gives the default).
    [1] submits and waits at every launch: the behaviour before batching. *)
let default_batch = 32

let batch_limit =
  ref
    (match
       Option.bind (Sys.getenv_opt "SPOC_VULKAN_BATCH") int_of_string_opt
     with
    | Some n when n >= 1 -> n
    | Some _ | None -> default_batch)

(** One command buffer with what its dispatches use. Idle: not [recording] and
    fence not pending. Open: [recording]. In flight: fence pending. *)
type slot = {
  command_buffer : vk_command_buffer structure ptr;
  fence : tracked_fence;
  pools : Vulkan_api_pools.t;
  mutable kernels : string list;  (** Newest first, one entry per dispatch. *)
  mutable dispatches : int;
  mutable recording : bool;
}

type t = {
  command_pool : vk_command_pool;
  slots : slot array;
  mutable active : int;  (** Index of the slot the next dispatch records in. *)
  mutable destroyed : bool;
  mutable deferred : (string * string) option;
      (** Kernels and reason of a submission that failed while launching; raised
          by the next wait of this stream. *)
  device : Device.t;
}

let allocate_command_buffer device command_pool =
  let alloc_info = make vk_command_buffer_allocate_info in
  setf
    alloc_info
    cmd_buf_alloc_sType
    (u32 vk_structure_type_command_buffer_allocate_info) ;
  setf alloc_info cmd_buf_alloc_pNext null ;
  setf alloc_info cmd_buf_alloc_commandPool command_pool ;
  setf alloc_info cmd_buf_alloc_level (u32 vk_command_buffer_level_primary) ;
  setf alloc_info cmd_buf_alloc_commandBufferCount (u32 1) ;
  let cmd_buf =
    allocate vk_command_buffer_ptr (from_voidp vk_command_buffer null)
  in
  check
    "vkAllocateCommandBuffers"
    (vkAllocateCommandBuffers device.Device.device (addr alloc_info) cmd_buf) ;
  !@cmd_buf

let create_slot device command_pool =
  {
    command_buffer = allocate_command_buffer device command_pool;
    fence = create_tracked_fence device.Device.device;
    pools = Vulkan_api_pools.create ();
    kernels = [];
    dispatches = 0;
    recording = false;
  }

(** {1 Registry}

    The streams of each device, so that a launch, a host access or a device wait
    can reach the batches of every stream. *)

let registry : (int, t list) Hashtbl.t = Hashtbl.create 4

let registry_lock = Mutex.create ()

let default_streams : (int, t) Hashtbl.t = Hashtbl.create 4

(** Guards [default_streams]. *)
let default_lock = Mutex.create ()

let streams_of (device : Device.t) =
  Mutex.protect registry_lock (fun () ->
      Option.value ~default:[] (Hashtbl.find_opt registry device.Device.id))

let register stream =
  let id = stream.device.Device.id in
  Mutex.protect registry_lock (fun () ->
      let known = Option.value ~default:[] (Hashtbl.find_opt registry id) in
      Hashtbl.replace registry id (stream :: known))

let unregister stream =
  let id = stream.device.Device.id in
  Mutex.protect registry_lock (fun () ->
      let known = Option.value ~default:[] (Hashtbl.find_opt registry id) in
      Hashtbl.replace registry id (List.filter (fun s -> s != stream) known))

let create device =
  let pool_info = make vk_command_pool_create_info in
  setf
    pool_info
    cmd_pool_create_sType
    (u32 vk_structure_type_command_pool_create_info) ;
  setf pool_info cmd_pool_create_pNext null ;
  (* VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT *)
  setf pool_info cmd_pool_create_flags (u32 0x02) ;
  setf
    pool_info
    cmd_pool_create_queueFamilyIndex
    (u32 device.Device.queue_family) ;
  let pool = allocate vk_command_pool vk_null_handle in
  check
    "vkCreateCommandPool"
    (vkCreateCommandPool device.Device.device (addr pool_info) null pool) ;
  let command_pool = !@pool in
  let slots =
    [|create_slot device command_pool; create_slot device command_pool|]
  in
  let stream =
    {
      command_pool;
      slots;
      active = 0;
      destroyed = false;
      deferred = None;
      device;
    }
  in
  register stream ;
  stream

(** {1 Batches}

    Every function below runs with [stream.device.batch_lock] held. *)

(** Kernel names of [slot] in launch order, a repeated name written once with
    its count. *)
let kernel_names slot =
  let add counts name =
    if List.mem_assoc name counts then
      List.map
        (fun (n, c) -> if String.equal n name then (n, c + 1) else (n, c))
        counts
    else counts @ [(name, 1)]
  in
  List.fold_left add [] (List.rev slot.kernels)
  |> List.map (fun (name, n) ->
      if n = 1 then name else Printf.sprintf "%s x%d" name n)
  |> String.concat ", "

(** The slot's work is over (completed, or never submitted): its descriptor sets
    and kernel list are released. *)
let retire stream slot =
  Vulkan_api_pools.reset stream.device slot.pools ;
  slot.kernels <- [] ;
  slot.dispatches <- 0 ;
  slot.recording <- false

(** Wait for the slot's submission, if any. A failing wait raises the batch
    error naming its kernels; the slot then stays in flight, since the device
    did not say that the work is over. *)
let await_slot stream slot =
  if slot.fence.pending then begin
    (match
       await_fence
         "vkWaitForFences (launch batch)"
         stream.device.Device.device
         slot.fence
     with
    | () -> ()
    | exception Vulkan_error.Vulkan_error e ->
        Vulkan_error.raise_error
          (Vulkan_error.kernel_launch_failed
             (kernel_names slot)
             (BE.to_string e))) ;
    retire stream slot
  end

let raise_deferred stream =
  match stream.deferred with
  | None -> ()
  | Some (kernels, reason) ->
      stream.deferred <- None ;
      Vulkan_error.raise_error
        (Vulkan_error.kernel_launch_failed kernels reason)

let submit_slot stream slot =
  let device = stream.device in
  Vulkan_api_barrier.end_of_batch slot.command_buffer ;
  check "vkEndCommandBuffer" (vkEndCommandBuffer slot.command_buffer) ;
  let cmd_buf_ptr = allocate vk_command_buffer_ptr slot.command_buffer in
  let submit_info = make vk_submit_info in
  setf submit_info submit_sType (u32 vk_structure_type_submit_info) ;
  setf submit_info submit_pNext null ;
  setf submit_info submit_waitSemaphoreCount (u32 0) ;
  setf submit_info submit_pWaitSemaphores (from_voidp vk_semaphore null) ;
  setf submit_info submit_pWaitDstStageMask (from_voidp vk_flags null) ;
  setf submit_info submit_commandBufferCount (u32 1) ;
  setf submit_info submit_pCommandBuffers cmd_buf_ptr ;
  setf submit_info submit_signalSemaphoreCount (u32 0) ;
  setf submit_info submit_pSignalSemaphores (from_voidp vk_semaphore null) ;
  submit_with_fence
    "vkQueueSubmit (launch batch)"
    device.Device.device
    slot.fence
    (Device.queue_submit device (u32 1) (addr submit_info)) ;
  ignore (Sys.opaque_identity cmd_buf_ptr) ;
  ignore (Sys.opaque_identity submit_info) ;
  slot.recording <- false

(** Submit the open batch of [stream], without waiting for it. A failure retires
    the batch; it is raised now when [defer] is false, and otherwise kept for
    the next wait of the stream (the first failure wins). *)
let submit_open ~defer stream =
  let slot = stream.slots.(stream.active) in
  if slot.recording && slot.dispatches > 0 then begin
    match submit_slot stream slot with
    | () -> stream.active <- (stream.active + 1) mod Array.length stream.slots
    | exception Vulkan_error.Vulkan_error e ->
        let reason = BE.to_string e in
        let kernels = kernel_names slot in
        retire stream slot ;
        if not defer then
          Vulkan_error.raise_error
            (Vulkan_error.kernel_launch_failed kernels reason)
        else if stream.deferred = None then
          stream.deferred <- Some (kernels, reason)
  end

(** Submit and wait for everything launched on [stream]; raises the first
    failure. *)
let drain_stream stream =
  submit_open ~defer:false stream ;
  let count = Array.length stream.slots in
  for i = 1 to count do
    await_slot stream stream.slots.((stream.active + i) mod count)
  done ;
  raise_deferred stream

(** Run [f] on every stream of [device], then re-raise the first exception: one
    failing stream must not leave the others undrained. *)
let on_all_streams device f =
  let first_error = ref None in
  List.iter
    (fun stream ->
      try f stream with e -> if !first_error = None then first_error := Some e)
    (streams_of device) ;
  Option.iter raise !first_error

(** Queued buffer releases (count, bytes) after which a launch first waits for
    all launches and runs them. *)
let max_deferred_frees = 64

let max_deferred_bytes = 256 * 1024 * 1024

let is_idle slot = (not slot.recording) && not slot.fence.pending

let all_idle device =
  List.for_all
    (fun stream -> Array.for_all is_idle stream.slots)
    (streams_of device)

(** Run the queued buffer releases. Caller holds [batch_lock] and every launch
    recorded so far has completed. *)
let release_deferred device =
  List.iter (fun release -> release ()) (Device.take_deferred device)

let drain_all device =
  on_all_streams device drain_stream ;
  release_deferred device

let drain_device device =
  Mutex.protect device.Device.batch_lock (fun () -> drain_all device)

let () = Device.drain_pending := drain_device

(** {!drain_device} unless the batch lock is held. *)
let try_drain_device device =
  if Mutex.try_lock device.Device.batch_lock then
    Fun.protect
      ~finally:(fun () -> Mutex.unlock device.Device.batch_lock)
      (fun () -> drain_all device)

let () = Device.try_drain_pending := try_drain_device

(** Wait for every launch of [device], then run the queued releases, without
    consuming any stream's deferred error: it belongs to the wait of that
    stream, not to the launch that happens to trigger this. A failing wait is
    kept as that stream's deferred error and the slot stays in flight, so the
    releases wait for it. *)
let settle device =
  on_all_streams device (fun stream ->
      submit_open ~defer:true stream ;
      Array.iter
        (fun slot ->
          try await_slot stream slot
          with Vulkan_error.Vulkan_error e ->
            if stream.deferred = None then
              stream.deferred <- Some (kernel_names slot, BE.to_string e))
        stream.slots) ;
  if all_idle device then release_deferred device

(** Submit the open batch of every stream of [device] without waiting. *)
let flush_device device =
  Mutex.protect device.Device.batch_lock (fun () ->
      on_all_streams device (submit_open ~defer:true))

(** Make the active slot of [stream] ready to record: wait for its previous
    submission if it is still in flight, then begin the command buffer. *)
let open_active stream =
  let slot = stream.slots.(stream.active) in
  if not slot.recording then begin
    await_slot stream slot ;
    let begin_info = make vk_command_buffer_begin_info in
    setf
      begin_info
      cmd_buf_begin_sType
      (u32 vk_structure_type_command_buffer_begin_info) ;
    setf begin_info cmd_buf_begin_pNext null ;
    setf
      begin_info
      cmd_buf_begin_flags
      (u32 vk_command_buffer_usage_one_time_submit_bit) ;
    setf begin_info cmd_buf_begin_pInheritanceInfo null ;
    check
      "vkBeginCommandBuffer"
      (vkBeginCommandBuffer slot.command_buffer (addr begin_info)) ;
    ignore (Sys.opaque_identity begin_info) ;
    slot.recording <- true
  end ;
  slot

(** Record one dispatch of [kernel] on [stream]. [record command_buffer set]
    issues the bind, push-constant and dispatch commands (after the barrier
    recorded here) with [set], a descriptor set of [layout] that it must fill.
    Nothing is recorded when the set cannot be allocated. *)
let record_dispatch stream ~kernel ~layout ~descriptors ~record =
  let device = stream.device in
  Mutex.protect device.Device.batch_lock (fun () ->
      (* Bound the memory held by freed buffers. *)
      if
        Atomic.get device.Device.deferred_count >= max_deferred_frees
        || Atomic.get device.Device.deferred_bytes >= max_deferred_bytes
      then settle device ;
      on_all_streams device (fun other ->
          if other != stream then submit_open ~defer:true other) ;
      let slot = open_active stream in
      let set =
        Vulkan_api_pools.allocate device slot.pools layout ~descriptors
      in
      Vulkan_api_barrier.before_dispatch slot.command_buffer ;
      record slot.command_buffer set ;
      slot.kernels <- kernel :: slot.kernels ;
      slot.dispatches <- slot.dispatches + 1 ;
      if !batch_limit <= 1 then drain_stream stream
      else if slot.dispatches >= !batch_limit then
        submit_open ~defer:true stream)

let synchronize stream =
  let device = stream.device in
  Mutex.protect device.Device.batch_lock (fun () ->
      drain_stream stream ;
      if all_idle device then release_deferred device)

let destroy_slot device slot =
  destroy_tracked_fence
    "vkWaitForFences (Stream.destroy)"
    device.Device.device
    slot.fence ;
  Vulkan_api_pools.destroy device slot.pools

let destroy_objects stream =
  Array.iter (destroy_slot stream.device) stream.slots ;
  vkDestroyCommandPool stream.device.Device.device stream.command_pool null

let forget_default stream =
  Mutex.protect default_lock (fun () ->
      match Hashtbl.find_opt default_streams stream.device.Device.id with
      | Some default when default == stream ->
          Hashtbl.remove default_streams stream.device.Device.id
      | Some _ | None -> ())

(** Nothing is unregistered if the wait for pending work raises: the stream
    stays reachable and intact. *)
let destroy stream =
  if not stream.destroyed then begin
    synchronize stream ;
    stream.destroyed <- true ;
    unregister stream ;
    forget_default stream ;
    destroy_objects stream
  end

let default device =
  Mutex.protect default_lock (fun () ->
      match Hashtbl.find_opt default_streams device.Device.id with
      | Some s -> s
      | None ->
          let s = create device in
          Hashtbl.add default_streams device.Device.id s ;
          s)

(** Launches recorded on [device] that have not completed: open batches and
    batches in flight. Zero after any synchronisation point. *)
let pending_launches device =
  Mutex.protect device.Device.batch_lock (fun () ->
      List.fold_left
        (fun total stream ->
          Array.fold_left
            (fun total slot ->
              if slot.recording || slot.fence.pending then
                total + slot.dispatches
              else total)
            total
            stream.slots)
        0
        (streams_of device))

(* Device.destroy has drained the device and runs this before vkDestroyDevice:
   the streams' command pools, fences and descriptor pools are device children
   and go first. The next device with this index gets fresh streams. *)
let () =
  Spoc_framework.Cache_hooks.on_device_destroy (fun ~backend index ->
      if String.equal backend "Vulkan" then begin
        let streams =
          Mutex.protect registry_lock (fun () ->
              let known =
                Option.value ~default:[] (Hashtbl.find_opt registry index)
              in
              Hashtbl.remove registry index ;
              known)
        in
        Mutex.protect default_lock (fun () ->
            Hashtbl.remove default_streams index) ;
        let first_error = ref None in
        List.iter
          (fun stream ->
            try
              if not stream.destroyed then begin
                stream.destroyed <- true ;
                destroy_objects stream
              end
            with e -> if !first_error = None then first_error := Some e)
          streams ;
        Option.iter raise !first_error
      end)
