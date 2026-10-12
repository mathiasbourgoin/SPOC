(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** State of the launch batches: a stream's two command-buffer slots, the
    per-device registry of streams, and the helpers that need no queue access
    (specs/vulkan-async-launches.md). *)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base
module Device = Vulkan_api_device

(** One command buffer with what its dispatches use. Idle: not [recording], not
    [armed] and fence not pending. Open: [recording]. In flight: fence pending.
*)
type slot = {
  command_buffer : vk_command_buffer structure ptr;
  fence : tracked_fence;
  pools : Vulkan_api_pools.t;
  mutable kernels : string list;  (** Newest first, one entry per dispatch. *)
  mutable dispatches : int;
  mutable recording : bool;
  mutable ended : bool;
      (** Set just before [vkEndCommandBuffer] is called, after the end-of-batch
          barrier was recorded: from here the command buffer's state is unknown
          to the host after an escaped exception, and the batch may or may not
          run. *)
  mutable cb_open : bool;
      (** The command buffer may be in the recording state: set just before
          [vkBeginCommandBuffer] is called, cleared after [vkEndCommandBuffer]
          or a reset. A begin on an open buffer is invalid; {!Vulkan_api_batch}
          counts it. *)
  mutable armed : bool;
      (** The fence was reset for a submission that may have reached the queue:
          set after the reset and before [vkQueueSubmit], cleared by {!retire}.
          An exception escaping while armed leaves it unknown whether the batch
          is running. *)
}

(** An event a stream has been told to wait for ([stream_wait_event]): its fence
    and the lock that serialises waits on it. *)
type awaited_event = {event_lock : Mutex.t; event_fence : tracked_fence}

type stream = {
  command_pool : vk_command_pool;
  slots : slot array;
  mutable active : int;  (** Index of the slot the next dispatch records in. *)
  destroyed : bool Atomic.t;
  mutable deferred : (string * string) option;
      (** Kernels and reason of a submission that failed while launching; raised
          by the next wait of this stream. *)
  mutable awaited : awaited_event list;
      (** Events this stream waits for, newest first, guarded by [batch_lock].
          Later work on the stream is ordered after them by the single queue and
          the barrier before every dispatch; the host-side effect is that
          {!Vulkan_api_batch.synchronize} also waits for them. Emptied by that
          wait. *)
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
    ended = false;
    cb_open = false;
    armed = false;
  }

let create_stream device =
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
  {
    command_pool;
    slots = [|create_slot device command_pool; create_slot device command_pool|];
    active = 0;
    destroyed = Atomic.make false;
    deferred = None;
    awaited = [];
    device;
  }

(** {1 Registry}

    The streams of each device, so that a launch, a host access or a device wait
    can reach the batches of every stream. [registry_lock] guards short table
    operations only and is never held across a call that can block. *)

let registry : (int, stream list) Hashtbl.t = Hashtbl.create 4

let registry_lock = Mutex.create ()

let default_streams : (int, stream) Hashtbl.t = Hashtbl.create 4

(** Guards [default_streams]. *)
let default_lock = Mutex.create ()

let streams_of (device : Device.t) =
  Mutex.protect registry_lock (fun () ->
      List.filter
        (fun s -> s.device == device)
        (Option.value ~default:[] (Hashtbl.find_opt registry device.Device.id)))

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

(** Kernel names of [slot] in launch order, a repeated name written once with
    its count. *)
let names_of_kernels kernels =
  let add counts name =
    if List.mem_assoc name counts then
      List.map
        (fun (n, c) -> if String.equal n name then (n, c + 1) else (n, c))
        counts
    else counts @ [(name, 1)]
  in
  List.fold_left add [] (List.rev kernels)
  |> List.map (fun (name, n) ->
      if n = 1 then name else Printf.sprintf "%s x%d" name n)
  |> String.concat ", "

let kernel_names slot = names_of_kernels slot.kernels

(** The slot's work is over (completed, or never submitted): its descriptor sets
    and kernel list are released. *)
let retire stream slot =
  slot.armed <- false ;
  slot.ended <- false ;
  Vulkan_api_pools.reset stream.device slot.pools ;
  slot.kernels <- [] ;
  slot.dispatches <- 0 ;
  slot.recording <- false

let is_idle slot = (not slot.recording) && not slot.fence.pending

let all_idle device =
  List.for_all
    (fun stream -> Array.for_all is_idle stream.slots)
    (streams_of device)

let destroy_slot device slot =
  destroy_tracked_fence
    "vkWaitForFences (Stream.destroy)"
    device.Device.device
    slot.fence ;
  Vulkan_api_pools.destroy device slot.pools

let destroy_objects stream =
  Array.iter (destroy_slot stream.device) stream.slots ;
  vkDestroyCommandPool stream.device.Device.device stream.command_pool null
