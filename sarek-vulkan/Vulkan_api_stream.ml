(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base
module Device = Vulkan_api_device

type t = {
  command_pool : vk_command_pool;
  command_buffer : vk_command_buffer structure ptr;
  fence : tracked_fence;
      (** Guards [command_buffer]: pending while a submission of it runs. *)
  device : Device.t;
}

let create device =
  (* Create command pool *)
  let pool_info = make vk_command_pool_create_info in
  setf
    pool_info
    cmd_pool_create_sType
    (u32 vk_structure_type_command_pool_create_info) ;
  setf pool_info cmd_pool_create_pNext null ;
  setf pool_info cmd_pool_create_flags (Unsigned.UInt32.of_int 0x02) ;
  (* VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT *)
  setf
    pool_info
    cmd_pool_create_queueFamilyIndex
    (Unsigned.UInt32.of_int device.Device.queue_family) ;

  let pool = allocate vk_command_pool vk_null_handle in
  check
    "vkCreateCommandPool"
    (vkCreateCommandPool device.Device.device (addr pool_info) null pool) ;

  (* Allocate command buffer *)
  let alloc_info = make vk_command_buffer_allocate_info in
  setf
    alloc_info
    cmd_buf_alloc_sType
    (u32 vk_structure_type_command_buffer_allocate_info) ;
  setf alloc_info cmd_buf_alloc_pNext null ;
  setf alloc_info cmd_buf_alloc_commandPool !@pool ;
  setf alloc_info cmd_buf_alloc_level (u32 vk_command_buffer_level_primary) ;
  setf alloc_info cmd_buf_alloc_commandBufferCount (Unsigned.UInt32.of_int 1) ;

  let cmd_buf =
    allocate vk_command_buffer_ptr (from_voidp vk_command_buffer null)
  in
  check
    "vkAllocateCommandBuffers"
    (vkAllocateCommandBuffers device.Device.device (addr alloc_info) cmd_buf) ;

  let fence = create_tracked_fence device.Device.device in
  {command_pool = !@pool; command_buffer = !@cmd_buf; fence; device}

let default_streams : (int, t) Hashtbl.t = Hashtbl.create 4

(** The [default_streams] entry is removed last, and only when it is [stream]:
    if the wait for pending work raises, a default stream stays reachable and
    intact. *)
let destroy stream =
  destroy_tracked_fence
    "vkWaitForFences (Stream.destroy)"
    stream.device.Device.device
    stream.fence ;
  vkDestroyCommandPool stream.device.Device.device stream.command_pool null ;
  match Hashtbl.find_opt default_streams stream.device.Device.id with
  | Some default when default == stream ->
      Hashtbl.remove default_streams stream.device.Device.id
  | Some _ | None -> ()

let synchronize stream =
  await_fence
    "vkWaitForFences (Stream.synchronize)"
    stream.device.Device.device
    stream.fence

let default device =
  match Hashtbl.find_opt default_streams device.Device.id with
  | Some s -> s
  | None ->
      let s = create device in
      Hashtbl.add default_streams device.Device.id s ;
      s
