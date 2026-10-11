(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Global memory barriers recorded into command buffers. Vulkan orders the
    start of commands in a queue, not their memory: every producer/consumer pair
    below needs one. The first scope of a barrier includes every command
    submitted to the queue before it, so a barrier at the head of a command
    buffer also orders it after earlier submissions. *)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base

let record command_buffer ~src_stage ~dst_stage ~src_access ~dst_access =
  let barrier = make vk_memory_barrier in
  setf barrier mem_barrier_sType (u32 vk_structure_type_memory_barrier) ;
  setf barrier mem_barrier_pNext null ;
  setf barrier mem_barrier_srcAccessMask (u32 src_access) ;
  setf barrier mem_barrier_dstAccessMask (u32 dst_access) ;
  vkCmdPipelineBarrier
    command_buffer
    (u32 src_stage)
    (u32 dst_stage)
    (u32 0)
    (u32 1)
    (to_voidp (addr barrier))
    (u32 0)
    (from_voidp vk_buffer_memory_barrier null)
    (u32 0)
    null ;
  ignore (Sys.opaque_identity barrier)

let compute = vk_pipeline_stage_compute_shader_bit

let transfer = vk_pipeline_stage_transfer_bit

let shader_rw = vk_access_shader_read_bit lor vk_access_shader_write_bit

let writes = vk_access_shader_write_bit lor vk_access_transfer_write_bit

(** Before a dispatch: the writes of earlier dispatches and copies are visible
    to it. *)
let before_dispatch command_buffer =
  record
    command_buffer
    ~src_stage:(compute lor transfer)
    ~dst_stage:compute
    ~src_access:writes
    ~dst_access:shader_rw

(** End of a launch batch: dispatch writes are visible to later copies and to
    the host. *)
let end_of_batch command_buffer =
  record
    command_buffer
    ~src_stage:compute
    ~dst_stage:(transfer lor vk_pipeline_stage_host_bit)
    ~src_access:vk_access_shader_write_bit
    ~dst_access:(vk_access_transfer_read_bit lor vk_access_host_read_bit)

(** Head of a staged copy: earlier dispatches and copies are visible to it. *)
let before_copy command_buffer =
  record
    command_buffer
    ~src_stage:(compute lor transfer)
    ~dst_stage:transfer
    ~src_access:writes
    ~dst_access:(vk_access_transfer_read_bit lor vk_access_transfer_write_bit)

(** Tail of a staged copy: its writes are visible to the host. *)
let after_copy command_buffer =
  record
    command_buffer
    ~src_stage:transfer
    ~dst_stage:vk_pipeline_stage_host_bit
    ~src_access:vk_access_transfer_write_bit
    ~dst_access:vk_access_host_read_bit
