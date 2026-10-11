(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base
module Device = Vulkan_api_device
module Stream = Vulkan_api_stream

(** An event is a fence. Invariant: the fence is either signalled or pending in
    the device's compute queue, never unsignalled with nothing submitted, so
    every wait on it terminates. A new event is signalled: synchronizing an
    event that was never recorded returns at once. *)
type t = {fence : vk_fence; device : Device.t}

let create_with_device device =
  let fence_info = make vk_fence_create_info in
  setf fence_info fence_create_sType (u32 vk_structure_type_fence_create_info) ;
  setf fence_info fence_create_pNext null ;
  setf fence_info fence_create_flags (u32 vk_fence_create_signaled_bit) ;

  let fence = allocate vk_fence vk_null_handle in
  check
    "vkCreateFence"
    (vkCreateFence device.Device.device (addr fence_info) null fence) ;
  {fence = !@fence; device}

let destroy event = vkDestroyFence event.device.Device.device event.fence null

(** Signal [event] once all work submitted to the queue of [stream] before this
    call has completed: an empty submission carrying the event's fence. A
    previous record may still be pending; it is waited for first, since a
    pending fence cannot be reset. *)
let record event (stream : Stream.t) =
  if stream.Stream.device.Device.id <> event.device.Device.id then
    Vulkan_error.raise_error
      (Vulkan_error.context_error
         "Event.record"
         (Printf.sprintf
            "event belongs to device %d, stream to device %d"
            event.device.Device.id
            stream.Stream.device.Device.id)) ;
  let vk_device = event.device.Device.device in
  wait_fence "vkWaitForFences (Event.record)" vk_device event.fence ;
  reset_fence "vkResetFences (Event.record)" vk_device event.fence ;
  check
    "vkQueueSubmit (Event.record)"
    (vkQueueSubmit
       event.device.Device.compute_queue
       (u32 0)
       (from_voidp vk_submit_info null)
       event.fence)

let synchronize event =
  wait_fence
    "vkWaitForFences (Event.synchronize)"
    event.device.Device.device
    event.fence

(** Elapsed time needs timestamp queries, which this backend does not issue;
    answering a number here would be a fabricated measurement. *)
let elapsed ~start:_ ~stop:_ =
  Vulkan_error.raise_error
    (Vulkan_error.feature_not_supported
       "event elapsed time (timestamp queries are not implemented)")
