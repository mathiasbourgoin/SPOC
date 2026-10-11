(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ctypes
open Vulkan_types
open Vulkan_api_base
module Device = Vulkan_api_device
module Stream = Vulkan_api_stream

(** An event is a tracked fence: pending from a successful {!record} until a
    wait observes it signalled. An event that was never recorded, or whose
    record failed, is not pending, so synchronizing it returns at once. [lock]
    serializes the wait/reset/submit sequences on the fence across domains. *)
type t = {fence : tracked_fence; device : Device.t; lock : Mutex.t}

let create_with_device device =
  {
    fence = create_tracked_fence device.Device.device;
    device;
    lock = Mutex.create ();
  }

let destroy event =
  Mutex.protect event.lock (fun () ->
      destroy_tracked_fence
        "vkWaitForFences (Event.destroy)"
        event.device.Device.device
        event.fence)

(** Signal [event] once all work launched on the device of [stream] before this
    call has completed: the open launch batches are submitted, then an empty
    submission carrying the event's fence follows them in the queue. A previous
    record still pending is waited for first, since a pending fence cannot be
    reset. *)
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
  (* The event covers the launches recorded before it: submit them (no wait). *)
  Stream.flush_device event.device ;
  Mutex.protect event.lock (fun () ->
      await_fence "vkWaitForFences (Event.record)" vk_device event.fence ;
      submit_with_fence
        "vkQueueSubmit (Event.record)"
        vk_device
        event.fence
        (Device.queue_submit
           event.device
           (u32 0)
           (from_voidp vk_submit_info null)))

let synchronize event =
  Mutex.protect event.lock (fun () ->
      await_fence
        "vkWaitForFences (Event.synchronize)"
        event.device.Device.device
        event.fence)

(** Elapsed time needs timestamp queries, which this backend does not issue;
    answering a number here would be a fabricated measurement. *)
let elapsed ~start:_ ~stop:_ =
  Vulkan_error.raise_error
    (Vulkan_error.feature_not_supported
       "event elapsed time (timestamp queries are not implemented)")
