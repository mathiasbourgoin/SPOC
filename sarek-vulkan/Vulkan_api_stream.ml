(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Streams (specs/vulkan-async-launches.md).

    A launch is appended to the stream's open command buffer and returns
    ({!Vulkan_api_batch.record_dispatch}). The buffer is submitted, not waited
    for, when it holds {!batch_limit} dispatches, when another stream of the
    device launches, and at any synchronisation point. The host blocks at
    {!synchronize}, at {!Vulkan_api_device.drain_pending} (host access to device
    memory, device wait, kernel eviction, device destroy) and when a command
    buffer is reused. A stream has two command buffers, so one can be recorded
    while the other runs.

    All streams of a device submit to one queue, in launch order, because a
    launch first submits the open batches of the device's other streams;
    dispatches are ordered in memory by barriers ({!Vulkan_api_barrier}). *)

open Vulkan_api_slot
module Device = Vulkan_api_device

type t = stream

let device (stream : t) = stream.device

let batch_limit = Vulkan_api_batch.batch_limit

let max_deferred_bytes = Vulkan_api_batch.max_deferred_bytes

let record_dispatch = Vulkan_api_batch.record_dispatch

let flush_device = Vulkan_api_batch.flush_device

let synchronize = Vulkan_api_batch.synchronize

let pending_launches = Vulkan_api_batch.pending_launches

let create device =
  let stream = create_stream device in
  register stream ;
  stream

let forget_default stream =
  Mutex.protect default_lock (fun () ->
      match Hashtbl.find_opt default_streams stream.device.Device.id with
      | Some default when default == stream ->
          Hashtbl.remove default_streams stream.device.Device.id
      | Some _ | None -> ())

(** Nothing is unregistered if the wait for pending work raises: the stream
    stays reachable and intact. *)
let destroy stream =
  synchronize stream ;
  if Atomic.compare_and_set stream.destroyed false true then begin
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
              if Atomic.compare_and_set stream.destroyed false true then
                destroy_objects stream
            with e -> if !first_error = None then first_error := Some e)
          streams ;
        Option.iter raise !first_error
      end)
