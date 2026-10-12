(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Events and stream waits on Vulkan (specs/vulkan-async-api.md), on a real
 * device.
 *
 * - [event_query] answers false while the recorded work is pending (a long
 *   chain of dependent launches precedes the record), true once it has
 *   completed, and false rather than block while the event's lock is held.
 * - After [stream_wait_event w e], synchronising [w] returns only once the work
 *   [e] covers has completed (core's [Transfer.synchronize_stream] releases
 *   the waited-for work transitively and relies on this).
 *
 * Skips when no Vulkan device is present.
 ******************************************************************************)

open Sarek_vulkan
open Vulkan_async_support
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Stream = Vulkan_api_stream
module Event = Vulkan_api_event
module Async = Vulkan_plugin_base.Vulkan.Async

(** A chain long enough that its last launch is still queued when the next
    statement runs. *)
let long_chain = 3000

let launch_chain device =
  let kernel = incr_kernel device "event_wait_incr" in
  let buf = upload device [|0.0|] in
  for _ = 1 to long_chain do
    launch_incr kernel buf
  done ;
  buf

let test_query_tracks_pending_work device =
  let buf = launch_chain device in
  let e = Event.create_with_device device in
  Event.record e (Stream.default device) ;
  Alcotest.(check bool)
    "pending right after recording behind a long chain"
    false
    (Event.query e) ;
  Event.synchronize e ;
  Alcotest.(check bool) "complete after synchronize" true (Event.query e) ;
  Event.destroy e ;
  Memory.free buf

let test_query_does_not_block device =
  let e = Event.create_with_device device in
  Mutex.lock e.Event.lock ;
  let answer =
    Fun.protect
      ~finally:(fun () -> Mutex.unlock e.Event.lock)
      (fun () -> Event.query e)
  in
  Alcotest.(check bool) "false while the event's lock is held" false answer ;
  Event.destroy e

let test_wait_then_synchronize device =
  let waiter = Stream.create device in
  for trial = 1 to 5 do
    let buf = launch_chain device in
    let e = Event.create_with_device device in
    Event.record e (Stream.default device) ;
    Async.stream_wait_event waiter e ;
    Stream.synchronize waiter ;
    Alcotest.(check bool)
      (Printf.sprintf
         "trial %d: the event's work is complete after the waiting stream \
          synchronises"
         trial)
      true
      (Event.query e) ;
    Event.synchronize e ;
    Event.destroy e ;
    Memory.free buf
  done ;
  Stream.destroy waiter

let awaited_count stream = List.length stream.Vulkan_api_slot.awaited

(* Completed events are dropped as new ones are added. *)
let test_awaited_stays_bounded device =
  let waiter = Stream.create device in
  for _ = 1 to 1000 do
    let e = Event.create_with_device device in
    Event.record e (Stream.default device) ;
    Event.synchronize e ;
    Event.wait_on e waiter ;
    Event.destroy e
  done ;
  Alcotest.(check bool)
    (Printf.sprintf
       "1000 completed events leave %d awaited"
       (awaited_count waiter))
    true
    (awaited_count waiter <= 1) ;
  Stream.destroy waiter

(* Events whose work finished while nobody looked are dropped by the next add. *)
let test_awaited_pruned_after_idle device =
  let waiter = Stream.create device in
  let buf = launch_chain device in
  let events =
    List.init 50 (fun _ ->
        let e = Event.create_with_device device in
        Event.record e (Stream.default device) ;
        Event.wait_on e waiter ;
        e)
  in
  Device.synchronize device ;
  let probe = Event.create_with_device device in
  Event.wait_on probe waiter ;
  Alcotest.(check bool)
    (Printf.sprintf
       "%d awaited after the device went idle"
       (awaited_count waiter))
    true
    (awaited_count waiter <= 2) ;
  List.iter Event.destroy (probe :: events) ;
  Stream.destroy waiter ;
  Memory.free buf

(* A wait that fails (here an injected device error) must not forget the event:
   the retried synchronize still waits for it. *)
let test_failed_wait_is_retried device =
  let waiter = Stream.create device in
  let buf = launch_chain device in
  let e = Event.create_with_device device in
  Event.record e (Stream.default device) ;
  Event.wait_on e waiter ;
  Vulkan_api_base.inject_wait_failure := Some Vulkan_types.VK_ERROR_DEVICE_LOST ;
  Fun.protect
    ~finally:(fun () -> Vulkan_api_base.inject_wait_failure := None)
    (fun () ->
      match Stream.synchronize waiter with
      | () -> Alcotest.fail "the injected wait failure did not surface"
      | exception _ -> ()) ;
  Alcotest.(check int) "the event is still awaited" 1 (awaited_count waiter) ;
  Stream.synchronize waiter ;
  Alcotest.(check bool)
    "the retried synchronize waited for it"
    true
    (Event.query e) ;
  Alcotest.(check int) "and forgot it afterwards" 0 (awaited_count waiter) ;
  Event.destroy e ;
  Stream.destroy waiter ;
  Memory.free buf

let tests =
  [
    Alcotest.test_case
      "event_query tracks pending work"
      `Quick
      (with_device test_query_tracks_pending_work);
    Alcotest.test_case
      "event_query does not block"
      `Quick
      (with_device test_query_does_not_block);
    Alcotest.test_case
      "synchronising a waiting stream completes the awaited event"
      `Quick
      (with_device test_wait_then_synchronize);
    Alcotest.test_case
      "awaited events stay bounded"
      `Quick
      (with_device test_awaited_stays_bounded);
    Alcotest.test_case
      "awaited events are pruned once the device is idle"
      `Quick
      (with_device test_awaited_pruned_after_idle);
    Alcotest.test_case
      "a failed wait on an awaited event is retried"
      `Quick
      (with_device test_failed_wait_is_retried);
  ]

let () = Alcotest.run "vulkan_event_wait" [("events", tests)]
