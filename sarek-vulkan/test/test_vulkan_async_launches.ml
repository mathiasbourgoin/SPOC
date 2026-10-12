(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Asynchronous, batched launches on Vulkan (specs/vulkan-async-launches.md),
 * on a real device.
 *
 * - N back-to-back launches cost far fewer than N fence waits and queue
 *   submissions (counted by [Vulkan_api_base.fence_wait_count] and
 *   [submission_count]): the deterministic form of "launches do not block".
 * - A dependent chain (launch k reads what launch k-1 wrote, across several
 *   batches) gives the exact result: dispatches inside and across command
 *   buffers are ordered in memory, not only in submission.
 * - Host reads after a partial batch, a buffer freed right after its launches,
 *   and two streams handing a buffer to each other see completed work.
 * - A failed submission surfaces as [Kernel_launch_failed] naming the kernel at
 *   the first wait, not at the launch, and leaves the stream usable.
 *
 * Skips when no Vulkan device is present.
 ******************************************************************************)

open Sarek_vulkan
open Vulkan_async_support
module Base = Vulkan_api_base
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Stream = Vulkan_api_stream
module Kernel = Vulkan_api_kernel
module BE = Sarek_backend_error.Backend_error

let test_launches_do_not_wait_each device =
  let kernel = incr_kernel device "async_incr" in
  let buf = upload device [|0.0|] in
  let launches = 100 in
  let waits0 = Atomic.get Base.fence_wait_count in
  let subs0 = Atomic.get Base.submission_count in
  for _ = 1 to launches do
    launch_incr kernel buf
  done ;
  Stream.synchronize (Stream.default device) ;
  let waits = Atomic.get Base.fence_wait_count - waits0 in
  let subs = Atomic.get Base.submission_count - subs0 in
  Alcotest.(check (float 0.0)) "all launches ran" 100.0 (download buf).(0) ;
  Memory.free buf ;
  (* Default batch of 32 dispatches: 4 submissions; the waits are the final
     synchronize plus at most one per reused command buffer. *)
  Alcotest.(check bool)
    (Printf.sprintf "%d launches made %d submissions (<= 8)" launches subs)
    true
    (subs <= 8) ;
  Alcotest.(check bool)
    (Printf.sprintf "%d launches made %d fence waits (<= 8)" launches waits)
    true
    (waits <= 8)

let test_dependent_chain device =
  check_all_equal
    "100-step dependent chain across batches"
    100.0
    (run_chain device 100)

let test_host_read_mid_batch device =
  let kernel = step_kernel device in
  let a = upload device zeros in
  let b = upload device zeros in
  for k = 0 to 4 do
    launch_step kernel (if k mod 2 = 0 then [a; b] else [b; a])
  done ;
  check_all_equal "read after 5 launches (partial batch)" 5.0 (download b) ;
  for k = 5 to 9 do
    launch_step kernel (if k mod 2 = 0 then [a; b] else [b; a])
  done ;
  check_all_equal "read after 10 launches" 10.0 (download a) ;
  Memory.free a ;
  Memory.free b

let test_free_after_launch device =
  let kernel = step_kernel device in
  for _ = 1 to 5 do
    let a = upload device zeros in
    let b = upload device zeros in
    for k = 0 to 9 do
      launch_step kernel (if k mod 2 = 0 then [a; b] else [b; a])
    done ;
    Memory.free a ;
    Memory.free b
  done ;
  check_all_equal "buffers after free-after-launch" 3.0 (run_chain device 3)

let test_two_streams device =
  let s1 = Stream.create device in
  let s2 = Stream.create device in
  let result =
    run_chain device 60 ~stream_of:(fun k -> if k mod 2 = 0 then s1 else s2)
  in
  check_all_equal "buffer handed between two streams" 60.0 result ;
  Stream.destroy s1 ;
  Stream.destroy s2

let test_failure_at_first_wait device =
  let kernel = incr_kernel device "async_fail_probe" in
  let buf = upload device [|0.0|] in
  Device.inject_submit_failure := Some Vulkan_types.VK_ERROR_DEVICE_LOST ;
  (match launch_incr kernel buf with
  | () -> ()
  | exception BE.Backend_error e ->
      Device.inject_submit_failure := None ;
      Alcotest.failf
        "the launch itself raised (%s); failures belong to the first wait"
        (BE.to_string e)) ;
  let outcome =
    match Stream.synchronize (Stream.default device) with
    | () -> `Returned
    | exception
        BE.Backend_error
          (BE.Runtime {error = BE.Kernel_launch_failed {kernel_name; _}; _}) ->
        `Named kernel_name
    | exception BE.Backend_error _ -> `Other
  in
  Device.inject_submit_failure := None ;
  (match outcome with
  | `Named name ->
      Alcotest.(check bool)
        ("the error names the kernel, got: " ^ name)
        true
        (contains name "async_fail_probe")
  | `Returned -> Alcotest.fail "synchronize did not raise"
  | `Other -> Alcotest.fail "synchronize raised something else") ;
  launch_incr kernel buf ;
  Alcotest.(check (float 0.0))
    "the stream is usable after the failure"
    1.0
    (download buf).(0) ;
  Memory.free buf

let test_batch_of_one_waits_each device =
  with_batch_limit 1 (fun () ->
      let kernel = incr_kernel device "async_incr" in
      let buf = upload device [|0.0|] in
      let waits0 = Atomic.get Base.fence_wait_count in
      let subs0 = Atomic.get Base.submission_count in
      for _ = 1 to 10 do
        launch_incr kernel buf
      done ;
      Alcotest.(check int)
        "SPOC_VULKAN_BATCH=1: one submission per launch"
        10
        (Atomic.get Base.submission_count - subs0) ;
      Alcotest.(check bool)
        "SPOC_VULKAN_BATCH=1: every launch is waited for"
        true
        (Atomic.get Base.fence_wait_count - waits0 >= 10) ;
      Alcotest.(check int)
        "nothing pending after a batch of one"
        0
        (Stream.pending_launches device) ;
      Alcotest.(check (float 0.0)) "result" 10.0 (download buf).(0) ;
      Memory.free buf)

let test_batch_of_one_raises_at_launch device =
  with_batch_limit 1 (fun () ->
      let kernel = incr_kernel device "async_fail_probe_1" in
      let buf = upload device [|0.0|] in
      Device.inject_submit_failure := Some Vulkan_types.VK_ERROR_DEVICE_LOST ;
      let raised =
        match launch_incr kernel buf with
        | () -> false
        | exception BE.Backend_error _ -> true
      in
      Device.inject_submit_failure := None ;
      Alcotest.(check bool)
        "SPOC_VULKAN_BATCH=1 raises at the launch"
        true
        raised ;
      launch_incr kernel buf ;
      Alcotest.(check (float 0.0)) "usable afterwards" 1.0 (download buf).(0) ;
      Memory.free buf)

let test_sync_points_complete_pending device =
  let kernel = incr_kernel device "async_incr" in
  let buf = upload device [|0.0|] in
  let pending_after what sync =
    for _ = 1 to 5 do
      launch_incr kernel buf
    done ;
    Alcotest.(check int)
      (what ^ ": launches are pending before")
      5
      (Stream.pending_launches device) ;
    sync () ;
    Alcotest.(check int)
      (what ^ ": nothing pending after")
      0
      (Stream.pending_launches device)
  in
  pending_after "Stream.synchronize" (fun () ->
      Stream.synchronize (Stream.default device)) ;
  pending_after "Device.synchronize" (fun () -> Device.synchronize device) ;
  pending_after "Memory.device_to_host" (fun () -> ignore (download buf)) ;
  pending_after "Memory.host_to_device" (fun () ->
      let host = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 1 in
      host.{0} <- 100.0 ;
      Memory.host_to_device ~src:host ~dst:buf) ;
  Alcotest.(check (float 0.0))
    "the upload came after the launches"
    100.0
    (download buf).(0) ;
  Memory.free buf

let test_event_after_batch device =
  let kernel = incr_kernel device "async_incr" in
  let buf = upload device [|0.0|] in
  for _ = 1 to 7 do
    launch_incr kernel buf
  done ;
  let event = Vulkan_api_event.create_with_device device in
  Vulkan_api_event.record event (Stream.default device) ;
  Vulkan_api_event.synchronize event ;
  Vulkan_api_event.destroy event ;
  Alcotest.(check (float 0.0))
    "launches before the event ran"
    7.0
    (download buf).(0) ;
  Memory.free buf

let test_many_sets_one_batch device =
  with_batch_limit 200 (fun () ->
      let kernel = incr_kernel device "async_incr" in
      let buf = upload device [|0.0|] in
      for _ = 1 to 150 do
        launch_incr kernel buf
      done ;
      Alcotest.(check (float 0.0))
        "150 descriptor sets in one batch (more than one pool)"
        150.0
        (download buf).(0) ;
      Memory.free buf)

(** Runs last: it destroys the device. Pending launches reach it before it goes,
    and the next device of the same index gets fresh streams. *)
let test_destroy_with_pending device =
  let kernel = incr_kernel device "async_incr" in
  let buf = upload device [|0.0|] in
  for _ = 1 to 5 do
    launch_incr kernel buf
  done ;
  Device.destroy device ;
  let fresh = Device.get 0 in
  let kernel = incr_kernel fresh "async_incr" in
  let buf = upload fresh [|0.0|] in
  launch_incr kernel buf ;
  Alcotest.(check (float 0.0))
    "a new device of the same index launches"
    1.0
    (download buf).(0) ;
  Memory.free buf

let () =
  Alcotest.run
    "Vulkan_async_launches"
    [
      ( "batching",
        [
          Alcotest.test_case
            "N launches do not make N waits"
            `Quick
            (with_batching_device test_launches_do_not_wait_each);
          Alcotest.test_case
            "SPOC_VULKAN_BATCH=1 waits at every launch"
            `Quick
            (with_batching_device test_batch_of_one_waits_each);
          Alcotest.test_case
            "descriptor sets beyond one pool in one batch"
            `Quick
            (with_batching_device test_many_sets_one_batch);
        ] );
      ( "ordering",
        [
          Alcotest.test_case
            "dependent chain across batches"
            `Quick
            (with_batching_device test_dependent_chain);
          Alcotest.test_case
            "host read in the middle of a batch"
            `Quick
            (with_batching_device test_host_read_mid_batch);
          Alcotest.test_case
            "free right after launches"
            `Quick
            (with_batching_device test_free_after_launch);
          Alcotest.test_case
            "two streams share a buffer"
            `Quick
            (with_batching_device test_two_streams);
          Alcotest.test_case
            "every synchronisation point completes pending launches"
            `Quick
            (with_batching_device test_sync_points_complete_pending);
          Alcotest.test_case
            "an event covers the launches before it"
            `Quick
            (with_batching_device test_event_after_batch);
        ] );
      ( "errors",
        [
          Alcotest.test_case
            "submit failure surfaces at the first wait"
            `Quick
            (with_batching_device test_failure_at_first_wait);
          Alcotest.test_case
            "SPOC_VULKAN_BATCH=1 raises at the launch"
            `Quick
            (with_batching_device test_batch_of_one_raises_at_launch);
        ] );
      ( "teardown",
        [
          Alcotest.test_case
            "device destroy with pending launches"
            `Quick
            (with_batching_device test_destroy_with_pending);
        ] );
    ]
