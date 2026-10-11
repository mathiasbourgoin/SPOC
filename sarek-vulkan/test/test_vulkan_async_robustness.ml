(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Robustness of the asynchronous launches (specs/vulkan-async-launches.md):
 * deferred errors, cross-stream and cross-domain use, buffer release,
 * on a real device.
 *
 * - A submission that fails while launching (batch full, another stream's
 *   launch, a release threshold) surfaces at the first wait naming the kernels.
 * - A second stream consuming an open or full batch, and two domains launching
 *   on one device, see ordered results.
 * - Buffer releases: from GC finalisers (never block or raise), explicit frees
 *   (prompt), after an allocation failure (retry), past the queue bound, and
 *   never before the batch that binds the buffer has completed.
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

(** [Gpu_memory.register_finalizer] frees device buffers from a [Gc.finalise]
    finaliser and ignores its exceptions. With a tiny minor heap the finaliser
    runs at allocation points inside the batch code; a free must neither raise
    there (a swallowed failure leaks the buffer) nor deadlock. *)
let test_free_from_finaliser device =
  let kernel = step_kernel device in
  let src = upload device zeros in
  let baseline = Atomic.get Memory.live_allocations in
  let failures = Atomic.make 0 in
  let saved = Gc.get () in
  Gc.set {saved with Gc.minor_heap_size = 256} ;
  Fun.protect
    ~finally:(fun () -> Gc.set saved)
    (fun () ->
      for _ = 1 to 300 do
        let dst = upload device zeros in
        let holder = ref (Some dst) in
        Gc.finalise
          (fun h ->
            match !h with
            | Some b -> ( try Memory.free b with _ -> Atomic.incr failures)
            | None -> ())
          holder ;
        launch_step kernel [src; dst] ;
        ignore (Sys.opaque_identity holder)
      done) ;
  Gc.full_major () ;
  Device.synchronize device ;
  Gc.full_major () ;
  Device.synchronize device ;
  Alcotest.(check int) "no free raised in a finaliser" 0 (Atomic.get failures) ;
  Alcotest.(check int)
    "every buffer dropped by the GC was released"
    baseline
    (Atomic.get Memory.live_allocations) ;
  Memory.free src

(** The submission fails while launching (batch full), not at a wait. *)
let test_deferred_failure_batch_full device =
  with_batch_limit 8 (fun () ->
      let kernel = incr_kernel device "async_deferred_full" in
      let buf = upload device [|0.0|] in
      Device.inject_submit_failure := Some Vulkan_types.VK_ERROR_DEVICE_LOST ;
      Fun.protect
        ~finally:(fun () -> Device.inject_submit_failure := None)
        (fun () ->
          for _ = 1 to 8 do
            launch_incr kernel buf
          done ;
          expect_named_failure
            "batch-full submit"
            "async_deferred_full"
            (fun () -> Stream.synchronize (Stream.default device))) ;
      launch_incr kernel buf ;
      Alcotest.(check (float 0.0)) "usable afterwards" 1.0 (download buf).(0) ;
      Memory.free buf)

(** The submission fails because a launch on another stream flushed the batch.
*)
let test_deferred_failure_other_stream device =
  let s1 = Stream.create device in
  let s2 = Stream.create device in
  let kernel = incr_kernel device "async_deferred_other" in
  let buf = upload device [|0.0|] in
  let one = Spoc_framework.Framework_sig.dims_1d 1 in
  let launch stream = launch_with ~stream kernel [buf] ~grid:one ~block:one in
  launch s1 ;
  Device.inject_submit_failure := Some Vulkan_types.VK_ERROR_DEVICE_LOST ;
  Fun.protect
    ~finally:(fun () -> Device.inject_submit_failure := None)
    (fun () ->
      launch s2 ;
      expect_named_failure
        "other-stream flush"
        "async_deferred_other"
        (fun () -> Stream.synchronize s1)) ;
  Stream.synchronize s2 ;
  Alcotest.(check (float 0.0))
    "second stream's launch ran"
    1.0
    (download buf).(0) ;
  Memory.free buf ;
  Stream.destroy s1 ;
  Stream.destroy s2

(** Stream 2 consumes what stream 1 produced while stream 1's batch is still
    open ([steps] = 5) or has just been submitted because it was full (8). *)
let test_cross_stream_batch_state device =
  with_batch_limit 8 (fun () ->
      List.iter
        (fun steps ->
          let s1 = Stream.create device in
          let s2 = Stream.create device in
          let kernel = step_kernel device in
          let a = upload device zeros in
          let b = upload device zeros in
          for k = 0 to steps - 1 do
            let src, dst = if k mod 2 = 0 then (a, b) else (b, a) in
            launch_step ~stream:s1 kernel [src; dst]
          done ;
          let last, other = if steps mod 2 = 0 then (a, b) else (b, a) in
          launch_step ~stream:s2 kernel [last; other] ;
          check_all_equal
            (Printf.sprintf "consumer on a second stream after %d steps" steps)
            (float (steps + 1))
            (download other) ;
          Memory.free a ;
          Memory.free b ;
          Stream.destroy s1 ;
          Stream.destroy s2)
        [5; 8])

(** Two domains launch on their own streams of one device; the second of the two
    synchronises while the first launches. Fence waits are bounded by
    SPOC_VULKAN_FENCE_TIMEOUT_S in the dune rule. Buffers are transferred
    outside the domains: the transfer command pool is not domain-safe. *)
let test_two_domains device =
  let steps = 300 in
  let kernel = step_kernel device in
  let make () =
    (Stream.create device, upload device zeros, upload device zeros)
  in
  let worker (stream, a, b) () =
    for k = 0 to steps - 1 do
      let src, dst = if k mod 2 = 0 then (a, b) else (b, a) in
      launch_step ~stream kernel [src; dst]
    done ;
    Stream.synchronize stream
  in
  let w1 = make () and w2 = make () in
  let d1 = Domain.spawn (worker w1) in
  let d2 = Domain.spawn (worker w2) in
  Domain.join d1 ;
  Domain.join d2 ;
  List.iter
    (fun (stream, a, b) ->
      check_all_equal
        "two-domain chain"
        (float steps)
        (download (if steps mod 2 = 0 then a else b)) ;
      Memory.free a ;
      Memory.free b ;
      Stream.destroy stream)
    [w1; w2]

let live () = Atomic.get Memory.live_allocations

(** An explicit free releases at once, without a synchronisation call. *)
let test_explicit_free_is_prompt device =
  let baseline = live () in
  for _ = 1 to 20 do
    let b = upload device zeros in
    Memory.free b ;
    Alcotest.(check int) "released by the free itself" baseline (live ())
  done

(** An allocation that fails while releases are queued completes the launches,
    runs the releases and retries. *)
let test_alloc_retries_after_release device =
  let victim = upload device zeros in
  let released = ref false in
  Device.defer_free device ~bytes:(n * 4) (fun () ->
      released := true ;
      Memory.release_now victim) ;
  Memory.inject_alloc_failure := true ;
  let fresh = Memory.alloc device 16 Bigarray.float32 in
  Alcotest.(check bool) "the queued release ran before the retry" true !released ;
  Memory.free fresh

(** Past 64 queued releases a launch completes all launches and runs them. *)
let test_threshold_release device =
  let kernel = step_kernel device in
  let a = upload device zeros and b = upload device zeros in
  let victims = List.init 64 (fun _ -> upload device [|0.0|]) in
  let before = live () in
  for k = 0 to 4 do
    launch_step kernel (if k mod 2 = 0 then [a; b] else [b; a])
  done ;
  List.iter
    (fun v ->
      Device.defer_free device ~bytes:4 (fun () -> Memory.release_now v))
    victims ;
  Alcotest.(check int) "queued, not yet released" before (live ()) ;
  launch_step kernel [a; b] ;
  Alcotest.(check int)
    "the launch completed the earlier ones, not its own"
    1
    (Stream.pending_launches device) ;
  Alcotest.(check int) "64 releases ran" (before - 64) (live ()) ;
  Device.synchronize device ;
  Memory.free a ;
  Memory.free b

(** A release queued while a batch binds the buffer waits for that batch. *)
let test_release_waits_for_batch device =
  let kernel = step_kernel device in
  let a = upload device zeros and b = upload device zeros in
  let before = live () in
  launch_step kernel [a; b] ;
  Device.defer_free device ~bytes:(n * 4) (fun () -> Memory.release_now b) ;
  Alcotest.(check int) "still live while the batch is open" before (live ()) ;
  Alcotest.(check int) "the batch is pending" 1 (Stream.pending_launches device) ;
  Device.synchronize device ;
  Alcotest.(check int)
    "released once the batch completed"
    (before - 1)
    (live ()) ;
  Memory.free a

(** Another stream's deferred error is not consumed by an unrelated launch that
    triggers the release threshold. *)
let test_threshold_keeps_other_stream_error device =
  let s1 = Stream.create device in
  let kernel = incr_kernel device "async_threshold_error" in
  let buf = upload device [|0.0|] in
  let one = Spoc_framework.Framework_sig.dims_1d 1 in
  let victims = List.init 64 (fun _ -> upload device [|0.0|]) in
  launch_with ~stream:s1 kernel [buf] ~grid:one ~block:one ;
  Device.inject_submit_failure := Some Vulkan_types.VK_ERROR_DEVICE_LOST ;
  Fun.protect
    ~finally:(fun () -> Device.inject_submit_failure := None)
    (fun () ->
      List.iter
        (fun v ->
          Device.defer_free device ~bytes:4 (fun () -> Memory.release_now v))
        victims ;
      (* Triggers the threshold on the default stream: s1's batch is submitted,
         fails, and the launch itself must still be recorded and not raise. *)
      launch_incr kernel buf ;
      expect_named_failure
        "other stream's error"
        "async_threshold_error"
        (fun () -> Stream.synchronize s1)) ;
  Stream.synchronize (Stream.default device) ;
  Alcotest.(check (float 0.0)) "the unrelated launch ran" 1.0 (download buf).(0) ;
  Memory.free buf ;
  Stream.destroy s1

let () =
  Alcotest.run
    "Vulkan_async_robustness"
    [
      ( "deferred errors",
        [
          Alcotest.test_case
            "deferred failure when the batch fills"
            `Quick
            (with_device test_deferred_failure_batch_full);
          Alcotest.test_case
            "deferred failure when another stream launches"
            `Quick
            (with_device test_deferred_failure_other_stream);
          Alcotest.test_case
            "a release threshold keeps another stream's error"
            `Quick
            (with_device test_threshold_keeps_other_stream_error);
        ] );
      ( "streams",
        [
          Alcotest.test_case
            "second stream consumes an open or full batch"
            `Quick
            (with_device test_cross_stream_batch_state);
          Alcotest.test_case
            "two domains on one device"
            `Quick
            (with_device test_two_domains);
        ] );
      ( "buffer release",
        [
          Alcotest.test_case
            "buffers freed from GC finalisers"
            `Quick
            (with_device test_free_from_finaliser);
          Alcotest.test_case
            "an explicit free releases at once"
            `Quick
            (with_device test_explicit_free_is_prompt);
          Alcotest.test_case
            "an allocation retries after queued releases"
            `Quick
            (with_device test_alloc_retries_after_release);
          Alcotest.test_case
            "past 64 queued releases a launch runs them"
            `Quick
            (with_device test_threshold_release);
          Alcotest.test_case
            "a release waits for the batch that binds the buffer"
            `Quick
            (with_device test_release_waits_for_batch);
        ] );
    ]
