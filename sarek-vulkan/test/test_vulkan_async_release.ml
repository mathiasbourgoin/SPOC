(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The buffer-free path of the asynchronous launches
 * (specs/vulkan-async-launches.md), on a real device. A free runs from GC
 * finalisers, at any allocation point: it must never submit, record, wait,
 * block on a lock or consume a launch error, and it releases only when no
 * launch can still use the buffer.
 ******************************************************************************)

open Sarek_vulkan
open Vulkan_async_support
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Stream = Vulkan_api_stream
module BE = Sarek_backend_error.Backend_error

let live () = Atomic.get Memory.live_allocations

(** Run [free] from a real [Gc.finalise] finaliser. *)
let free_from_finaliser b =
  let ran = ref false in
  let holder = ref (Some b) in
  Gc.finalise
    (fun h ->
      ran := true ;
      Option.iter Memory.free !h)
    holder ;
  ignore (Sys.opaque_identity holder) ;
  fun () ->
    Gc.full_major () ;
    !ran

let one = Spoc_framework.Framework_sig.dims_1d 1

(** A finaliser frees a buffer while this domain holds the queue lock and a
    batch is open: nothing is submitted or waited for, and the batch still runs
    and gives the right result. *)
let test_finaliser_under_queue_lock device =
  let kernel = incr_kernel device "async_ql" in
  let buf = upload device [|0.0|] in
  launch_incr kernel buf ;
  let before = live () in
  let collect = free_from_finaliser (Memory.alloc device 16 Bigarray.float32) in
  let ran = Device.with_queue_lock device (fun () -> collect ()) in
  Alcotest.(check bool) "the finaliser ran" true ran ;
  Alcotest.(check int)
    "the free neither submitted nor waited"
    1
    (Stream.pending_launches device) ;
  Alcotest.(check int) "the release stays queued" (before + 1) (live ()) ;
  launch_incr kernel buf ;
  Stream.synchronize (Stream.default device) ;
  Alcotest.(check (float 0.0)) "both launches ran" 2.0 (download buf).(0) ;
  Alcotest.(check int) "released at the idle synchronize" before (live ()) ;
  Memory.free buf

(** A free between a failed submission and the wait must not hide the failure:
    [explicit] frees directly, otherwise from a finaliser. *)
let free_keeps_error ~explicit device =
  let name = if explicit then "async_err_explicit" else "async_err_gc" in
  let kernel = incr_kernel device name in
  let buf = upload device [|0.0|] in
  let s1 = Stream.create device and s2 = Stream.create device in
  launch_with ~stream:s1 kernel [buf] ~grid:one ~block:one ;
  Device.inject_submit_failure := Some Vulkan_types.VK_ERROR_DEVICE_LOST ;
  launch_with ~stream:s2 kernel [buf] ~grid:one ~block:one ;
  Device.inject_submit_failure := None ;
  let victim = Memory.alloc device 16 Bigarray.float32 in
  if explicit then Memory.free victim
  else ignore ((free_from_finaliser victim) ()) ;
  expect_named_failure "error kept across a free" name (fun () ->
      Stream.synchronize s1) ;
  Stream.synchronize s2 ;
  Alcotest.(check (float 0.0))
    "only the second stream's kernel ran"
    1.0
    (download buf).(0) ;
  Memory.free buf ;
  Stream.destroy s1 ;
  Stream.destroy s2

(** The byte bound: one large queued release makes the next launch run the queue
    even though the release count is far below its bound. *)
let test_byte_threshold device =
  let saved = !Stream.max_deferred_bytes in
  Stream.max_deferred_bytes := 1000 ;
  Fun.protect
    ~finally:(fun () -> Stream.max_deferred_bytes := saved)
    (fun () ->
      let kernel = incr_kernel device "async_bytes" in
      let buf = upload device [|0.0|] in
      let victim = upload device zeros in
      let before = live () in
      Device.defer_free device ~bytes:2000 (fun () -> Memory.release_now victim) ;
      Alcotest.(check int) "queued" before (live ()) ;
      launch_incr kernel buf ;
      Alcotest.(check int) "released by the byte bound" (before - 1) (live ()) ;
      Alcotest.(check int)
        "the launch itself is recorded"
        1
        (Stream.pending_launches device) ;
      Stream.synchronize (Stream.default device) ;
      Memory.free buf)

(** A stream synchronisation that leaves the device idle releases the queue. *)
let test_idle_synchronize_releases device =
  let kernel = incr_kernel device "async_idle" in
  let buf = upload device [|0.0|] in
  let victim = upload device zeros in
  let before = live () in
  launch_incr kernel buf ;
  Device.defer_free device ~bytes:(n * 4) (fun () -> Memory.release_now victim) ;
  Alcotest.(check int) "queued behind the open batch" before (live ()) ;
  Stream.synchronize (Stream.default device) ;
  Alcotest.(check int) "released once idle" (before - 1) (live ()) ;
  Memory.free buf

(** An allocation failure with releases queued retries once; a retry that fails
    again re-raises the original error. *)
let test_alloc_retry_reraises device =
  let victim = upload device zeros in
  let ran = ref false in
  Device.defer_free device ~bytes:(n * 4) (fun () ->
      ran := true ;
      Memory.release_now victim) ;
  Memory.inject_alloc_failure := 2 ;
  let outcome =
    match Memory.alloc device 16 Bigarray.float32 with
    | b ->
        Memory.free b ;
        `Allocated
    | exception BE.Backend_error (BE.Runtime {error = BE.Context_error _; _}) ->
        `Original_error
  in
  Memory.inject_alloc_failure := 0 ;
  Alcotest.(check bool) "queued release ran before the retry" true !ran ;
  Alcotest.(check bool)
    "original error re-raised"
    true
    (outcome = `Original_error)

let wait_queue_idle device =
  Device.with_queue_lock device (fun () ->
      ignore (Vulkan_bindings.vkQueueWaitIdle device.Device.compute_queue))

(** A slot whose fence has signalled but that nobody waited for is retired by
    the free path, which then releases at once. *)
let test_signalled_slot_is_reaped device =
  let kernel = incr_kernel device "async_reap" in
  let buf = upload device [|0.0|] in
  launch_incr kernel buf ;
  Stream.flush_device device ;
  wait_queue_idle device ;
  let victim = Memory.alloc device 16 Bigarray.float32 in
  let before = live () in
  Memory.free victim ;
  Alcotest.(check int) "released by the free itself" (before - 1) (live ()) ;
  Alcotest.(check int) "the slot was retired" 0 (Stream.pending_launches device) ;
  Alcotest.(check (float 0.0)) "the launch ran" 1.0 (download buf).(0) ;
  Memory.free buf

(** A wait that fails because the device is lost raises once and leaves the
    stream usable. *)
let test_lost_wait_raises_once device =
  let kernel = incr_kernel device "async_lost_wait" in
  let buf = upload device [|0.0|] in
  launch_incr kernel buf ;
  Stream.flush_device device ;
  wait_queue_idle device ;
  Vulkan_api_base.inject_wait_failure := Some Vulkan_types.VK_ERROR_DEVICE_LOST ;
  Fun.protect
    ~finally:(fun () -> Vulkan_api_base.inject_wait_failure := None)
    (fun () ->
      expect_named_failure "failed wait" "async_lost_wait" (fun () ->
          Stream.synchronize (Stream.default device)) ;
      (* The device keeps failing; the abandoned slot is not waited for. *)
      Stream.synchronize (Stream.default device)) ;
  launch_incr kernel buf ;
  Stream.synchronize (Stream.default device) ;
  Alcotest.(check (float 0.0)) "both launches ran" 2.0 (download buf).(0) ;
  Memory.free buf

(** A wait that fails for another reason does not end the slot's work: it stays
    in flight and the next wait retires it. *)
let test_other_wait_error_keeps_slot device =
  let kernel = incr_kernel device "async_oom_wait" in
  let buf = upload device [|0.0|] in
  launch_incr kernel buf ;
  Stream.flush_device device ;
  wait_queue_idle device ;
  Vulkan_api_base.inject_wait_failure :=
    Some Vulkan_types.VK_ERROR_OUT_OF_DEVICE_MEMORY ;
  Fun.protect
    ~finally:(fun () -> Vulkan_api_base.inject_wait_failure := None)
    (fun () ->
      expect_named_failure "failed wait" "async_oom_wait" (fun () ->
          Stream.synchronize (Stream.default device))) ;
  Alcotest.(check int)
    "the slot is still in flight"
    1
    (Stream.pending_launches device) ;
  Stream.synchronize (Stream.default device) ;
  Alcotest.(check int)
    "retired by the next wait"
    0
    (Stream.pending_launches device) ;
  Memory.free buf

(** A status query that reports a lost device, seen by the free path: the error
    is kept once as the stream's deferred error, the slot is abandoned, and the
    next wait raises it exactly once. *)
let test_reap_lost_device device =
  let kernel = incr_kernel device "async_reap_lost" in
  let buf = upload device [|0.0|] in
  launch_incr kernel buf ;
  Stream.flush_device device ;
  wait_queue_idle device ;
  let victim = Memory.alloc device 16 Bigarray.float32 in
  Vulkan_api_base.inject_status_failure :=
    Some Vulkan_types.VK_ERROR_DEVICE_LOST ;
  Fun.protect
    ~finally:(fun () -> Vulkan_api_base.inject_status_failure := None)
    (fun () -> Memory.free victim) ;
  Alcotest.(check int)
    "the slot was abandoned by the free path"
    0
    (Stream.pending_launches device) ;
  expect_named_failure "kept status error" "async_reap_lost" (fun () ->
      Stream.synchronize (Stream.default device)) ;
  Stream.synchronize (Stream.default device) ;
  Memory.free buf

(** Sweep an asynchronous exception ([Exit], raised from a memprof callback at
    the k-th allocation of [Stream.synchronize] and at the next [burst]
    allocations) over every point of a submission of [batch] launches. Whatever
    the point: only [Exit] reaches the caller; no slot is retired while the
    queue still runs its batch (fence not signalled right away, signalled after
    a queue wait); the batch ran entirely or not at all; and a batch the
    exception left open (it came before the command buffer was ended) is not
    lost. The sweep must see both kinds of round. *)
let async_exception_sweep ~burst device =
  let batch = 20 in
  let kernel = incr_kernel device "async_memprof" in
  let buf = upload device [|0.0|] in
  let stream = Stream.default device in
  let slots () = stream.Vulkan_api_slot.slots in
  let status (sl : Vulkan_api_slot.slot) =
    Vulkan_api_base.fence_signalled "test" device.Device.device sl.fence.handle
  in
  let previous = ref 0.0 in
  let begins0 = Atomic.get Vulkan_api_batch.begin_on_open_buffer in
  let k = ref 0 and fired = ref true in
  let kept_open = ref 0 and lost_or_ran = ref 0 in
  while !fired && !k < 800 do
    incr k ;
    for _ = 1 to batch do
      launch_incr kernel buf
    done ;
    let count = ref 0 in
    let ended_at_fire = ref true in
    fired := false ;
    let callback _ =
      incr count ;
      if !count >= !k && !count <= !k + burst then begin
        if not !fired then
          ended_at_fire :=
            (slots ()).(stream.Vulkan_api_slot.active).Vulkan_api_slot.ended ;
        fired := true ;
        raise Exit
      end ;
      None
    in
    let tracker =
      {
        Gc.Memprof.null_tracker with
        alloc_minor = callback;
        alloc_major = callback;
      }
    in
    ignore (Gc.Memprof.start ~sampling_rate:1.0 ~callstack_size:0 tracker) ;
    let outcome =
      match Stream.synchronize stream with
      | () -> `Ok
      | exception Exit -> `Exit
      | exception e -> `Other (Printexc.to_string e)
    in
    Gc.Memprof.stop () ;
    (match outcome with
    | `Other what ->
        Alcotest.failf "k=%d: an exception other than Exit escaped: %s" !k what
    | `Ok | `Exit -> ()) ;
    let before =
      Array.map
        (fun sl -> (sl.Vulkan_api_slot.fence.pending, status sl))
        (slots ())
    in
    let still_open =
      Array.exists
        (fun (sl : Vulkan_api_slot.slot) -> sl.recording && sl.dispatches > 0)
        (slots ())
    in
    wait_queue_idle device ;
    Array.iteri
      (fun i (pending, signalled_before) ->
        if (not pending) && (not signalled_before) && status (slots ()).(i) then
          Alcotest.failf
            "k=%d burst=%d: a slot was retired while the queue ran its batch"
            !k
            burst)
      before ;
    Stream.synchronize stream ;
    Alcotest.(check int) "nothing pending" 0 (Stream.pending_launches device) ;
    let value = (download buf).(0) in
    let fb = float batch in
    Alcotest.(check bool)
      (Printf.sprintf
         "k=%d: all or none of the batch ran (%.0f -> %.0f)"
         !k
         !previous
         value)
      true
      (value = !previous || value = !previous +. fb) ;
    if outcome = `Ok then
      Alcotest.(check (float 0.0))
        "an unfired round ran"
        (!previous +. fb)
        value ;
    if outcome = `Exit && not !ended_at_fire then
      Alcotest.(check (float 0.0))
        (Printf.sprintf
           "k=%d: an exception before the command buffer ended loses nothing"
           !k)
        (!previous +. fb)
        value ;
    if still_open then begin
      incr kept_open ;
      Alcotest.(check (float 0.0))
        (Printf.sprintf "k=%d: a batch left open was not lost" !k)
        (!previous +. fb)
        value
    end
    else if outcome = `Exit then incr lost_or_ran ;
    previous := value
  done ;
  Alcotest.(check bool)
    "the sweep passed the end of the submission"
    false
    !fired ;
  Alcotest.(check int)
    "no command buffer was begun while still recording"
    0
    (Atomic.get Vulkan_api_batch.begin_on_open_buffer - begins0) ;
  Alcotest.(check bool)
    "some exception left the batch open"
    true
    (!kept_open > 0) ;
  Alcotest.(check bool)
    "some exception hit after the command buffer ended"
    true
    (!lost_or_ran > 0) ;
  Memory.free buf

let test_async_exception_in_submit = async_exception_sweep ~burst:0

let test_async_exception_burst = async_exception_sweep ~burst:3

let test_async_exception_long_burst = async_exception_sweep ~burst:8

(** Runs last: destroys the device. A second destroy and a destroy through a
    stale handle do nothing, a free of a buffer of the destroyed device runs no
    Vulkan call, and a re-created device is unaffected. *)
let test_destroy_twice_and_stale_free device =
  let buf = Memory.alloc device 16 Bigarray.float32 in
  Device.destroy device ;
  Device.destroy device ;
  let fresh = Device.get 0 in
  Alcotest.(check bool) "a new device was created" true (fresh != device) ;
  Memory.free buf ;
  Device.destroy device ;
  Alcotest.(check bool)
    "the new device stays cached"
    true
    (Device.get 0 == fresh) ;
  let kernel = incr_kernel fresh "async_stale" in
  let b = upload fresh [|0.0|] in
  launch_incr kernel b ;
  Alcotest.(check (float 0.0)) "the new device launches" 1.0 (download b).(0) ;
  Memory.free b

let () =
  Alcotest.run
    "Vulkan_async_release"
    [
      ( "free path",
        [
          Alcotest.test_case
            "finaliser under the queue lock with an open batch"
            `Quick
            (with_batching_device test_finaliser_under_queue_lock);
          Alcotest.test_case
            "explicit free keeps a deferred launch error"
            `Quick
            (with_batching_device (free_keeps_error ~explicit:true));
          Alcotest.test_case
            "finaliser free keeps a deferred launch error"
            `Quick
            (with_batching_device (free_keeps_error ~explicit:false));
          Alcotest.test_case
            "byte bound of the release queue"
            `Quick
            (with_batching_device test_byte_threshold);
          Alcotest.test_case
            "idle synchronize releases"
            `Quick
            (with_batching_device test_idle_synchronize_releases);
          Alcotest.test_case
            "allocation retry re-raises the original error"
            `Quick
            (with_device test_alloc_retry_reraises);
          Alcotest.test_case
            "a signalled slot is reaped by a free"
            `Quick
            (with_batching_device test_signalled_slot_is_reaped);
          Alcotest.test_case
            "a lost device seen by the free path is kept once"
            `Quick
            (with_batching_device test_reap_lost_device);
          Alcotest.test_case
            "a lost-device wait raises once"
            `Quick
            (with_batching_device test_lost_wait_raises_once);
          Alcotest.test_case
            "another wait error keeps the slot in flight"
            `Quick
            (with_batching_device test_other_wait_error_keeps_slot);
          Alcotest.test_case
            "an asynchronous exception during submission"
            `Quick
            (with_batching_device test_async_exception_in_submit);
          Alcotest.test_case
            "a burst of asynchronous exceptions during submission"
            `Quick
            (with_batching_device test_async_exception_burst);
          Alcotest.test_case
            "a long burst of asynchronous exceptions during submission"
            `Quick
            (with_batching_device test_async_exception_long_burst);
        ] );
      ( "teardown",
        [
          Alcotest.test_case
            "destroy twice, stale handle, free after destroy"
            `Quick
            (with_device test_destroy_twice_and_stale_free);
        ] );
    ]
