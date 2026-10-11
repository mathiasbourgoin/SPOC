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
module Base = Vulkan_api_base
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Stream = Vulkan_api_stream
module Kernel = Vulkan_api_kernel
module BE = Sarek_backend_error.Backend_error

let n = 4096

(** [dst[i] = src[i] + 1]: several workgroups, so a missing barrier between
    dependent dispatches shows as a wrong element, not as a lucky ordering. *)
let step_source =
  Printf.sprintf
    {|
#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer In { float src[]; };
layout(std430, binding = 1) buffer Out { float dst[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < %du) dst[i] = src[i] + 1.0;
}
|}
    n

let incr_source =
  {|
#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Buf { float data[]; };
void main() {
    data[0] = data[0] + 1.0;
}
|}

let with_device f () =
  if not (Vulkan_api.is_available ()) then begin
    Printf.printf "[SKIP] No Vulkan device available\n%!" ;
    Alcotest.skip ()
  end
  else begin
    Device.init () ;
    f (Device.get 0)
  end

let upload device values =
  let size = Array.length values in
  let buf = Memory.alloc device size Bigarray.float32 in
  let host = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout size in
  Array.iteri (fun i v -> host.{i} <- v) values ;
  Memory.host_to_device ~src:host ~dst:buf ;
  buf

let download buf =
  let host =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout buf.Memory.size
  in
  Memory.device_to_host ~src:buf ~dst:host ;
  Array.init buf.Memory.size (fun i -> host.{i})

let launch_with ?stream kernel buffers ~grid ~block =
  let args = Kernel.create_args () in
  List.iteri (fun i b -> Kernel.set_arg_buffer args i b) buffers ;
  Kernel.launch kernel ~args ~grid ~block ~shared_mem:0 ~stream

let launch_step ?stream kernel buffers =
  launch_with
    ?stream
    kernel
    buffers
    ~grid:(Spoc_framework.Framework_sig.dims_1d (n / 64))
    ~block:(Spoc_framework.Framework_sig.dims_1d 64)

let launch_incr kernel buf =
  let one = Spoc_framework.Framework_sig.dims_1d 1 in
  launch_with kernel [buf] ~grid:one ~block:one

let step_kernel device =
  Kernel.compile_cached device ~name:"async_step" ~source:step_source

let incr_kernel device name =
  Kernel.compile_cached device ~name ~source:incr_source

let zeros = Array.make n 0.0

let check_all_equal what expected values =
  let wrong = ref 0 in
  Array.iter (fun v -> if v <> expected then incr wrong) values ;
  Alcotest.(check int) (what ^ ": elements differing from expected") 0 !wrong

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

let run_chain ?stream_of device steps =
  let kernel = step_kernel device in
  let a = upload device zeros in
  let b = upload device zeros in
  for k = 0 to steps - 1 do
    let src, dst = if k mod 2 = 0 then (a, b) else (b, a) in
    let stream = Option.map (fun f -> f k) stream_of in
    launch_step ?stream kernel [src; dst]
  done ;
  let final = if steps mod 2 = 0 then a else b in
  let result = download final in
  Memory.free a ;
  Memory.free b ;
  result

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

let contains haystack needle =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  go 0

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

let with_batch_limit limit f =
  let saved = !Stream.batch_limit in
  Stream.batch_limit := limit ;
  Fun.protect ~finally:(fun () -> Stream.batch_limit := saved) f

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
  pending_after "Memory.free" (fun () ->
      let other = upload device [|0.0|] in
      Memory.free other ;
      ()) ;
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

let expect_named_failure what name sync =
  match sync () with
  | () -> Alcotest.failf "%s: the wait did not raise" what
  | exception
      BE.Backend_error
        (BE.Runtime {error = BE.Kernel_launch_failed {kernel_name; _}; _}) ->
      Alcotest.(check bool)
        (what ^ ": the error names the kernel, got: " ^ kernel_name)
        true
        (contains kernel_name name)
  | exception BE.Backend_error _ ->
      Alcotest.failf "%s: raised something other than Kernel_launch_failed" what

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

let () =
  Alcotest.run
    "Vulkan_async_launches"
    [
      ( "batching",
        [
          Alcotest.test_case
            "N launches do not make N waits"
            `Quick
            (with_device test_launches_do_not_wait_each);
          Alcotest.test_case
            "SPOC_VULKAN_BATCH=1 waits at every launch"
            `Quick
            (with_device test_batch_of_one_waits_each);
          Alcotest.test_case
            "descriptor sets beyond one pool in one batch"
            `Quick
            (with_device test_many_sets_one_batch);
        ] );
      ( "ordering",
        [
          Alcotest.test_case
            "dependent chain across batches"
            `Quick
            (with_device test_dependent_chain);
          Alcotest.test_case
            "host read in the middle of a batch"
            `Quick
            (with_device test_host_read_mid_batch);
          Alcotest.test_case
            "free right after launches"
            `Quick
            (with_device test_free_after_launch);
          Alcotest.test_case
            "two streams share a buffer"
            `Quick
            (with_device test_two_streams);
          Alcotest.test_case
            "second stream consumes an open or full batch"
            `Quick
            (with_device test_cross_stream_batch_state);
          Alcotest.test_case
            "two domains on one device"
            `Quick
            (with_device test_two_domains);
          Alcotest.test_case
            "buffers freed from GC finalisers"
            `Quick
            (with_device test_free_from_finaliser);
          Alcotest.test_case
            "every synchronisation point completes pending launches"
            `Quick
            (with_device test_sync_points_complete_pending);
          Alcotest.test_case
            "an event covers the launches before it"
            `Quick
            (with_device test_event_after_batch);
        ] );
      ( "errors",
        [
          Alcotest.test_case
            "submit failure surfaces at the first wait"
            `Quick
            (with_device test_failure_at_first_wait);
          Alcotest.test_case
            "SPOC_VULKAN_BATCH=1 raises at the launch"
            `Quick
            (with_device test_batch_of_one_raises_at_launch);
          Alcotest.test_case
            "deferred failure when the batch fills"
            `Quick
            (with_device test_deferred_failure_batch_full);
          Alcotest.test_case
            "deferred failure when another stream launches"
            `Quick
            (with_device test_deferred_failure_other_stream);
        ] );
      ( "teardown",
        [
          Alcotest.test_case
            "device destroy with pending launches"
            `Quick
            (with_device test_destroy_with_pending);
        ] );
    ]
