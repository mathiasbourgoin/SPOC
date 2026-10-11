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
        ] );
      ( "errors",
        [
          Alcotest.test_case
            "submit failure surfaces at the first wait"
            `Quick
            (with_device test_failure_at_first_wait);
        ] );
    ]
