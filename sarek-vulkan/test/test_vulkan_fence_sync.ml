(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Fence discipline of the Vulkan backend, on a real device.
 *
 * - [Event.synchronize] on a recorded event returns, and so does a second
 *   record/synchronize on the same event.
 * - An event that was never recorded synchronizes immediately.
 * - [Event.elapsed] refuses with [Feature_not_supported] instead of answering
 *   a fabricated 0.0.
 * - A launch that fails on the host (push-constant block over the 128-byte
 *   limit) does not make the next launch on the same stream hang.
 * - A failed [vkQueueSubmit] (injected through
 *   [Vulkan_api_device.inject_submit_failure]) leaves neither the launch's
 *   stream nor a recorded event waiting on a fence that was never submitted.
 *
 * A broken fence shows up as a hang; the dune rule bounds every fence wait
 * with SPOC_VULKAN_FENCE_TIMEOUT_S so a hang ends in an error. Skips when no
 * Vulkan device is present.
 ******************************************************************************)

open Sarek_vulkan
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Stream = Vulkan_api_stream
module Event = Vulkan_api_event
module Kernel = Vulkan_api_kernel
module BE = Sarek_backend_error.Backend_error

let source =
  {|
#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Buf { float data[]; };
void main() {
    data[0] = 1.0;
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

let test_record_then_synchronize device =
  let stream = Stream.default device in
  let event = Event.create_with_device device in
  Event.record event stream ;
  Event.synchronize event ;
  (* A second record re-arms the same event. *)
  Event.record event stream ;
  Event.synchronize event ;
  Event.destroy event

let test_unrecorded_synchronize device =
  let event = Event.create_with_device device in
  Event.synchronize event ;
  Event.destroy event

let test_elapsed_refuses device =
  let start = Event.create_with_device device in
  let stop = Event.create_with_device device in
  let outcome =
    match Event.elapsed ~start ~stop with
    | ms -> Printf.sprintf "returned %f" ms
    | exception
        BE.Backend_error
          (BE.Plugin {error = BE.Feature_not_supported _; backend = "Vulkan"})
      ->
        "feature_not_supported"
  in
  Event.destroy start ;
  Event.destroy stop ;
  Alcotest.(check string)
    "elapsed refuses with Feature_not_supported"
    "feature_not_supported"
    outcome

let launch_once compiled buf ~extra_scalars =
  let args = Kernel.create_args () in
  Kernel.set_arg_buffer args 0 buf ;
  for i = 1 to extra_scalars do
    Kernel.set_arg_int32 args i (Int32.of_int i)
  done ;
  let dims = Spoc_framework.Framework_sig.dims_1d 1 in
  Kernel.launch compiled ~args ~grid:dims ~block:dims ~shared_mem:0 ~stream:None

let read_back_and_free device buf =
  Device.synchronize device ;
  let host = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 1 in
  Memory.device_to_host ~src:buf ~dst:host ;
  Memory.free buf ;
  host.{0}

let test_failed_launch_does_not_poison_stream device =
  let compiled = Kernel.compile_cached device ~name:"fence_sync" ~source in
  let buf = Memory.alloc device 1 Bigarray.float32 in
  (* 4 bytes of vector length + 40 * 4 bytes of scalars > 128 bytes. *)
  let failed =
    match launch_once compiled buf ~extra_scalars:40 with
    | () -> false
    | exception BE.Backend_error _ -> true
  in
  Alcotest.(check bool) "oversized push constants are refused" true failed ;
  launch_once compiled buf ~extra_scalars:0 ;
  Alcotest.(check (float 0.0001))
    "the launch after a failed one runs"
    1.0
    (read_back_and_free device buf)

let inject_device_lost () =
  Device.inject_submit_failure := Some Vulkan_types.VK_ERROR_DEVICE_LOST

let raises_backend_error f =
  match f () with () -> false | exception BE.Backend_error _ -> true

let test_failed_submit_launch device =
  let compiled = Kernel.compile_cached device ~name:"fence_sync" ~source in
  let buf = Memory.alloc device 1 Bigarray.float32 in
  inject_device_lost () ;
  Alcotest.(check bool)
    "an injected submit failure is raised"
    true
    (raises_backend_error (fun () -> launch_once compiled buf ~extra_scalars:0)) ;
  launch_once compiled buf ~extra_scalars:0 ;
  Alcotest.(check (float 0.0001))
    "the launch after a failed submit runs"
    1.0
    (read_back_and_free device buf)

let test_failed_submit_event device =
  let stream = Stream.default device in
  let event = Event.create_with_device device in
  inject_device_lost () ;
  Alcotest.(check bool)
    "an injected submit failure is raised"
    true
    (raises_backend_error (fun () -> Event.record event stream)) ;
  Event.synchronize event ;
  Event.record event stream ;
  Event.synchronize event ;
  Event.destroy event

let () =
  Alcotest.run
    "Vulkan_fence_sync"
    [
      ( "event",
        [
          Alcotest.test_case
            "record then synchronize returns"
            `Quick
            (with_device test_record_then_synchronize);
          Alcotest.test_case
            "synchronize on an unrecorded event returns"
            `Quick
            (with_device test_unrecorded_synchronize);
          Alcotest.test_case
            "elapsed refuses instead of answering 0.0"
            `Quick
            (with_device test_elapsed_refuses);
        ] );
      ( "launch",
        [
          Alcotest.test_case
            "failed launch does not hang the next one"
            `Quick
            (with_device test_failed_launch_does_not_poison_stream);
          Alcotest.test_case
            "failed submit does not hang the next launch"
            `Quick
            (with_device test_failed_submit_launch);
        ] );
      ( "submit",
        [
          Alcotest.test_case
            "failed event submit leaves the event usable"
            `Quick
            (with_device test_failed_submit_event);
        ] );
    ]
