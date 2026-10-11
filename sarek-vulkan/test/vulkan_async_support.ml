(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Shared helpers of test_vulkan_async_launches and
   test_vulkan_async_robustness. *)

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

let contains haystack needle =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  go 0

let with_batch_limit limit f =
  let saved = !Stream.batch_limit in
  Stream.batch_limit := limit ;
  Fun.protect ~finally:(fun () -> Stream.batch_limit := saved) f

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
