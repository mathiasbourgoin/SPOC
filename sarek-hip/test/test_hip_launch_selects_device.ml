(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A kernel launched through the raw core path ([Spoc_core.Kernel.launch],
 * [Spoc_core.Runtime.run]) must run on ITS device whatever device is current:
 * launching device A's module with device B current is hipErrorInvalidDevice.
 * Needs two HIP devices; self-skips otherwise.
 ******************************************************************************)

module Device = Spoc_core.Device
module Memory = Spoc_core.Memory
module Kernel = Spoc_core.Kernel
module Runtime = Spoc_core.Runtime
module H = Sarek_hip.Hip_api

let () = Sarek_hip.Hip_plugin.register ()

let source =
  {|extern "C" __global__ void inc(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += 1.0f;
}|}

let n = 256

let fail fmt =
  Printf.ksprintf
    (fun s ->
      Printf.printf "test_hip_launch_selects_device FAILED: %s\n" s ;
      exit 1)
    fmt

let make_current (other : Device.t) =
  H.Device.set_current (H.Device.get other.Device.backend_id)

let check_result what (buf : float Memory.buffer) =
  let out = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  Memory.device_to_host ~src:buf ~dst:out ;
  for i = 0 to n - 1 do
    if out.{i} <> 2.0 then fail "%s: element %d = %g, expected 2" what i out.{i}
  done

let fresh_buffer dev =
  let buf = Memory.alloc dev n Bigarray.float32 in
  let host = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  Bigarray.Array1.fill host 1.0 ;
  Memory.host_to_device ~src:host ~dst:buf ;
  buf

let grid = Runtime.dims1d 1

let block = Runtime.dims1d n

let kernel_args buf = [Runtime.ArgBuffer buf; Runtime.ArgInt32 (Int32.of_int n)]

let test_raw_launch a b =
  let kernel = Kernel.compile a ~name:"inc" ~source in
  let buf = fresh_buffer a in
  let args = Runtime.set_args a (kernel_args buf) in
  make_current b ;
  Runtime.run_kernel kernel ~args ~grid ~block () ;
  check_result "raw Kernel.launch on A with B current" buf

(* Runtime.run with another device current at the call: a miss on A, a miss on
   B, then a cache HIT on A. The buffer is prepared first, because allocating
   and copying select the buffer's device. *)
let test_runtime_alternation a b =
  let run dev other =
    let buf = fresh_buffer dev in
    make_current other ;
    Runtime.run dev ~name:"inc" ~source ~args:(kernel_args buf) ~grid ~block () ;
    check_result "Runtime.run" buf
  in
  run a a ;
  run b b ;
  run a b

let () =
  if Sarek_hip.Hip_shared.is_disabled () || not (H.is_available ()) then begin
    print_endline "test_hip_launch_selects_device SKIPPED (no HIP)" ;
    exit 0
  end ;
  match Array.to_list (Device.by_framework "HIP") with
  | a :: b :: _ ->
      test_raw_launch a b ;
      test_runtime_alternation a b ;
      print_endline "test_hip_launch_selects_device PASSED"
  | l ->
      print_endline
        (Printf.sprintf
           "test_hip_launch_selects_device SKIPPED (%d HIP device(s) of %d \
            total)"
           (List.length l)
           (Spoc_core.Device.count ()))
