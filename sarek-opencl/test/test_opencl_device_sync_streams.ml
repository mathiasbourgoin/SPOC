(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Device.synchronize waits for every queue of the device, not only the
 * default one: [Sarek.Device.synchronize] releases the stream keep-alive
 * ledger of the whole device on return (R-6), which is only sound if work
 * enqueued on a stream's own command queue has completed.
 *
 * A single work item spins, then writes 42 through a stream's queue; after
 * Device.synchronize, a read on the default queue (which is not ordered after
 * the stream's queue) must see 42. Backend level; skips without OpenCL.
 ******************************************************************************)

open Sarek_opencl
module Backend = Opencl_plugin_base.Opencl

let slow_write_source =
  {|
__kernel void slow_write(__global float *out, int iters) {
    float acc = 0.0f;
    for (int k = 0; k < iters; k++) acc = acc * 0.5f + 1.0f;
    out[0] = acc + 40.0f;
}
|}

let iters = 50_000_000l

let check_device device =
  let out = Backend.Memory.alloc device 1 Bigarray.float32 in
  let host = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 1 in
  host.{0} <- -1.0 ;
  Backend.Memory.host_to_device ~src:host ~dst:out ;
  let stream = Backend.Stream.create device in
  let compiled =
    Backend.Kernel.compile_cached
      device
      ~name:"slow_write"
      ~source:slow_write_source
  in
  let args = Backend.Kernel.create_args () in
  Backend.Kernel.set_arg_buffer args 0 out ;
  Backend.Kernel.set_arg_int32 args 1 iters ;
  let dims = Spoc_framework.Framework_sig.dims_1d in
  Backend.Kernel.launch
    compiled
    ~args
    ~grid:(dims 1)
    ~block:(dims 1)
    ~shared_mem:0
    ~stream:(Some stream) ;
  Backend.Device.synchronize device ;
  Backend.Memory.device_to_host ~src:out ~dst:host ;
  Backend.Stream.synchronize stream ;
  Backend.Stream.destroy stream ;
  Backend.Memory.free out ;
  Alcotest.(check (float 0.0))
    "the stream's kernel completed before Device.synchronize returned"
    42.0
    host.{0}

let on_every_device () =
  if not (Backend.is_available ()) then begin
    Printf.printf "[SKIP] No OpenCL device available\n%!" ;
    Alcotest.skip ()
  end
  else begin
    Backend.Device.init () ;
    for id = 0 to Backend.Device.count () - 1 do
      check_device (Backend.Device.get id)
    done
  end

let () =
  Alcotest.run
    "Opencl_device_sync_streams"
    [
      ( "device_synchronize",
        [
          Alcotest.test_case
            "finishes every stream queue of the device"
            `Quick
            on_every_device;
        ] );
    ]
