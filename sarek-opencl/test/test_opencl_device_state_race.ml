(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Several domains using an OpenCL device for the first time at once get one
 * device state (context, default queue, stream list) between them. With two,
 * a stream registered in one would be missed by a Device.synchronize that
 * finishes the other's queues. Backend level; skips without OpenCL.
 ******************************************************************************)

open Sarek_opencl
module Backend = Opencl_plugin_base.Opencl

let domains = 8

let created () = Atomic.get Opencl_plugin_base.For_testing.device_states_created

(* The OpenCL bindings are lazy, and forcing one lazy from several domains at
   once raises [CamlinternalLazy.Undefined]: force the ones a stream creation
   uses here, without creating a plugin device state. *)
let force_bindings () =
  let device = Opencl_api.Device.get 0 in
  let context = Opencl_api.Context.create device in
  let queue = Opencl_api.CommandQueue.create context () in
  Opencl_api.CommandQueue.release queue ;
  Opencl_api.Context.release context

(* All [domains] domains create a stream on [device] at the same moment, then
   destroy it. *)
let race device =
  let go = Atomic.make false in
  let ready = Atomic.make 0 in
  let racers =
    List.init domains (fun _ ->
        Domain.spawn (fun () ->
            Atomic.incr ready ;
            while not (Atomic.get go) do
              Domain.cpu_relax ()
            done ;
            Backend.Stream.create device))
  in
  while Atomic.get ready < domains do
    Domain.cpu_relax ()
  done ;
  Atomic.set go true ;
  List.iter (fun d -> Backend.Stream.destroy (Domain.join d)) racers

let check () =
  if not (Backend.is_available ()) then begin
    Printf.printf "[SKIP] No OpenCL device available\n%!" ;
    Alcotest.skip ()
  end
  else begin
    Backend.Device.init () ;
    force_bindings () ;
    for id = 0 to Backend.Device.count () - 1 do
      let before = created () in
      race (Backend.Device.get id) ;
      Alcotest.(check int)
        (Printf.sprintf "device %d: one state for %d racing domains" id domains)
        (before + 1)
        (created ())
    done
  end

let () =
  Alcotest.run
    "Opencl_device_state_race"
    [
      ( "device_state",
        [Alcotest.test_case "first use from several domains" `Quick check] );
    ]
