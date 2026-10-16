(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Non-blocking streams ([Transfer.create_stream ~non_blocking:true]) on every
 * linked backend: event ordering, no hidden synchronisation with the default
 * stream, auto-transfer correctness. Optional arguments restrict the run to
 * "Framework" or "Framework:index" (index among that framework's devices), so
 * a shared GPU host can exclude a device (for instance an iGPU).
 ******************************************************************************)

module Device = Spoc_core.Device
module Suite = Device_buffers_suite

let selected (devices : Device.t list) : Device.t list =
  let wanted = List.tl (Array.to_list Sys.argv) in
  if wanted = [] then devices
  else
    let index = Hashtbl.create 4 in
    List.filter
      (fun (d : Device.t) ->
        let i =
          Option.value ~default:0 (Hashtbl.find_opt index d.Device.framework)
        in
        Hashtbl.replace index d.Device.framework (i + 1) ;
        List.exists
          (fun w ->
            w = d.Device.framework
            || w = Printf.sprintf "%s:%d" d.Device.framework i)
          wanted)
      devices

let () =
  Test_helpers.Benchmarks.init_backends () ;
  let devices = Array.to_list (Device.init ()) in
  List.iter Suite.run_non_blocking_on (selected devices) ;
  exit (Suite.exit_code ())
