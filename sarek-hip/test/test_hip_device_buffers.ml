(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Device-only vectors, ranged/async copies, pinned memory, stream launches and
 * events on HIP hardware: the same suite test_device_buffers.exe runs on the
 * CPU backends (specs/device-buffers-async-copy.md). Self-skips without a HIP
 * device. Small sizes: the GPU is shared.
 ******************************************************************************)

module Device = Spoc_core.Device
module Suite = Device_buffers_suite

let () = Sarek_hip.Hip_plugin.register ()

let () =
  let devices = Device.init () in
  let hip =
    Array.to_list devices |> List.filter (fun d -> d.Device.framework = "HIP")
  in
  match hip with
  | [] ->
      print_endline "test_hip_device_buffers: [SKIP] no HIP device present" ;
      exit 0
  | devs ->
      List.iter Suite.run_on devs ;
      exit (Suite.exit_code ())
