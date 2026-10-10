(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Device-only vectors, ranged/async copies, pinned memory, stream launches and
 * events on every device this test links: the full suite on the CPU backends
 * (Native, Interpreter) and on CUDA devices; the capability refusal on
 * OpenCL, Vulkan and Metal devices. HIP runs the same suite from
 * sarek-hip/test/test_hip_device_buffers.exe.
 ******************************************************************************)

module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer
module Suite = Device_buffers_suite

let () =
  Test_helpers.Benchmarks.init_backends () ;
  let devices = Device.init () in
  Printf.printf
    "Device buffers / async copies (%d devices)\n%!"
    (Array.length devices) ;
  (* A stream of one device must not order work on another. *)
  (match
     ( Array.to_list devices
       |> List.find_opt (fun d -> d.Device.framework = "Native"),
       Array.to_list devices
       |> List.find_opt (fun d -> d.Device.framework = "Interpreter") )
   with
  | Some nat, Some interp ->
      let s = Transfer.create_stream interp in
      let v = Transfer.create_device_only nat Spoc_core.Vector.int32 4 in
      Suite.check
        "stream of another device refused"
        (Suite.refuses_invalid (fun () ->
             Transfer.copy_to_device
               ~stream:s
               (Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout 1)
               ~dst:v
               ~dst_offset:0)) ;
      Transfer.destroy_stream s
  | _ -> Suite.check "Native and Interpreter devices present" false) ;
  Array.iter
    (fun (d : Device.t) ->
      match d.framework with
      | "Native" | "Interpreter" | "CUDA/PTX" | "CUDA/C" -> Suite.run_on d
      | "OpenCL" | "Vulkan" | "Metal" -> Suite.check_refused d
      | other -> Printf.printf "  [%s] skipped\n" other)
    devices ;
  exit (Suite.exit_code ())
