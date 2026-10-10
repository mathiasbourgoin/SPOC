(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Host->device bandwidth of a 256 MiB ranged copy into a device-only vector:
 * pageable vs pinned source, synchronous vs on a stream. INFORMATION ONLY,
 * not a gate (alias e2e-hip-bench). Best of 5 per configuration.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let bytes = 256 * 1024 * 1024

let reps = 5

let best f =
  let t = ref infinity in
  for _ = 1 to reps do
    let t0 = Unix.gettimeofday () in
    f () ;
    t := Float.min !t (Unix.gettimeofday () -. t0)
  done ;
  !t

let gbps s = float_of_int bytes /. s /. 1e9

let () = Sarek_hip.Hip_plugin.register ()

let () =
  let devices = Device.init () in
  match
    Array.to_list devices |> List.find_opt (fun d -> d.Device.framework = "HIP")
  with
  | None -> print_endline "bench_hip_pinned_copy: [SKIP] no HIP device present"
  | Some dev ->
      let dst = Transfer.create_device_only dev Vector.char bytes in
      let pageable =
        Bigarray.Array1.create Bigarray.char Bigarray.c_layout bytes
      in
      Bigarray.Array1.fill pageable 'a' ;
      let pinned = Transfer.alloc_pinned dev Bigarray.char bytes in
      Bigarray.Array1.fill pinned 'b' ;
      let s = Transfer.create_stream dev in
      let sync src () = Transfer.copy_to_device src ~dst ~dst_offset:0 in
      let async src () =
        Transfer.copy_to_device ~stream:s src ~dst ~dst_offset:0 ;
        Transfer.synchronize_stream s
      in
      Printf.printf
        "%s, 256 MiB host->device, best of %d\n"
        dev.Device.name
        reps ;
      List.iter
        (fun (label, f) ->
          Printf.printf "  %-26s %6.2f GB/s\n%!" label (gbps (best f)))
        [
          ("pageable, synchronous", sync pageable);
          ("pinned,   synchronous", sync pinned);
          ("pageable, stream", async pageable);
          ("pinned,   stream", async pinned);
        ] ;
      Transfer.destroy_stream s ;
      Transfer.free_pinned pinned ;
      Transfer.free_all_buffers dst
