(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Bandwidth of a 256 MiB ranged copy into / out of a device-only Metal vector:
 * pageable vs pinned (page-aligned) host memory, synchronous vs on a stream
 * (synchronous semantics on Metal). INFORMATION ONLY, not a gate (alias
 * e2e-metal-bench). Best of 5 per configuration. On unified memory each copy
 * is one CPU memcpy into the shared MTLBuffer.
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

let () = Sarek_metal.Metal_plugin.init ()

let () =
  let devices = Device.init () in
  match
    Array.to_list devices
    |> List.find_opt (fun d -> d.Device.framework = "Metal")
  with
  | None -> print_endline "bench_metal_copy: [SKIP] no Metal device present"
  | Some dev ->
      let dev_vec = Transfer.create_device_only dev Vector.char bytes in
      let pageable =
        Bigarray.Array1.create Bigarray.char Bigarray.c_layout bytes
      in
      Bigarray.Array1.fill pageable 'a' ;
      let pinned = Transfer.alloc_pinned dev Bigarray.char bytes in
      Bigarray.Array1.fill pinned 'b' ;
      let s = Transfer.create_stream dev in
      let h2d ?stream src () =
        Transfer.copy_to_device ?stream src ~dst:dev_vec ~dst_offset:0 ;
        Option.iter Transfer.synchronize_stream stream
      in
      let d2h ?stream dst () =
        Transfer.copy_to_host ?stream ~src:dev_vec ~src_offset:0 dst ;
        Option.iter Transfer.synchronize_stream stream
      in
      Printf.printf
        "%s, 256 MiB ranged copies, best of %d\n"
        dev.Device.name
        reps ;
      List.iter
        (fun (label, f) ->
          Printf.printf "  %-34s %6.2f GB/s\n%!" label (gbps (best f)))
        [
          ("host->device pageable, synchronous", h2d pageable);
          ("host->device pinned,   synchronous", h2d pinned);
          ("host->device pageable, stream", h2d ~stream:s pageable);
          ("host->device pinned,   stream", h2d ~stream:s pinned);
          ("device->host pageable, synchronous", d2h pageable);
          ("device->host pinned,   synchronous", d2h pinned);
        ] ;
      Transfer.destroy_stream s ;
      Transfer.free_pinned pinned ;
      Transfer.free_all_buffers dev_vec
