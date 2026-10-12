(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Ranged host<->device copy bandwidth on Vulkan, by path: host-mapped device
 * buffer (memcpy), unmapped device buffer with a pageable host range (staging
 * buffer), unmapped device buffer with a pinned host range (imported
 * VK_EXT_external_memory_host buffer, no staging). Configurations are
 * interleaved (A, B, C, A, B, C, ...) so machine load hits all alike; median
 * and min-max over [rounds] samples. INFORMATION ONLY, not a gate. Usage:
 * bench_vulkan_pinned_copy.exe [KiB] [rounds].
 ******************************************************************************)

open Sarek_vulkan
module Device = Vulkan_api.Device
module Memory = Vulkan_api.Memory
module Transfer = Vulkan_api.Transfer
module Host = Vulkan_api.Host

let kib = try int_of_string Sys.argv.(1) with _ -> 65536

let rounds = try int_of_string Sys.argv.(2) with _ -> 7

let bytes = kib * 1024

(* Samples per timing: small copies are timed in batches. *)
let batch = max 1 (67108864 / bytes)

let load () =
  try In_channel.with_open_text "/proc/loadavg" In_channel.input_line
  with _ -> None

let alloc_device dev ~unmapped =
  Atomic.set Memory.force_unmapped unmapped ;
  Fun.protect
    ~finally:(fun () -> Atomic.set Memory.force_unmapped false)
    (fun () -> Memory.alloc dev bytes Bigarray.char)

let gbps seconds = float_of_int (bytes * batch) /. seconds /. 1e9

let median sorted = List.nth sorted (List.length sorted / 2)

let time f =
  let t0 = Unix.gettimeofday () in
  for _ = 1 to batch do
    f ()
  done ;
  Unix.gettimeofday () -. t0

let report label seconds =
  let sorted = List.sort compare (List.map gbps seconds) in
  Printf.printf
    "  %-44s median %6.2f GB/s  min-max %6.2f-%6.2f  (n=%d)\n%!"
    label
    (median sorted)
    (List.hd sorted)
    (List.nth sorted (List.length sorted - 1))
    (List.length sorted)

let () =
  if not (Vulkan_api.is_available ()) then
    print_endline "bench_vulkan_pinned_copy: [SKIP] no Vulkan device present"
  else begin
    Device.init () ;
    let dev = Device.get 0 in
    let pageable =
      Bigarray.Array1.create Bigarray.char Bigarray.c_layout bytes
    in
    Bigarray.Array1.fill pageable 'a' ;
    let pageable_addr =
      Ctypes.raw_address_of_ptr (Spoc_core.Memory.borrow_bigarray_ptr pageable)
    in
    let pinned_addr = Host.alloc_pinned dev ~byte_size:bytes in
    let mapped = alloc_device dev ~unmapped:false in
    let unmapped = alloc_device dev ~unmapped:true in
    Printf.printf
      "%s, %d KiB, device buffer host-mapped by default: %b\nload before: %s\n"
      dev.Device.name
      kib
      (mapped.Memory.mapped_ptr <> None)
      (Option.value ~default:"?" (load ())) ;
    let h2d buf addr () =
      Transfer.host_to_device
        ~src_ptr:addr
        ~dst:buf
        ~dst_offset:0
        ~byte_size:bytes
    in
    let d2h buf addr () =
      Transfer.device_to_host
        ~src:buf
        ~src_offset:0
        ~dst_ptr:addr
        ~byte_size:bytes
    in
    let configs =
      [
        ("h2d mapped device buffer, pageable (memcpy)", h2d mapped pageable_addr);
        ( "h2d mapped device buffer, pinned (dma above threshold)",
          h2d mapped pinned_addr );
        ( "h2d unmapped device buffer, pageable (staged)",
          h2d unmapped pageable_addr );
        ("h2d unmapped device buffer, pinned (import)", h2d unmapped pinned_addr);
        ("d2h mapped device buffer, pageable (memcpy)", d2h mapped pageable_addr);
        ("d2h mapped device buffer, pinned (dma)", d2h mapped pinned_addr);
        ( "d2h unmapped device buffer, pageable (staged)",
          d2h unmapped pageable_addr );
        ("d2h unmapped device buffer, pinned (import)", d2h unmapped pinned_addr);
      ]
    in
    List.iter (fun (_, f) -> f ()) configs ;
    let samples = List.map (fun (l, _) -> (l, ref [])) configs in
    for _ = 1 to rounds do
      List.iter2 (fun (_, f) (_, acc) -> acc := time f :: !acc) configs samples
    done ;
    List.iter (fun (label, acc) -> report label !acc) samples ;
    Printf.printf "load after: %s\n" (Option.value ~default:"?" (load ())) ;
    ignore (Sys.opaque_identity pageable) ;
    Host.free_pinned dev pinned_addr ;
    Memory.free mapped ;
    Memory.free unmapped
  end
