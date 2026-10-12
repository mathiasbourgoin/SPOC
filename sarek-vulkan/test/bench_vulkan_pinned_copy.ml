(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Host<->device copy bandwidth on Vulkan, by path: host-mapped device buffer
 * (memcpy), unmapped device buffer with a pageable host range (staging
 * buffer), either buffer with a pinned host range (imported
 * VK_EXT_external_memory_host buffer, no staging), and the whole-buffer
 * Memory.device_to_host. Configurations are interleaved (A, B, C, A, B, C, ...)
 * so machine load hits all alike; median and min-max over [rounds] samples.
 * INFORMATION ONLY, not a gate. Usage: bench_vulkan_pinned_copy.exe [KiB]
 * [rounds].
 ******************************************************************************)

open Sarek_vulkan
module Device = Vulkan_api.Device
module Memory = Vulkan_api.Memory
module Transfer = Vulkan_api.Transfer
module Host = Vulkan_api.Host

let kib = try int_of_string Sys.argv.(1) with _ -> 65536

let rounds = try int_of_string Sys.argv.(2) with _ -> 7

let bytes = kib * 1024

(* Copies per timing: small copies are timed in batches. *)
let batch = max 1 (67108864 / bytes)

let load () =
  try In_channel.with_open_text "/proc/loadavg" In_channel.input_line
  with _ -> None

let load_text () = Option.value ~default:"?" (load ())

let alloc_device dev ~unmapped =
  Atomic.set Memory.force_unmapped unmapped ;
  Fun.protect
    ~finally:(fun () -> Atomic.set Memory.force_unmapped false)
    (fun () -> Memory.alloc dev bytes Bigarray.char)

let gbps seconds = float_of_int (bytes * batch) /. seconds /. 1e9

let time f =
  let t0 = Unix.gettimeofday () in
  for _ = 1 to batch do
    f ()
  done ;
  Unix.gettimeofday () -. t0

let report label seconds =
  let sorted = List.sort compare (List.map gbps seconds) in
  let n = List.length sorted in
  Printf.printf
    "  %-52s median %6.2f GB/s  min-max %6.2f-%6.2f  (n=%d)\n%!"
    label
    (List.nth sorted (n / 2))
    (List.hd sorted)
    (List.nth sorted (n - 1))
    n

type host_ranges = {
  pageable :
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
  pageable_addr : nativeint;
  pinned_addr : nativeint;
}

let make_ranges dev =
  let pageable = Bigarray.Array1.create Bigarray.char Bigarray.c_layout bytes in
  Bigarray.Array1.fill pageable 'a' ;
  {
    pageable;
    pageable_addr =
      Ctypes.raw_address_of_ptr (Spoc_core.Memory.borrow_bigarray_ptr pageable);
    pinned_addr = Host.alloc_pinned dev ~byte_size:bytes;
  }

let h2d buf addr () =
  Transfer.host_to_device ~src_ptr:addr ~dst:buf ~dst_offset:0 ~byte_size:bytes

let d2h buf addr () =
  Transfer.device_to_host ~src:buf ~src_offset:0 ~dst_ptr:addr ~byte_size:bytes

let configs r ~mapped ~unmapped =
  [
    ("h2d mapped buffer, pageable (memcpy)", h2d mapped r.pageable_addr);
    ( "h2d mapped buffer, pinned (memcpy or DMA by size)",
      h2d mapped r.pinned_addr );
    ("h2d unmapped buffer, pageable (staged)", h2d unmapped r.pageable_addr);
    ("h2d unmapped buffer, pinned (import)", h2d unmapped r.pinned_addr);
    ("d2h mapped buffer, pageable (ranged)", d2h mapped r.pageable_addr);
    ("d2h mapped buffer, pinned (DMA)", d2h mapped r.pinned_addr);
    ("d2h unmapped buffer, pageable (ranged)", d2h unmapped r.pageable_addr);
    ("d2h unmapped buffer, pinned (import)", d2h unmapped r.pinned_addr);
    ( "d2h whole-buffer Memory.device_to_host, mapped",
      fun () -> Memory.device_to_host ~src:mapped ~dst:r.pageable );
    ( "d2h whole-buffer Memory.device_to_host, unmapped",
      fun () -> Memory.device_to_host ~src:unmapped ~dst:r.pageable );
  ]

let run_interleaved configs =
  let samples = List.map (fun (label, _) -> (label, ref [])) configs in
  List.iter (fun (_, f) -> f ()) configs ;
  for _ = 1 to rounds do
    List.iter2 (fun (_, f) (_, acc) -> acc := time f :: !acc) configs samples
  done ;
  List.iter (fun (label, acc) -> report label !acc) samples

let bench dev =
  let r = make_ranges dev in
  let mapped = alloc_device dev ~unmapped:false in
  let unmapped = alloc_device dev ~unmapped:true in
  Printf.printf
    "%s, %d KiB, device buffer host-mapped by default: %b, cached: %b\n\
     load before: %s\n"
    dev.Device.name
    kib
    (mapped.Memory.mapped_ptr <> None)
    mapped.Memory.mapped_cached
    (load_text ()) ;
  run_interleaved (configs r ~mapped ~unmapped) ;
  Printf.printf "load after: %s\n" (load_text ()) ;
  ignore (Sys.opaque_identity r.pageable) ;
  Host.free_pinned dev r.pinned_addr ;
  Memory.free mapped ;
  Memory.free unmapped

let () =
  if not (Vulkan_api.is_available ()) then
    print_endline "bench_vulkan_pinned_copy: [SKIP] no Vulkan device present"
  else begin
    Device.init () ;
    bench (Device.get 0)
  end
