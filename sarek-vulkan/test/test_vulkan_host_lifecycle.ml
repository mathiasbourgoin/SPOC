(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Pinned and registered host memory across a device's destruction on Vulkan
 * (specs/vulkan-async-api.md): imports are destroyed with their device; a late
 * [unregister] of a dropped registration succeeds; a pinned allocation made on
 * the old device can be registered on, and freed through, the re-created one.
 * Skips when no Vulkan device is present.
 ******************************************************************************)

open Sarek_vulkan
open Vulkan_async_support
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Transfer = Vulkan_api_transfer
module Host = Vulkan_api_host

let size = 100_000

let char_array n = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

let address_of ba =
  Ctypes.raw_address_of_ptr (Spoc_core.Memory.borrow_bigarray_ptr ba)

(** Round-trip [size] bytes of [host] through a fresh buffer of [device]; the
    number of bytes that differ. *)
let roundtrip device host =
  let buf = Memory.alloc device size Bigarray.char in
  let out = char_array size in
  Transfer.host_to_device
    ~src_ptr:(address_of host)
    ~dst:buf
    ~dst_offset:0
    ~byte_size:size ;
  Transfer.device_to_host
    ~src:buf
    ~src_offset:0
    ~dst_ptr:(address_of out)
    ~byte_size:size ;
  let bad = ref 0 in
  for i = 0 to size - 1 do
    if Bigarray.Array1.get out i <> Bigarray.Array1.get host i then incr bad
  done ;
  Memory.free buf ;
  !bad

let test_import_alignment device =
  match device.Device.host_import_alignment with
  | None -> Alcotest.skip ()
  | Some alignment ->
      Alcotest.(check bool)
        "the import granule is a positive power of two"
        true
        (alignment > 0 && alignment land (alignment - 1) = 0)

let test_across_device_destroy device =
  if device.Device.host_import_alignment = None then Alcotest.skip () ;
  let reg = char_array size in
  Bigarray.Array1.fill reg 'r' ;
  let reg_addr = address_of reg in
  Host.register device reg_addr ~byte_size:size ;
  let pinned = Host.alloc_pinned device ~byte_size:size in
  Device.destroy device ;
  (* The registration was dropped with the device: a late unregister is quiet. *)
  Host.unregister device reg_addr ;
  let fresh = Device.get 0 in
  (* The pinned block belongs to no device now; the new one can register it. *)
  Host.register fresh pinned ~byte_size:size ;
  Host.unregister fresh pinned ;
  Host.register fresh reg_addr ~byte_size:size ;
  Alcotest.(check int)
    "copies through a range registered on the re-created device"
    0
    (roundtrip fresh reg) ;
  Host.unregister fresh reg_addr ;
  Host.free_pinned fresh pinned ;
  ignore (Sys.opaque_identity reg)

(* The properties function is the owning driver's entry point: every device
   record resolves its own, none is shared through a process-wide cache. *)
let test_per_device_function device =
  match device.Device.host_pointer_properties_fn with
  | None ->
      Alcotest.(check bool)
        "no function without an import granule"
        true
        (device.Device.host_import_alignment = None)
  | Some old_fn -> (
      Alcotest.(check bool)
        "a function exactly when there is an import granule"
        true
        (device.Device.host_import_alignment <> None) ;
      Device.destroy device ;
      let fresh = Device.get 0 in
      Alcotest.(check bool) "a new device record" true (fresh != device) ;
      match fresh.Device.host_pointer_properties_fn with
      | Some fresh_fn ->
          Alcotest.(check bool)
            "the new device resolved its own function"
            true
            (fresh_fn != old_fn)
      | None -> Alcotest.fail "the re-created device lost the extension")

let () =
  Alcotest.run
    "vulkan_host_lifecycle"
    [
      ( "host memory",
        [
          Alcotest.test_case
            "import granule"
            `Quick
            (with_device test_import_alignment);
          Alcotest.test_case
            "per-device properties function"
            `Quick
            (with_device test_per_device_function);
          Alcotest.test_case
            "across device destruction"
            `Quick
            (with_device test_across_device_destroy);
        ] );
    ]
