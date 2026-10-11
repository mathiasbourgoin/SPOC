(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Hardware test for the OpenCL persistent on-disk compile cache (H5c).
 *
 * Every OpenCL device the process can see is exercised. A device whose driver
 * returns no program binary legitimately never populates the cache; for it the
 * "stores an entry" assertions are skipped, the "never an error" ones are not.
 *
 * Isolation: XDG_CACHE_HOME points at a private directory per test, so the
 * owner's ~/.cache/spoc is never read or written. Cache activity is observed
 * through [Framework_cache.get_stats] and the directory contents only, i.e.
 * through the public surface, not through a test-only hook.
 *
 * Skips (does not fail) when no OpenCL device is present.
 ******************************************************************************)

open Sarek_opencl
module Backend = Opencl_plugin_base.Opencl
module Cache = Spoc_framework_registry.Framework_cache

let count = 256

let source =
  {|
__kernel void poly(__global float *out) {
    int i = get_global_id(0);
    float x = (float)(i + 1) * 0.37f;
    out[i] = x * x + 0.1f * x / (x + 3.3f) + sqrt(x) / 3.0f;
}
|}

let fresh_cache_dir () =
  let dir = Filename.temp_file "h5c-cache" "" in
  Sys.remove dir ;
  Unix.mkdir dir 0o755 ;
  Unix.putenv "XDG_CACHE_HOME" dir ;
  Filename.concat dir "spoc"

let entries dir =
  if Sys.file_exists dir then Array.to_list (Sys.readdir dir) else []

let read_file path = In_channel.with_open_bin path In_channel.input_all

let write_file path data =
  Out_channel.with_open_bin path (fun oc -> output_string oc data)

(* Compile [source] through the plugin (no in-process memo: [compile], not
   [compile_cached]), run it, and return the output bits. *)
let run_poly device =
  let compiled = Backend.Kernel.compile device ~name:"poly" ~source in
  let buf = Backend.Memory.alloc device count Bigarray.float32 in
  let args = Backend.Kernel.create_args () in
  Backend.Kernel.set_arg_buffer args 0 buf ;
  Backend.Kernel.launch
    compiled
    ~args
    ~grid:(Spoc_framework.Framework_sig.dims_1d count)
    ~block:(Spoc_framework.Framework_sig.dims_1d 1)
    ~shared_mem:0
    ~stream:None ;
  Backend.Device.synchronize device ;
  let host = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout count in
  Backend.Memory.device_to_host ~src:buf ~dst:host ;
  Backend.Memory.free buf ;
  Array.init count (fun i -> Int32.bits_of_float host.{i})

let for_each_device f =
  if not (Backend.is_available ()) then begin
    Printf.printf "[SKIP] No OpenCL device available\n%!" ;
    Alcotest.skip ()
  end
  else
    for idx = 0 to Backend.Device.count () - 1 do
      let device = Backend.Device.get idx in
      Printf.printf "[H5c] device %d: %s\n%!" idx (Backend.Device.name device) ;
      f device
    done

let check_bits msg a b = Alcotest.(check (array int32)) msg a b

let test_second_build_is_disk_hit () =
  for_each_device (fun device ->
      let dir = fresh_cache_dir () in
      Cache.reset_stats () ;
      let first = run_poly device in
      let stored = (Cache.get_stats ()).Cache.puts in
      let second = run_poly device in
      let stats = Cache.get_stats () in
      check_bits "bit-identical across cold and cached build" first second ;
      Alcotest.(check int) "first build stores one entry" 1 stored ;
      Alcotest.(check int)
        "one cache file on disk"
        1
        (List.length (entries dir)) ;
      Alcotest.(check int) "second build is a disk hit" 1 stats.Cache.hits ;
      Alcotest.(check int) "second build stores nothing" 1 stats.Cache.puts)

let build_with device options =
  let context = Opencl_api.Context.create device in
  let program = Opencl_api.Program.create_from_source context source in
  Opencl_api.Program.build program ~options () ;
  let kernel = Opencl_api.Kernel.create program "poly" in
  Opencl_api.Kernel.release kernel ;
  Opencl_api.Program.release program

let test_options_change_misses () =
  for_each_device (fun device ->
      let dir = fresh_cache_dir () in
      let api_device = Opencl_api.Device.get (Backend.Device.id device) in
      Cache.reset_stats () ;
      build_with api_device "" ;
      build_with api_device "-w" ;
      let after_two = Cache.get_stats () in
      Alcotest.(check int) "different options never hit" 0 after_two.Cache.hits ;
      Alcotest.(check int) "one entry per option string" 2 after_two.Cache.puts ;
      Alcotest.(check int) "two files" 2 (List.length (entries dir)) ;
      build_with api_device "-w" ;
      Alcotest.(check int) "same options hit" 1 (Cache.get_stats ()).Cache.hits)

(* [garbage] replaces every stored entry. A raw string fails the entry
   checksum; a correctly sealed one reaches the driver, which must refuse it
   (CL_INVALID_BINARY) and leave the build to fall back to source. *)
let check_damaged_entry_recovers ~garbage =
  for_each_device (fun device ->
      let dir = fresh_cache_dir () in
      let reference = run_poly device in
      Alcotest.(check bool)
        "an entry was stored to corrupt"
        true
        (entries dir <> []) ;
      List.iter
        (fun f -> write_file (Filename.concat dir f) garbage)
        (entries dir) ;
      let recovered = run_poly device in
      check_bits
        "corrupt entry: source fallback is bit-identical"
        reference
        recovered ;
      List.iter
        (fun f ->
          Alcotest.(check bool)
            "corrupt entry was overwritten"
            true
            (read_file (Filename.concat dir f) <> garbage))
        (entries dir) ;
      Cache.reset_stats () ;
      check_bits "rewritten entry loads" reference (run_poly device) ;
      Alcotest.(check int)
        "rewritten entry is a hit"
        1
        (Cache.get_stats ()).Cache.hits)

let test_corrupt_entry_falls_back_and_rewrites () =
  check_damaged_entry_recovers ~garbage:"not an OpenCL binary"

let test_driver_refused_binary_falls_back_and_rewrites () =
  check_damaged_entry_recovers
    ~garbage:(Opencl_disk_cache.seal "sealed but not an OpenCL binary")

let test_opt_out () =
  for_each_device (fun device ->
      let dir = fresh_cache_dir () in
      Unix.putenv "SAREK_DISK_CACHE" "0" ;
      Fun.protect
        ~finally:(fun () -> Unix.putenv "SAREK_DISK_CACHE" "")
        (fun () ->
          Cache.reset_stats () ;
          ignore (run_poly device) ;
          Alcotest.(check int)
            "no entries written"
            0
            (List.length (entries dir)) ;
          Alcotest.(check int) "no puts" 0 (Cache.get_stats ()).Cache.puts))

(* Pure (no device): every component of the build target is part of the key. *)
let test_key_covers_every_target_field () =
  let base =
    {
      Opencl_disk_cache.platform_name = "p";
      platform_version = "pv";
      device_name = "d";
      device_version = "dv";
      driver_version = "drv";
      build_options = "";
    }
  in
  let key target = Opencl_disk_cache.key target ~source:"src" in
  let variants =
    [
      ("platform name", {base with platform_name = "p2"});
      ("platform version", {base with platform_version = "pv2"});
      ("device name", {base with device_name = "d2"});
      ("device version", {base with device_version = "dv2"});
      ("driver version", {base with driver_version = "drv2"});
      ("build options", {base with build_options = "-w"});
    ]
  in
  List.iter
    (fun (field, target) ->
      Alcotest.(check bool)
        (field ^ " changes the key")
        true
        (key target <> key base))
    variants ;
  Alcotest.(check bool)
    "source changes the key"
    true
    (Opencl_disk_cache.key base ~source:"src2" <> key base)

let test_seal_rejects_damage () =
  let entry = Opencl_disk_cache.seal "binary-bytes" in
  Alcotest.(check (option string))
    "round trip"
    (Some "binary-bytes")
    (Opencl_disk_cache.unseal entry) ;
  let flipped = Bytes.of_string entry in
  Bytes.set flipped (Bytes.length flipped - 1) 'X' ;
  Alcotest.(check (option string))
    "bit flip"
    None
    (Opencl_disk_cache.unseal (Bytes.to_string flipped)) ;
  Alcotest.(check (option string))
    "truncated"
    None
    (Opencl_disk_cache.unseal (String.sub entry 0 10)) ;
  Alcotest.(check (option string)) "empty" None (Opencl_disk_cache.unseal "")

let () =
  Alcotest.run
    "Opencl_disk_cache"
    [
      ( "entry_format",
        [
          Alcotest.test_case
            "key covers every target field"
            `Quick
            test_key_covers_every_target_field;
          Alcotest.test_case
            "damaged entries are rejected"
            `Quick
            test_seal_rejects_damage;
        ] );
      ( "disk_cache",
        [
          Alcotest.test_case
            "second build is a disk hit, bit-identical"
            `Quick
            test_second_build_is_disk_hit;
          Alcotest.test_case
            "changed build options miss"
            `Quick
            test_options_change_misses;
          Alcotest.test_case
            "corrupt entry falls back to source and is rewritten"
            `Quick
            test_corrupt_entry_falls_back_and_rewrites;
          Alcotest.test_case
            "binary refused by the driver falls back to source and is rewritten"
            `Quick
            test_driver_refused_binary_falls_back_and_rewrites;
          Alcotest.test_case
            "SAREK_DISK_CACHE=0 disables the cache"
            `Quick
            test_opt_out;
        ] );
    ]
