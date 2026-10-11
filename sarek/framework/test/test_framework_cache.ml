(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Test suite for Framework_cache *)

open Spoc_framework_registry

let rec delete_tree path =
  if Sys.is_directory path then begin
    Array.iter
      (fun f -> delete_tree (Filename.concat path f))
      (Sys.readdir path) ;
    Sys.rmdir path
  end
  else Sys.remove path

(* Every test runs against a private cache root: never the developer's
   ~/.cache/spoc. *)
let fresh_root () =
  let root = Filename.temp_file "framework-cache-test" "" in
  Sys.remove root ;
  Unix.mkdir root 0o700 ;
  at_exit (fun () -> try delete_tree root with Sys_error _ -> ()) ;
  Unix.putenv "XDG_CACHE_HOME" root ;
  root

let (_ : string) = fresh_root ()

let test_get_cache_dir () =
  let dir = Framework_cache.get_cache_dir () in
  Alcotest.(check bool)
    "Cache directory is not empty"
    true
    (String.length dir > 0) ;
  Alcotest.(check bool)
    "Cache directory exists"
    true
    (Sys.file_exists dir && Sys.is_directory dir)

let test_compute_key () =
  let key1 =
    Framework_cache.compute_key
      ~dev_name:"Device1"
      ~driver_version:"1.0"
      ~name:"kernel_a"
      ~source:"kernel code"
  in

  let key2 =
    Framework_cache.compute_key
      ~dev_name:"Device1"
      ~driver_version:"1.0"
      ~name:"kernel_a"
      ~source:"kernel code"
  in

  Alcotest.(check string) "Same inputs produce same key" key1 key2 ;

  (* compute_key delegates to Compile_cache.make_key for the digest, so the
     key is "v2:" followed by 4 ':'-joined 32-char hex digests (device, name,
     source, options) - see Compile_cache.make_key's documented key shape. *)
  let is_hex32 s =
    String.length s = 32
    && String.for_all
         (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
         s
  in
  let parts = String.split_on_char ':' key1 in
  Alcotest.(check bool)
    "Key is schema-versioned, 4-component hex digest string"
    true
    (match parts with
    | [schema; device; name; source; options] ->
        schema = "v2" && is_hex32 device && is_hex32 name && is_hex32 source
        && is_hex32 options
    | _ -> false)

let test_compute_key_different () =
  let key1 =
    Framework_cache.compute_key
      ~dev_name:"Device1"
      ~driver_version:"1.0"
      ~name:"kernel_a"
      ~source:"code1"
  in

  let key2 =
    Framework_cache.compute_key
      ~dev_name:"Device1"
      ~driver_version:"1.0"
      ~name:"kernel_a"
      ~source:"code2"
  in

  Alcotest.(check bool)
    "Different sources produce different keys"
    true
    (key1 <> key2)

(* Non-vacuous regression test for the cache-key fix: two kernels compiled
   from the same source string (same device, same driver) must never
   collide. Before the fix, [compute_key] omitted the kernel name entirely,
   so this test would have failed by asserting key1 = key2. *)
let test_compute_key_name_differs () =
  let shared_source = "shared source defining kernel_a and kernel_b" in
  let key_a =
    Framework_cache.compute_key
      ~dev_name:"Device1"
      ~driver_version:"1.0"
      ~name:"kernel_a"
      ~source:shared_source
  in
  let key_b =
    Framework_cache.compute_key
      ~dev_name:"Device1"
      ~driver_version:"1.0"
      ~name:"kernel_b"
      ~source:shared_source
  in
  Alcotest.(check bool)
    "Different kernel names in the same source produce different keys"
    true
    (key_a <> key_b)

let test_put_and_get () =
  let key = "test_key_12345" in
  let data = "test binary data" in

  Framework_cache.put ~key ~data ;

  let retrieved = Framework_cache.get ~key in
  Alcotest.(check bool)
    "Retrieved data matches"
    true
    (match retrieved with Some d -> d = data | None -> false)

let test_get_nonexistent () =
  let key = "nonexistent_key_67890" in
  let result = Framework_cache.get ~key in
  Alcotest.(check bool)
    "Nonexistent key returns None"
    true
    (match result with None -> true | Some _ -> false)

let test_stats_initialization () =
  Framework_cache.reset_stats () ;
  let stats = Framework_cache.get_stats () in
  Alcotest.(check int) "Hits initialized to 0" 0 stats.hits ;
  Alcotest.(check int) "Misses initialized to 0" 0 stats.misses ;
  Alcotest.(check int) "Puts initialized to 0" 0 stats.puts ;
  Alcotest.(check int) "Errors initialized to 0" 0 stats.errors

let test_stats_hit () =
  Framework_cache.reset_stats () ;
  let key = "stat_test_hit" in
  let data = "hit data" in

  Framework_cache.put ~key ~data ;
  let _ = Framework_cache.get ~key in

  let stats = Framework_cache.get_stats () in
  Alcotest.(check bool) "Hit count incremented" true (stats.hits > 0)

let test_stats_miss () =
  Framework_cache.reset_stats () ;
  let _ = Framework_cache.get ~key:"nonexistent_miss_test" in

  let stats = Framework_cache.get_stats () in
  Alcotest.(check bool) "Miss count incremented" true (stats.misses > 0)

let test_stats_put () =
  Framework_cache.reset_stats () ;
  Framework_cache.put ~key:"stat_test_put" ~data:"put data" ;

  let stats = Framework_cache.get_stats () in
  Alcotest.(check bool) "Put count incremented" true (stats.puts > 0)

let test_hit_rate () =
  Framework_cache.reset_stats () ;

  (* Put and hit *)
  Framework_cache.put ~key:"rate_test" ~data:"data" ;
  let _ = Framework_cache.get ~key:"rate_test" in
  let _ = Framework_cache.get ~key:"rate_test" in
  (* Another hit *)
  let _ = Framework_cache.get ~key:"nonexistent_rate" in
  (* Miss *)

  let rate = Framework_cache.hit_rate () in
  Alcotest.(check bool)
    "Hit rate is between 0 and 1"
    true
    (rate >= 0.0 && rate <= 1.0) ;

  (* Should be ~0.66 (2 hits, 1 miss) *)
  Alcotest.(check bool) "Hit rate is reasonable" true (rate > 0.5)

let test_print_stats () =
  (* Just check it doesn't crash *)
  Framework_cache.reset_stats () ;
  Framework_cache.print_stats () ;
  Alcotest.(check bool) "print_stats executes without error" true true

let test_cache_persistence () =
  let key = "persistence_test" in
  let data = "persistent data" in

  (* Write to cache *)
  Framework_cache.put ~key ~data ;

  (* Read back immediately *)
  let retrieved = Framework_cache.get ~key in
  Alcotest.(check bool)
    "Data persists in cache"
    true
    (match retrieved with Some d -> d = data | None -> false)

let test_options_change_key () =
  let key options =
    Framework_cache.compute_key_with_options
      ~options
      ~dev_name:"Device1"
      ~driver_version:"1.0"
      ~name:"kernel_a"
      ~source:"code"
  in
  Alcotest.(check bool)
    "Different options produce different keys"
    true
    (key [("build_options", "")] <> key [("build_options", "-w")]) ;
  Alcotest.(check string)
    "No options is exactly compute_key"
    (Framework_cache.compute_key
       ~dev_name:"Device1"
       ~driver_version:"1.0"
       ~name:"kernel_a"
       ~source:"code")
    (key [])

let test_disabled_by_env () =
  let key = "disabled_by_env_key" in
  Framework_cache.put ~key ~data:"x" ;
  Unix.putenv "SAREK_DISK_CACHE" "0" ;
  Fun.protect
    ~finally:(fun () -> Unix.putenv "SAREK_DISK_CACHE" "")
    (fun () ->
      Alcotest.(check bool) "disabled" false (Framework_cache.enabled ()) ;
      Alcotest.(check (option string))
        "get misses even for an existing entry"
        None
        (Framework_cache.get ~key)) ;
  Alcotest.(check bool) "enabled again" true (Framework_cache.enabled ()) ;
  Alcotest.(check (option string))
    "entry untouched"
    (Some "x")
    (Framework_cache.get ~key)

(* Two writer domains share one process (so one PID) and one key while a reader
   domain polls it. Every read must be one writer's complete payload: a shared
   temporary name would let the writers interleave into a mixed or short
   file. *)
let test_concurrent_writers_never_torn () =
  let key = "concurrent_writers_key" in
  let size = 1_000_000 in
  let payload c = String.make size c in
  let writers_left = Atomic.make 2 in
  let writer c =
    Domain.spawn (fun () ->
        for _ = 1 to 100 do
          Framework_cache.put ~key ~data:(payload c)
        done ;
        Atomic.decr writers_left)
  in
  let reader =
    Domain.spawn (fun () ->
        let torn = ref 0 and reads = ref 0 in
        while Atomic.get writers_left > 0 do
          match Framework_cache.get ~key with
          | None -> ()
          | Some d ->
              incr reads ;
              if
                String.length d <> size
                || not (String.for_all (Char.equal d.[0]) d)
              then incr torn
        done ;
        (!torn, !reads))
  in
  let a = writer 'A' and b = writer 'B' in
  Domain.join a ;
  Domain.join b ;
  let torn, reads = Domain.join reader in
  Alcotest.(check int) "no torn read" 0 torn ;
  Alcotest.(check bool) "the reader did read entries" true (reads > 0)

(* Cached OpenCL entries are native binaries a driver loads: a directory that
   other users can write to must not be read or written. *)
let test_untrusted_dir_refused () =
  let root = fresh_root () in
  let dir = Framework_cache.get_cache_dir () in
  Alcotest.(check string)
    "dir under the private root"
    root
    (Filename.dirname dir) ;
  let key = "untrusted_dir_key" in
  Framework_cache.put ~key ~data:"trusted" ;
  Unix.chmod dir 0o777 ;
  Fun.protect
    ~finally:(fun () -> Unix.chmod dir 0o700)
    (fun () ->
      Alcotest.(check (option string))
        "get refuses an existing entry in a world-writable dir"
        None
        (Framework_cache.get ~key) ;
      Framework_cache.put ~key:"second_key" ~data:"x" ;
      Alcotest.(check bool)
        "put writes nothing there"
        false
        (Sys.file_exists (Filename.concat dir "second_key"))) ;
  Alcotest.(check (option string))
    "closed again: entry readable"
    (Some "trusted")
    (Framework_cache.get ~key)

let () =
  let open Alcotest in
  run
    "Framework_cache"
    [
      ("directory", [test_case "Get cache directory" `Quick test_get_cache_dir]);
      ( "key_computation",
        [
          test_case "Compute key" `Quick test_compute_key;
          test_case
            "Different inputs different keys"
            `Quick
            test_compute_key_different;
          test_case
            "Different kernel names in same source differ"
            `Quick
            test_compute_key_name_differs;
          test_case "Options change the key" `Quick test_options_change_key;
        ] );
      ( "cache_operations",
        [
          test_case "Put and get" `Quick test_put_and_get;
          test_case "Get nonexistent" `Quick test_get_nonexistent;
          test_case "Cache persistence" `Quick test_cache_persistence;
          test_case "SAREK_DISK_CACHE=0 disables" `Quick test_disabled_by_env;
          test_case
            "Concurrent writers never tear a read"
            `Quick
            test_concurrent_writers_never_torn;
          test_case
            "World-writable cache dir is refused"
            `Quick
            test_untrusted_dir_refused;
        ] );
      ( "statistics",
        [
          test_case "Stats initialization" `Quick test_stats_initialization;
          test_case "Stats hit" `Quick test_stats_hit;
          test_case "Stats miss" `Quick test_stats_miss;
          test_case "Stats put" `Quick test_stats_put;
          test_case "Hit rate" `Quick test_hit_rate;
          test_case "Print stats" `Quick test_print_stats;
        ] );
    ]
