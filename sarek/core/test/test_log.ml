(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Log - SAREK_DEBUG parsing and the component/level gate
 *
 * Every assertion is about what reaches stdout, the stream Log writes to.
 ******************************************************************************)

module Log = Spoc_core.Log

(* Everything [f] prints on stdout. *)
let captured f =
  flush stdout ;
  let file = Filename.temp_file "test_log" ".out" in
  let saved = Unix.dup Unix.stdout in
  let fd = Unix.openfile file [Unix.O_WRONLY; Unix.O_TRUNC] 0o600 in
  Unix.dup2 fd Unix.stdout ;
  Unix.close fd ;
  Fun.protect
    ~finally:(fun () ->
      flush stdout ;
      Unix.dup2 saved Unix.stdout ;
      Unix.close saved)
    f ;
  let text = In_channel.with_open_bin file In_channel.input_all in
  Sys.remove file ;
  text

(* Back to the load-time defaults, then read [SAREK_DEBUG=value]. *)
let configure value =
  Log.disable Log.All ;
  Log.set_level Log.Warn ;
  Unix.putenv "SAREK_DEBUG" value ;
  Log.init_from_env ()

let emits ~expected name f =
  Alcotest.(check bool) name expected (captured f <> "")

let test_component_implies_debug () =
  configure "kernel" ;
  emits ~expected:true "debug on the named component" (fun () ->
      Log.debug Log.Kernel "source") ;
  emits ~expected:true "debugf on the named component" (fun () ->
      Log.debugf Log.Kernel "source %d" 1) ;
  emits ~expected:false "other components stay silent" (fun () ->
      Log.debug Log.Transfer "upload")

let test_all_implies_debug () =
  configure "all" ;
  emits ~expected:true "debug on every component" (fun () ->
      Log.debug Log.Memory "alloc")

let test_explicit_level_wins () =
  configure "kernel,info" ;
  emits ~expected:false "debug below the explicit level" (fun () ->
      Log.debug Log.Kernel "source") ;
  emits ~expected:true "info at the explicit level" (fun () ->
      Log.info Log.Kernel "compiled") ;
  configure "warn,kernel" ;
  emits ~expected:false "level first, then component" (fun () ->
      Log.info Log.Kernel "compiled")

let test_level_alone () =
  configure "debug" ;
  emits ~expected:false "a level enables no component" (fun () ->
      Log.debug Log.Kernel "source")

let test_runtime_switches () =
  configure "" ;
  emits ~expected:false "off by default" (fun () -> Log.warn Log.Device "w") ;
  Log.enable Log.Device ;
  emits ~expected:true "enable" (fun () -> Log.warn Log.Device "w") ;
  emits ~expected:false "level still applies" (fun () ->
      Log.debug Log.Device "d") ;
  Log.set_level Log.Debug ;
  emits ~expected:true "set_level" (fun () -> Log.debug Log.Device "d") ;
  Alcotest.(check bool) "is_enabled" true (Log.is_enabled Log.Device) ;
  Alcotest.(check bool) "others" false (Log.is_enabled Log.Memory) ;
  Log.disable Log.Device ;
  emits ~expected:false "disable" (fun () -> Log.debug Log.Device "d") ;
  Log.enable Log.All ;
  Alcotest.(check bool) "All enables each" true (Log.is_enabled Log.Memory) ;
  Log.disable Log.All ;
  Alcotest.(check bool) "All disables each" false (Log.is_enabled Log.Memory)

let () =
  Alcotest.run
    "Log"
    [
      ( "SAREK_DEBUG",
        [
          Alcotest.test_case
            "component implies debug"
            `Quick
            test_component_implies_debug;
          Alcotest.test_case "all implies debug" `Quick test_all_implies_debug;
          Alcotest.test_case
            "explicit level wins"
            `Quick
            test_explicit_level_wins;
          Alcotest.test_case "level alone" `Quick test_level_alone;
        ] );
      ("switches", [Alcotest.test_case "runtime" `Quick test_runtime_switches]);
    ]
