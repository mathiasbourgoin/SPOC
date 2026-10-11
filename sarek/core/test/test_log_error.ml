(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Log with Spoc_core.Error linked, under SAREK_DEBUG=kernel (set by the dune
 * rule before the program starts, so both modules initialise with it).
 *
 * Linking Error must not turn on the components SAREK_DEBUG did not name, and
 * Error's own messages must still print.
 ******************************************************************************)

module Log = Spoc_core.Log
module Error = Spoc_core.Error

(* Everything [f] prints on stdout. *)
let captured f =
  flush stdout ;
  let file = Filename.temp_file "test_log_error" ".out" in
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

let emits ~expected name f =
  Alcotest.(check bool) name expected (captured f <> "")

let test_named_component_only () =
  Alcotest.(check (option string))
    "SAREK_DEBUG"
    (Some "kernel")
    (Sys.getenv_opt "SAREK_DEBUG") ;
  emits ~expected:true "the named component prints debug" (fun () ->
      Log.debug Log.Kernel "source") ;
  emits ~expected:false "another component stays silent" (fun () ->
      Log.debug Log.Transfer "upload") ;
  Alcotest.(check bool)
    "Transfer not enabled"
    false
    (Log.is_enabled Log.Transfer)

let test_error_messages_print () =
  emits ~expected:true "Error.error prints" (fun () ->
      Error.error "failure %d" 1) ;
  emits ~expected:false "below Error's level: silent" (fun () ->
      Error.info "detail")

let () =
  Alcotest.run
    "Log with Error linked"
    [
      ( "SAREK_DEBUG=kernel",
        [
          Alcotest.test_case
            "named component only"
            `Quick
            test_named_component_only;
          Alcotest.test_case
            "Error messages print"
            `Quick
            test_error_messages_print;
        ] );
    ]
