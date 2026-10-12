(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* SAREK_OPT_DISABLE=epilogue (set by the dune rule) switches epilogue fusion off
   for the process, one-way: a later [set_epilogue_backend ~backend true] does
   not bring it back (specs/epilogue-fusion.md FR-104). *)

module LE = Sarek_execute.Launch_epilogue

let refused = function
  | Sarek_policy.Refused _ -> true
  | Sarek_policy.Applied _ -> false

let test_env_off_is_one_way () =
  Alcotest.(check bool)
    "the variable restricts fusion"
    true
    (Sarek_policy.restricted Sarek_policy.Epilogue_fusion) ;
  List.iter
    (fun backend ->
      LE.set_backend_enabled ~backend true ;
      Alcotest.(check bool)
        (backend ^ ": refused after set_backend_enabled true")
        true
        (refused (LE.decide_pair ~backend ~producer:"env_p" ~consumer:"env_c")) ;
      Alcotest.(check bool)
        (backend ^ ": not available")
        false
        (LE.available backend))
    ["HIP"; "Metal"; "OpenCL"]

let () =
  Alcotest.run
    "epilogue env off"
    [
      ( "SAREK_OPT_DISABLE=epilogue",
        [Alcotest.test_case "one-way" `Quick test_env_off_is_one_way] );
    ]
