(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Unit test of the shared backend-failure classifier. Device-free. *)
let () =
  Test_helpers.Backend_failure.self_check ~test_name:"test_backend_failure" ;
  print_endline "test_backend_failure: PASSED"
