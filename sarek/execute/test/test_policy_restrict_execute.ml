(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Process restrictions of the launch path (specs/optimisation-policy.md, M2):
   [Sarek_policy.restrict] of [Launch_window], [Matrix_rewrite] and
   [Epilogue_fusion] switches that optimisation off, and only that one.

   A restriction is one-way and process-wide: its own executable, cases in
   order (the defaults first, then each restriction). No device. *)

module LW = Sarek_execute.Launch_window
module LE = Sarek_execute.Launch_epilogue
module Spec = Sarek_execute.Matrix_specialise

let decision_text = function
  | Sarek_policy.Applied Sarek_policy.Exact_by_construction -> "exact"
  | Sarek_policy.Applied (Sarek_policy.Granted {reason; _}) ->
      "granted: " ^ reason
  | Sarek_policy.Applied (Sarek_policy.Proof p) -> "proof: " ^ p
  | Sarek_policy.Refused why -> "refused: " ^ why

let disabled = "refused: disabled (SAREK_OPT_DISABLE)"

let epilogue () =
  decision_text (LE.decide_pair ~backend:"HIP" ~producer:"p" ~consumer:"c")

let fake_hip : Spoc_core.Device.t =
  {
    id = 0;
    backend_id = 0;
    name = "fake HIP";
    framework = "HIP";
    capabilities =
      {
        max_threads_per_block = 256;
        max_block_dims = (256, 256, 64);
        max_grid_dims = (65535, 65535, 65535);
        shared_mem_per_block = 16384;
        total_global_mem = 1073741824L;
        compute_capability = (0, 0);
        device_features = [];
        coopmat = None;
        supports_atomics = false;
        warp_size = 32;
        max_registers_per_block = 0;
        clock_rate_khz = 0;
        multiprocessor_count = 1;
        is_cpu = false;
      };
  }

let matrix_reason () =
  match
    Spec.choose
      ~device:fake_hip
      ~ir:Sarek_ir_types.default_kernel
      ~launch:(fun () -> None)
      ()
  with
  | Spec.Original why -> why
  | Spec.Specialised _ -> Alcotest.fail "rewritten"

let test_defaults () =
  Alcotest.(check bool) "window defers on HIP" true (LW.defers "HIP") ;
  Alcotest.(check string) "epilogue on HIP" "exact" (epilogue ()) ;
  Alcotest.(check bool)
    "matrix: not disabled"
    false
    (matrix_reason () = "disabled (SAREK_OPT_DISABLE)")

let test_window () =
  Sarek_policy.restrict Sarek_policy.Launch_window ;
  Alcotest.(check bool) "window does not defer" false (LW.defers "HIP") ;
  Alcotest.(check bool)
    "every backend"
    false
    (List.exists LW.defers LW.default_backends) ;
  Alcotest.(check string) "epilogue unaffected" "exact" (epilogue ())

let test_matrix () =
  Sarek_policy.restrict Sarek_policy.Matrix_rewrite ;
  Alcotest.(check string)
    "matrix refused"
    "disabled (SAREK_OPT_DISABLE)"
    (matrix_reason ()) ;
  Alcotest.(check string) "epilogue unaffected" "exact" (epilogue ())

let test_epilogue () =
  Sarek_policy.restrict Sarek_policy.Epilogue_fusion ;
  Alcotest.(check string) "epilogue refused" disabled (epilogue ())

let () =
  let case name f = Alcotest.test_case name `Quick f in
  Alcotest.run
    "policy_restrict_execute"
    [
      ( "restrictions (in order)",
        [
          case "defaults" test_defaults;
          case "launch window" test_window;
          case "matrix rewrite" test_matrix;
          case "epilogue fusion" test_epilogue;
        ] );
    ]
