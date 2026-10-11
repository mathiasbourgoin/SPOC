(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Policy of the launch-time matrix rewrite (specs/matrix-pattern-dynamic.md):
   the SAREK_MATRIX_REWRITE parser and the opt-ins of the automatic float
   rewrite. No device needed. The opt-ins are one-way and process-wide, so the
   cases that grant them run last, in order. *)

module Spec = Sarek_execute.Matrix_specialise
module Adv = Sarek_matrix_advice
module Rw = Sarek_matrix_rewrite

let mode_testable =
  Alcotest.testable
    (fun fmt m -> Format.pp_print_string fmt (Spec.mode_name m))
    ( = )

let contains ~needle s =
  let n = String.length needle in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = needle || at (i + 1))
  in
  at 0

let test_parse_accepted () =
  List.iter
    (fun (value, expected) ->
      let mode, warning = Spec.parse_mode (Some value) in
      Alcotest.check mode_testable value expected mode ;
      Alcotest.(check (option string)) (value ^ ": no warning") None warning)
    [
      ("off", Spec.Off);
      ("OFF", Spec.Off);
      ("0", Spec.Off);
      ("false", Spec.Off);
      ("False", Spec.Off);
      ("no", Spec.Off);
      ("exact", Spec.Exact);
      ("Exact", Spec.Exact);
      ("auto", Spec.Auto);
      ("ON", Spec.Auto);
      ("1", Spec.Auto);
      ("true", Spec.Auto);
      ("yes", Spec.Auto);
    ]

let test_parse_default () =
  List.iter
    (fun value ->
      let mode, warning = Spec.parse_mode value in
      Alcotest.check mode_testable "default is auto" Spec.Auto mode ;
      Alcotest.(check (option string)) "no warning" None warning)
    [None; Some ""]

let test_parse_bogus () =
  let mode, warning = Spec.parse_mode (Some "bogus") in
  Alcotest.check mode_testable "falls back to auto" Spec.Auto mode ;
  match warning with
  | None -> Alcotest.fail "no warning for an unrecognised value"
  | Some text ->
      List.iter
        (fun needle ->
          Alcotest.(check bool)
            ("warning names " ^ needle)
            true
            (contains ~needle text))
        ["SAREK_MATRIX_REWRITE"; "bogus"; "off"; "exact"; "auto"]

let lowering ?precision ~explicit case =
  Spec.lowering_for ~mode:Spec.Auto ~explicit case precision

let is_ok = function Ok _ -> true | Error _ -> false

let check_refused_naming what needle = function
  | Ok _ -> Alcotest.failf "%s: rewritten" what
  | Error why ->
      Alcotest.(check bool)
        (what ^ ": the reason names " ^ needle ^ " (" ^ why ^ ")")
        true
        (contains ~needle why)

let optin = "accept_automatic_float_rewrite"

(* Before the new opt-in, with the accuracy opt-ins granted. *)
let test_float_needs_automatic_optin () =
  Sarek_coopmat.accept_relaxed_accumulation ~reason:"test" ;
  Sarek_coopmat.accept_operand_rounding
    ~precision:Sarek_coopmat.Operands_f16
    ~reason:"test" ;
  check_refused_naming
    "automatic f16"
    optin
    (lowering ~explicit:false Adv.F16_inputs) ;
  check_refused_naming
    "automatic f32"
    optin
    (lowering ~explicit:false Adv.F32_inputs) ;
  Alcotest.(check bool)
    "int32 needs no opt-in"
    true
    (is_ok (lowering ~explicit:false Adv.Int32_inputs)) ;
  Alcotest.(check bool)
    "explicit f16 needs no automatic opt-in"
    true
    (is_ok (lowering ~explicit:true Adv.F16_inputs)) ;
  Alcotest.(check bool)
    "explicit f32 needs no automatic opt-in"
    true
    (is_ok
       (lowering
          ~explicit:true
          ~precision:Sarek_coopmat.Operands_f16
          Adv.F32_inputs))

let test_modes_still_refuse () =
  List.iter
    (fun mode ->
      Alcotest.(check bool)
        (Spec.mode_name mode ^ " refuses float even when explicit")
        false
        (is_ok (Spec.lowering_for ~mode ~explicit:true Adv.F16_inputs None)))
    [Spec.Off; Spec.Exact]

(* After the new opt-in the automatic path decides as the explicit one. *)
let test_float_after_automatic_optin () =
  Sarek_coopmat.accept_automatic_float_rewrite ~reason:"test" ;
  Alcotest.(check bool)
    "automatic f16"
    true
    (is_ok (lowering ~explicit:false Adv.F16_inputs)) ;
  Alcotest.(check bool)
    "automatic f32 with accepted rounding"
    true
    (is_ok (lowering ~explicit:false Adv.F32_inputs)) ;
  match
    lowering
      ~explicit:false
      ~precision:Sarek_coopmat.Operands_bf16
      Adv.F32_inputs
  with
  | Ok _ -> Alcotest.fail "bf16 rounding was never accepted"
  | Error why ->
      Alcotest.(check bool)
        "an unaccepted precision is still refused"
        true
        (contains ~needle:"accept_operand_rounding" why)

let () =
  Alcotest.run
    "Matrix_specialise policy"
    [
      ( "SAREK_MATRIX_REWRITE",
        [
          Alcotest.test_case "accepted values" `Quick test_parse_accepted;
          Alcotest.test_case "unset or empty" `Quick test_parse_default;
          Alcotest.test_case "unrecognised value" `Quick test_parse_bogus;
        ] );
      ( "opt-ins",
        [
          Alcotest.test_case
            "float needs the automatic opt-in"
            `Quick
            test_float_needs_automatic_optin;
          Alcotest.test_case "off and exact" `Quick test_modes_still_refuse;
          Alcotest.test_case
            "after the automatic opt-in"
            `Quick
            test_float_after_automatic_optin;
        ] );
    ]
