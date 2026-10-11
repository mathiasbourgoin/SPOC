(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Policy of the launch-time matrix rewrite (specs/matrix-pattern-dynamic.md):
   the SAREK_MATRIX_REWRITE parser and the opt-ins of the automatic float
   rewrite. No device needed. The opt-ins are one-way and process-wide, so the
   cases that grant them run last, in order. *)

[@@@warning "-33"]

[@@@alert "-sarek_matrix"]

open Sarek
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

let k_f16 =
  [%kernel
    fun (a : float16 vector)
        (b : float16 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum :=
            sum
            +. float32_of_float16 a.((row * k) + i)
               *. float32_of_float16 b.((i * n) + col)
        done ;
        c.((row * n) + col) <- sum
      end]

(* A device that exists only as a record: the decision is taken before the
   device is used further (no capabilities advertised). *)
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

(* The [~explicit] wiring of [choose] into [lowering_for], without a GPU: the
   accuracy opt-ins are granted by the previous case, the automatic one is not.
   The automatic launch is declined for the missing opt-in; the explicit one
   gets past that check (and stops later, on the device's capabilities). *)
let test_choose_wires_explicit () =
  let ir =
    match (snd k_f16).Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> Alcotest.fail "no IR"
  in
  let decide explicit =
    Spec.choose ~explicit ~device:fake_hip ~ir ~launch:(fun () -> None) ()
  in
  (match decide false with
  | Spec.Original why ->
      Alcotest.(check bool)
        ("automatic: names the opt-in (" ^ why ^ ")")
        true
        (contains ~needle:optin why)
  | Spec.Specialised _ -> Alcotest.fail "automatic: rewritten") ;
  match decide true with
  | Spec.Original why ->
      Alcotest.(check bool)
        ("explicit: not declined for the automatic opt-in (" ^ why ^ ")")
        false
        (contains ~needle:optin why)
  | Spec.Specialised _ -> Alcotest.fail "explicit: rewritten without a device"

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
          Alcotest.test_case
            "choose passes explicit"
            `Quick
            test_choose_wires_explicit;
          Alcotest.test_case "off and exact" `Quick test_modes_still_refuse;
          Alcotest.test_case
            "after the automatic opt-in"
            `Quick
            test_float_after_automatic_optin;
        ] );
    ]
