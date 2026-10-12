(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Policy of the launch-time matrix rewrite (specs/matrix-pattern-dynamic.md):
   the opt-ins of the automatic float rewrite. No device needed. The opt-ins are one-way and process-wide, so the
   cases that grant them run last, in order. *)

[@@@warning "-33"]

[@@@alert "-sarek_matrix"]

open Sarek
module Spec = Sarek_execute.Matrix_specialise
module Adv = Sarek_matrix_advice
module Rw = Sarek_matrix_rewrite

let contains ~needle s =
  let n = String.length needle in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = needle || at (i + 1))
  in
  at 0

let lowering ?precision ~explicit case =
  Spec.lowering_for
    ~budget:(Sarek_policy.budget ~kernels:[])
    ~explicit
    case
    precision

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

(* The evidence of a rewrite: the grants it rests on (the consent only on the
   automatic path), the program's reasons once each, the description of what
   was done as the trailing lowering field. *)
let test_evidence () =
  let cfg =
    {
      Sarek_coopmat.cfg_shape = {Sarek_coopmat.m = 16; n = 16; k = 16};
      cfg_a = Sarek_coopmat.Float16;
      cfg_b = Sarek_coopmat.Float16;
      cfg_c = Sarek_coopmat.Float32;
      cfg_result = Sarek_coopmat.Float32;
      cfg_saturating = false;
      cfg_scope = Sarek_coopmat.Subgroup;
    }
  in
  let budget = Sarek_policy.budget ~kernels:[] in
  let described lowering =
    Printf.sprintf
      "%s, %s, %s"
      (Rw.lowering_name lowering)
      (Spec.variant_name Spec.Staged)
      (Sarek_coopmat.config_name cfg)
  in
  let evidence ~explicit ~case lowering =
    Spec.evidence_for ~budget ~explicit ~case lowering Spec.Staged cfg
  in
  let granted grants lowering =
    Sarek_policy.Granted
      {grants; reason = "test"; lowering = Some (described lowering)}
  in
  let acc = Sarek_policy.Deviation Sarek_policy.Accumulation_order in
  let consent = Sarek_policy.Consent Sarek_policy.Automatic_float_rewrite in
  let rounding =
    Sarek_policy.Deviation
      (Sarek_policy.Operand_rounding Sarek_coopmat.Operands_f16)
  in
  Alcotest.(check bool)
    "automatic f16: the consent and the accumulation order, one reason"
    true
    (evidence ~explicit:false ~case:Adv.F16_inputs Rw.F16_operands
    = granted [consent; acc] Rw.F16_operands) ;
  Alcotest.(check bool)
    "explicit f16: no consent"
    true
    (evidence ~explicit:true ~case:Adv.F16_inputs Rw.F16_operands
    = granted [acc] Rw.F16_operands) ;
  Alcotest.(check bool)
    "explicit f32 rounded to f16: the rounding too"
    true
    (evidence ~explicit:true ~case:Adv.F32_inputs Rw.F16_operands
    = granted [acc; rounding] Rw.F16_operands) ;
  Alcotest.(check bool)
    "int32: a proof, bit-identical"
    true
    (evidence ~explicit:false ~case:Adv.Int32_inputs Rw.Int32_byte_planes
    = Sarek_policy.Proof
        (described Rw.Int32_byte_planes ^ ": bit-identical modulo 2^32"))

let () =
  Alcotest.run
    "Matrix_specialise policy"
    [
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
          Alcotest.test_case
            "after the automatic opt-in"
            `Quick
            test_float_after_automatic_optin;
          Alcotest.test_case "evidence of a rewrite" `Quick test_evidence;
        ] );
    ]
