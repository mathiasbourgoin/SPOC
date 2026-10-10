(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative matrix: the DEVICE-facing half (backlog-62 slice 2).

    The vocabulary lives in {!Sarek_coopmat_types}, which this module re-exports
    wholesale — see there for why the split exists. What is left here is exactly
    the part that consults {!Sarek_capability}: given a device's probed support,
    is a requested configuration permitted? *)

include Sarek_coopmat_types

let device_lacks_config ?(evidence = "") cfg =
  {
    Sarek_capability.cap_name = "cooperative-matrix";
    (* Device_optional, per docs/design/capability-model.md §2: the backend can
       spell the instruction, a given device may not provide it. The local box
       has one device that does and one that does not under the SAME driver
       (Vulkan: radv; HIP: ROCm), which is why a driver-keyed or backend-keyed
       refusal would be wrong here. *)
    cap_kind = Sarek_capability.Device_optional;
    cap_why =
      Printf.sprintf
        "the device offers no cooperative-matrix configuration matching %s%s"
        (config_name cfg)
        (if evidence = "" then "" else " (" ^ evidence ^ ")");
    cap_evidence =
      Sarek_capability.Measured
        "Vulkan: AMD Radeon RX 7900 XTX (RADV NAVI31) advertises \
         VK_KHR_cooperative_matrix revision 2 with 14 configurations; the AMD \
         Ryzen 9 7950X iGPU (RADV RAPHAEL_MENDOCINO) does not, under the same \
         radv / Mesa 26.1.4-arch3.1. HIP: the RX 7900 XTX (gfx1100) executes \
         the gfx11 WMMA instructions and the iGPU (gfx1036, RDNA2) has none, \
         under ROCm 7.2.4 (specs/coopmat-hip-wmma.md)";
    cap_remedy =
      Some
        "Use an ordinary multiply-accumulate kernel, or select a device that \
         offers this cooperative-matrix configuration.";
  }

let verdict ~support cfg =
  match support with
  | None ->
      Sarek_capability.Unknown
        "device cooperative-matrix support was not probed, so no configuration \
         can be confirmed"
  | Some s ->
      if
        List.exists
          (fun advertised ->
            config_matches
              ~shape:cfg.cfg_shape
              ~a:cfg.cfg_a
              ~b:cfg.cfg_b
              ~c:cfg.cfg_c
              ~result:cfg.cfg_result
              advertised
            && advertised.cfg_saturating = cfg.cfg_saturating
            && advertised.cfg_scope = cfg.cfg_scope)
          s.ds_configs
      then Sarek_capability.Available
      else
        Sarek_capability.Unavailable
          (device_lacks_config ~evidence:s.ds_evidence cfg)

(* ------------------------------------------------------------------------ *)
(* The accuracy policy: docs/design/f16-relaxed-accuracy.md §6.1             *)
(* ------------------------------------------------------------------------ *)

(* One-way, process-wide. An [Atomic] rather than a [ref] because launches run
   on several domains and the value is read on every capability check; there is
   deliberately no way to withdraw the opt-in, so a capability check that passed
   (and that Execute memoises) can never become stale. *)
let relaxed_reason : string option Atomic.t = Atomic.make None

let accept_relaxed_accumulation ~reason =
  if String.trim reason = "" then
    invalid_arg
      "Sarek_coopmat.accept_relaxed_accumulation: the reason must not be empty \
       (it is the record that the bounded deviation was understood)" ;
  (* First reason wins: a second call does not overwrite the record. *)
  ignore (Atomic.compare_and_set relaxed_reason None (Some reason))

let relaxed_accumulation_reason () = Atomic.get relaxed_reason

(* The one float configuration a bound has been derived for: f16 x f16 + f32
   -> f32 (§5.2). Products are exact in binary32 and only the order of the
   additions is free. *)
let has_derived_bound cfg =
  cfg.cfg_a = Float16 && cfg.cfg_b = Float16 && cfg.cfg_c = Float32
  && cfg.cfg_result = Float32 && not cfg.cfg_saturating

let relaxed_not_accepted cfg =
  {
    Sarek_capability.cap_name = "coopmat-relaxed-accumulation";
    cap_kind = Sarek_capability.Policy;
    cap_why =
      Printf.sprintf
        "%s accumulates in floating point in an order the hardware chooses, so \
         its result is only BOUNDED (|D - exact| <= gamma_n * (sum|A*B| + \
         |C|)), not bit-identical to the interpreter, and the program has not \
         accepted that"
        (config_name cfg);
    cap_evidence =
      Sarek_capability.Quoted
        "docs/design/f16-relaxed-accuracy.md §5 (the bound) and §6.1 (a \
         bounded deviation needs an explicit opt-in), decided 2026-07-27";
    cap_remedy =
      Some
        "Call Sarek_coopmat.accept_relaxed_accumulation ~reason:\"...\" once \
         before launching, or use an integer configuration (exact).";
  }

let no_derived_bound cfg =
  {
    Sarek_capability.cap_name = "coopmat-float-accumulation";
    cap_kind = Sarek_capability.Policy;
    cap_why =
      Printf.sprintf
        "%s has no derived error bound: only f16*f16+f32->f32 has one (an f16 \
         result rounds every partial sum to binary16; a float operand with an \
         integer accumulator has no defined meaning)"
        (config_name cfg);
    cap_evidence =
      Sarek_capability.Quoted "docs/design/f16-relaxed-accuracy.md §5.4";
    cap_remedy = Some "Accumulate in f32 (f16*f16+f32->f32).";
  }

(* Keyed on ALL FOUR component types, not on {!regime}: [regime] reads only
   the addend and the result, so a float-operand configuration with an integer
   accumulator would come out [Strict] — a configuration no device offers and
   whose meaning (convert a float operand to an integer?) nothing defines. *)
let policy_verdict cfg =
  let all_integer =
    List.for_all
      component_is_integer
      [cfg.cfg_a; cfg.cfg_b; cfg.cfg_c; cfg.cfg_result]
  in
  if all_integer then Sarek_capability.Available
  else if not (has_derived_bound cfg) then
    Sarek_capability.Unavailable (no_derived_bound cfg)
  else if Option.is_some (relaxed_accumulation_reason ()) then
    Sarek_capability.Available
  else Sarek_capability.Unavailable (relaxed_not_accepted cfg)

(* ------------------------------------------------------------------------ *)
(* Fragments without a multiply-add                                         *)
(* ------------------------------------------------------------------------ *)

let fragment_name f =
  Printf.sprintf
    "%s %s fragment of %s (%s scope)"
    (component_name f.frag_component)
    (use_name f.frag_use)
    (shape_name f.frag_shape)
    (scope_name f.frag_scope)

let device_lacks_fragment ?(evidence = "") f =
  {
    Sarek_capability.cap_name = "cooperative-matrix";
    cap_kind = Sarek_capability.Device_optional;
    cap_why =
      Printf.sprintf
        "the device offers no cooperative-matrix configuration with a %s%s"
        (fragment_name f)
        (if evidence = "" then "" else " (" ^ evidence ^ ")");
    cap_evidence =
      Sarek_capability.Quoted
        "a fragment is the per-lane layout of one operand of an advertised \
         configuration; a device that offers none has no layout to load it \
         with (specs/coopmat-hip-wmma.md)";
    cap_remedy =
      Some "Select a device that offers a configuration with this fragment.";
  }

let fragment_verdict ~support f =
  match support with
  | None ->
      Sarek_capability.Unknown
        "device cooperative-matrix support was not probed, so no fragment \
         layout can be confirmed"
  | Some s ->
      let offered cfg =
        let a, b, c, r = fragments_of_config cfg in
        List.mem f [a; b; c; r]
      in
      if List.exists offered s.ds_configs then Sarek_capability.Available
      else
        Sarek_capability.Unavailable
          (device_lacks_fragment ~evidence:s.ds_evidence f)
