(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The process half of SAREK_OPT_DISABLE and SAREK_EXPLAIN
   (specs/optimisation-policy.md, M3). The variables are read once, when the
   process starts, and every restriction is one-way: one scenario per process,
   chosen by argv.(1), with the environment set by the dune rule that runs it
   (see dune). No device: the decision points are driven through fake devices,
   the pure code generators and the policy's own entry points.

   Scenarios:
   - baseline: no variable; every default, to compare the others against.
   - window, epilogue, matrix, narrowing-proof, unroll: that optimisation is
     taken away and no other.
   - single-rounding, cross-kernel-contraction, automatic-float-rewrite: the
     grant is revoked; a program that grants it is recorded but not obeyed.
   - unknown: an unknown item warns (stderr is compared with a golden) and the
     rest applies.
   - removed: the five replaced variables are set and change nothing.
   - accumulation-order, operand-rounding-f16: the revoked accuracy grant is
     named in the refusals it causes.
   - model: the process state after SAREK_OPT_DISABLE equals [Env.narrow].
   - explain, explain-off, explain-unknown, explain-replaced: SAREK_EXPLAIN. *)


[@@@warning "-33"]

[@@@alert "-sarek_matrix"]

open Sarek_ir_types
open Policy_env_fixtures

(* ------------------------------------------------------------------------ *)
(* Scenarios                                                                 *)
(* ------------------------------------------------------------------------ *)

let test_baseline () =
  check_restricted [] ;
  Alcotest.(check bool) "window defers on HIP" true (deferred fake_hip) ;
  List.iter
    (fun device ->
      Alcotest.(check bool)
        (device.Spoc_core.Device.name ^ ": one function decides")
        (Sarek_execute.Execute.Window.defers_device device)
        (deferred device))
    [fake_hip; fake_native] ;
  Alcotest.(check string)
    "epilogue on HIP"
    "exact"
    (decision_text (epilogue "HIP")) ;
  Alcotest.(check (list int)) "unroll hints" [1; 1; 1; 1] (unroll_hints ()) ;
  Alcotest.(check int) "provable narrowing: no barrier" 0 (barriers t2_f16_sum) ;
  Alcotest.(check bool)
    "unprovable narrowing: barrier"
    true
    (barriers q4k_like > 0) ;
  Alcotest.(check bool)
    "int32 is lowered"
    true
    (is_ok (lowering ~explicit:false Adv.Int32_inputs))

let test_window () =
  check_restricted [Sarek_policy.Launch_window] ;
  Alcotest.(check bool) "no deferral on HIP" false (deferred fake_hip) ;
  Alcotest.(check string)
    "epilogue unaffected"
    "exact"
    (decision_text (epilogue "HIP")) ;
  Alcotest.(check (list int)) "unroll unaffected" [1; 1; 1; 1] (unroll_hints ())

let test_unroll () =
  check_restricted [Sarek_policy.Auto_unroll] ;
  Alcotest.(check (list int))
    "no unroll pragma in any generator"
    [0; 0; 0; 0]
    (unroll_hints ()) ;
  Alcotest.(check bool) "window unaffected" true (deferred fake_hip) ;
  Alcotest.(check int) "narrowing unaffected" 0 (barriers t2_f16_sum)

let test_narrowing_proof () =
  check_restricted [Sarek_policy.Narrowing_proof] ;
  Alcotest.(check bool)
    "barriers kept on a provable kernel"
    true
    (barriers t2_f16_sum > 0) ;
  Alcotest.(check (list int)) "unroll unaffected" [1; 1; 1; 1] (unroll_hints ())

let test_matrix () =
  check_restricted [Sarek_policy.Matrix_rewrite] ;
  (match
     Spec.choose ~device:fake_hip ~ir:gemm_ir ~launch:(fun () -> None) ()
   with
  | Spec.Original why ->
      Alcotest.(check string)
        "nothing rewritten"
        "disabled (SAREK_OPT_DISABLE)"
        why
  | Spec.Specialised _ -> Alcotest.fail "rewritten") ;
  Alcotest.(check bool) "window unaffected" true (deferred fake_hip)

let test_epilogue () =
  check_restricted [Sarek_policy.Epilogue_fusion] ;
  Alcotest.(check string)
    "no epilogue fusion on HIP"
    disabled_text
    (decision_text (epilogue "HIP")) ;
  Alcotest.(check bool) "window unaffected" true (deferred fake_hip)

(* The program grants single rounding for q4k_like; the process revokes it. *)
let test_single_rounding () =
  check_restricted [] ;
  ok_or_fail
    (Sarek_f16_policy.accept_single_rounding
       (Sarek_f16_policy.Kernel "q4k_like")
       ~reason:"the program accepts it") ;
  Alcotest.(check bool) "revoked" true (Sarek_policy.revoked single) ;
  Alcotest.(check (option string))
    "the reader sees no grant"
    None
    (Sarek_policy.reason ~kernel:"q4k_like" single) ;
  Alcotest.(check bool)
    "the budget holds none"
    false
    (Sarek_policy.Budget.permits
       (Sarek_policy.budget ~kernels:["q4k_like"])
       single) ;
  Alcotest.(check bool)
    "barriers kept: the grant has no effect"
    true
    (barriers q4k_like > 0) ;
  Alcotest.(check int) "provable kernel unchanged" 0 (barriers t2_f16_sum) ;
  let events = ref [] in
  Sarek_policy.set_reporter (Some (fun e -> events := e :: !events)) ;
  ignore (cuda_c q4k_like) ;
  Sarek_policy.set_reporter None ;
  let lines = List.rev_map Sarek_policy.Env.explain_line !events in
  Alcotest.(check bool)
    "explain says the grant is revoked"
    true
    (List.mem
       "sarek-explain q4k_like - policy refused grant-revoked single-rounding"
       lines)

let test_contraction () =
  ok_or_fail
    (Sarek_policy.grant
       Sarek_policy.Process
       contraction
       ~reason:"the program accepts one contraction") ;
  Alcotest.(check bool) "revoked" true (Sarek_policy.revoked contraction) ;
  Alcotest.(check string)
    "no epilogue on OpenCL despite the grant"
    "refused: grant cross-kernel-contraction revoked (SAREK_OPT_DISABLE)"
    (decision_text (epilogue "OpenCL")) ;
  Alcotest.(check string)
    "no epilogue on Vulkan either"
    "refused: grant cross-kernel-contraction revoked (SAREK_OPT_DISABLE)"
    (decision_text (epilogue "Vulkan")) ;
  Alcotest.(check string)
    "HIP needs no grant"
    "exact"
    (decision_text (epilogue "HIP"))

let test_automatic_float_rewrite () =
  Cm.accept_relaxed_accumulation ~reason:"env scenario" ;
  Cm.accept_operand_rounding ~precision:Cm.Operands_f16 ~reason:"env scenario" ;
  Cm.accept_automatic_float_rewrite ~reason:"env scenario" ;
  Alcotest.(check bool) "revoked" true (Sarek_policy.revoked automatic) ;
  (match lowering ~explicit:false Adv.F16_inputs with
  | Ok _ -> Alcotest.fail "f16 rewritten automatically despite the revocation"
  | Error why ->
      Alcotest.(check bool)
        ("the reason names the variable (" ^ why ^ ")")
        true
        (contains ~needle:"SAREK_OPT_DISABLE" why)) ;
  Alcotest.(check bool)
    "f32 not rewritten automatically either"
    false
    (is_ok (lowering ~explicit:false ~precision:Cm.Operands_f16 Adv.F32_inputs)) ;
  Alcotest.(check bool)
    "int32 is still rewritten"
    true
    (is_ok (lowering ~explicit:false Adv.Int32_inputs)) ;
  Alcotest.(check bool)
    "an explicit request needs no consent"
    true
    (is_ok (lowering ~explicit:true Adv.F16_inputs)) ;
  Alcotest.(check bool)
    "the accuracy opt-ins are untouched"
    true
    (Cm.relaxed_accumulation_reason () <> None)

let test_unknown () =
  check_restricted [Sarek_policy.Launch_window; Sarek_policy.Auto_unroll] ;
  Alcotest.(check bool) "window off" false (deferred fake_hip) ;
  Alcotest.(check (list int)) "unroll off" [0; 0; 0; 0] (unroll_hints ()) ;
  Alcotest.(check string) "epilogue on" "exact" (decision_text (epilogue "HIP"))


(* Revoking an accuracy grant is named in the refusals it causes. *)
let coopmat_cfg =
  match Sarek_ir_analysis.kernel_coopmat_configs coopmat_kernel with
  | [cfg] -> cfg
  | _ -> failwith "one configuration"

let refusal_text = function
  | Sarek_capability.Unavailable cap -> cap.Sarek_capability.cap_why
  | Sarek_capability.Available -> "available"
  | Sarek_capability.Unknown why -> why

let test_accumulation_order_revoked () =
  Cm.accept_relaxed_accumulation ~reason:"the program accepts it" ;
  Alcotest.(check bool)
    "revoked"
    true
    (Sarek_policy.revoked (Sarek_policy.Deviation Sarek_policy.Accumulation_order)) ;
  let why =
    refusal_text
      (Cm.policy_verdict_for
         (Sarek_policy.budget ~kernels:[coopmat_kernel.kern_name])
         coopmat_cfg)
  in
  Alcotest.(check bool)
    ("an explicit coopmat kernel is refused and the refusal says why (" ^ why ^ ")")
    true
    (contains ~needle:"revoked by SAREK_OPT_DISABLE" why) ;
  Alcotest.(check bool)
    "the float lowering is refused too"
    false
    (is_ok (lowering ~explicit:true Adv.F16_inputs)) ;
  Alcotest.(check bool)
    "int32 is still rewritten"
    true
    (is_ok (lowering ~explicit:false Adv.Int32_inputs))

let test_operand_rounding_revoked () =
  Cm.accept_operand_rounding ~precision:Cm.Operands_f16 ~reason:"the program" ;
  match lowering ~explicit:true ~precision:Cm.Operands_f16 Adv.F32_inputs with
  | Ok _ -> Alcotest.fail "rounded to f16 despite the revocation"
  | Error why ->
      Alcotest.(check bool)
        ("the error names the revocation (" ^ why ^ ")")
        true
        (contains ~needle:"revoked by SAREK_OPT_DISABLE" why)

(* The process state equals the pure model: every grant at process scope, then
   the effective policy of a kernel against [Env.narrow] of the parsed value
   applied to the full budget. *)
let test_model () =
  List.iter
    (fun g ->
      ok_or_fail
        (Sarek_policy.grant Sarek_policy.Process g ~reason:("model " ^ Sarek_policy.grant_name g)))
    Sarek_policy.all_grants ;
  let full =
    List.fold_left
      (fun b g ->
        match Sarek_policy.Budget.add g ~reason:"model" b with
        | Ok b -> b
        | Error _ -> Alcotest.fail "add refused")
      Sarek_policy.Budget.exact
      Sarek_policy.all_grants
  in
  let restrictions, _ = Sarek_policy.Env.parse "window,Single-Rounding, matrix" in
  let modelled =
    Sarek_policy.Env.narrow
      restrictions
      {Sarek_policy.budget = full; disabled = []}
  in
  Alcotest.(check string)
    "apply_environment equals narrow"
    (Sarek_policy.effective_digest modelled)
    (Sarek_policy.effective_digest (Sarek_policy.effective ~kernel:"model_k"))

(* The five replaced variables are set (dune): none of them does anything. *)
let test_removed () =
  check_restricted [] ;
  Alcotest.(check bool) "window defers" true (deferred fake_hip) ;
  Alcotest.(check string) "epilogue on" "exact" (decision_text (epilogue "HIP")) ;
  Alcotest.(check (list int)) "unroll hints" [1; 1; 1; 1] (unroll_hints ()) ;
  Cm.accept_relaxed_accumulation ~reason:"env scenario" ;
  Cm.accept_operand_rounding ~precision:Cm.Operands_f16 ~reason:"env scenario" ;
  Cm.accept_automatic_float_rewrite ~reason:"env scenario" ;
  Alcotest.(check bool)
    "float rewrite allowed (the removed mode variable is ignored)"
    true
    (is_ok (lowering ~explicit:false Adv.F16_inputs)) ;
  match
    Spec.choose ~device:fake_hip ~ir:gemm_ir ~launch:(fun () -> None) ()
  with
  | Spec.Original why ->
      Alcotest.(check bool)
        ("not switched off (" ^ why ^ ")")
        false
        (contains ~needle:"switched off" why
        || contains ~needle:"SAREK_OPT_DISABLE" why)
  | Spec.Specialised _ -> Alcotest.fail "rewritten without a device"

(* ------------------------------------------------------------------------ *)
(* SAREK_EXPLAIN                                                             *)
(* ------------------------------------------------------------------------ *)

let test_explain_off () =
  Alcotest.(check bool) "no reporter" false (Sarek_policy.reporting ()) ;
  ignore (cuda_c unroll_loop) ;
  ignore (deferred fake_hip)

let test_explain_replaced () =
  Alcotest.(check bool)
    "the default reporter is installed"
    true
    (Sarek_policy.reporting ()) ;
  let seen = ref 0 in
  Sarek_policy.set_reporter (Some (fun _ -> incr seen)) ;
  ignore (cuda_c unroll_loop) ;
  Alcotest.(check int) "the program's reporter gets the record" 1 !seen

(* The launch gate of Execute, on a device that cannot run the kernel: it
   raises, after the policy half of the gate has been reported. *)
let gate_through_execute () =
  match
    Sarek_execute.Execute.check_device_capabilities
      ~device:fake_hip
      coopmat_kernel
  with
  | () -> Alcotest.fail "a device without cooperative matrices accepted it"
  | exception _ -> ()

(* Every decision point that needs no device, once; the lines go to stderr and
   are compared with explain.expected. *)
let test_explain_golden () =
  Alcotest.(check bool) "reporting" true (Sarek_policy.reporting ()) ;
  (* codegen: unroll per generator, narrowing per kernel *)
  List.iter
    (fun gen -> ignore (gen unroll_loop))
    [cuda_c; cuda_c; opencl; metal; glsl] ;
  ignore (cuda_c t2_f16_sum) ;
  ignore (cuda_c q4k_like) ;
  ok_or_fail
    (Sarek_f16_policy.accept_single_rounding
       (Sarek_f16_policy.Kernel "q4k_like")
       ~reason:"profile fast") ;
  ignore (cuda_c q4k_like) ;
  (* the launch window: deferred, on a CPU device, on a stream *)
  ignore (deferred ~kernel:"env_gemm" fake_hip) ;
  ignore (deferred ~kernel:"env_gemm" fake_hip) ;
  ignore (deferred ~kernel:"env_gemm" fake_native) ;
  ignore (deferred ~kernel:"env_gemm" ~streamed:true fake_hip) ;
  ignore (deferred ~kernel:"env_gemm" ~soa_params:["v"] fake_hip) ;
  (* the matrix rewrite: declined for the missing consent *)
  ignore (Spec.choose ~device:fake_hip ~ir:gemm_ir ~launch:(fun () -> None) ()) ;
  (* the explicit coopmat gate: refused, then granted *)
  gate_through_execute () ;
  Cm.accept_relaxed_accumulation ~reason:"profile fast" ;
  gate_through_execute () ;
  (* a grant on a generated name never counts, and the report says so *)
  ok_or_fail
    (Sarek_policy.grant
       (Sarek_policy.Kernel "env_fused_1")
       single
       ~reason:"on a generated name") ;
  Sarek_policy.register_fused ~name:"env_fused_1" ~members:["env_gemm"] ;
  ignore (deferred ~kernel:"env_fused_1" fake_hip)

(* ------------------------------------------------------------------------ *)

let scenarios =
  let case name f = Alcotest.test_case name `Quick f in
  [
    ("baseline", [case "defaults" test_baseline]);
    ("window", [case "window" test_window]);
    ("epilogue", [case "epilogue" test_epilogue]);
    ("matrix", [case "matrix" test_matrix]);
    ("narrowing-proof", [case "narrowing-proof" test_narrowing_proof]);
    ("unroll", [case "unroll" test_unroll]);
    ("single-rounding", [case "single-rounding revoked" test_single_rounding]);
    ( "cross-kernel-contraction",
      [case "cross-kernel-contraction revoked" test_contraction] );
    ( "automatic-float-rewrite",
      [case "automatic-float-rewrite revoked" test_automatic_float_rewrite] );
    ( "accumulation-order",
      [case "accumulation-order revoked" test_accumulation_order_revoked] );
    ( "operand-rounding-f16",
      [case "operand-rounding-f16 revoked" test_operand_rounding_revoked] );
    ("model", [case "model of the environment" test_model]);
    ("unknown", [case "unknown item" test_unknown]);
    ("removed", [case "replaced variables" test_removed]);
    ("explain", [case "golden" test_explain_golden]);
    ("explain-off", [case "off" test_explain_off]);
    ("explain-unknown", [case "off" test_explain_off]);
    ("explain-replaced", [case "replaced" test_explain_replaced]);
  ]

let () =
  match Array.to_list Sys.argv with
  | [_; scenario] when List.mem_assoc scenario scenarios ->
      (* Verbose: Alcotest otherwise captures stderr, which the golden
         scenarios compare. *)
      Alcotest.run
        ~verbose:true
        ~argv:[|Sys.argv.(0)|]
        "policy_env"
        [(scenario, List.assoc scenario scenarios)]
  | _ ->
      prerr_endline "usage: test_policy_env <scenario>" ;
      exit 2
