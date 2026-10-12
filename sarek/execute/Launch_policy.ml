(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Launch-path decisions that report to the explain stream
   (specs/optimisation-policy.md, section 3.5) and have no module of their own:
   whether a launch is deferred by the launch window, and the explicit
   cooperative-matrix gate. The reporting is one atomic read when nobody
   listens. *)

open Sarek_ir_types
module Device = Spoc_core.Device

(* Why [device] does not defer a launch, in the order the reasons are checked;
   [None] when it does. *)
let window_refusal ~(device : Device.t) ~streamed ~soa_params =
  let framework = device.framework in
  if not (Atomic.get Launch_window.enabled) then
    Some "switched off (Execute.Window.set_enabled)"
  else if Sarek_policy.restricted Sarek_policy.Launch_window then
    Some Sarek_policy.disabled_reason
  else if device.capabilities.Spoc_framework.Framework_sig.is_cpu then
    Some "CPU device (host buffers can alias)"
  else if not (List.mem framework (Atomic.get Launch_window.backends)) then
    Some ("not on " ^ framework)
  else if streamed then Some "stream launch"
  else if soa_params <> [] then Some "structure-of-arrays arguments"
  else None

let deferred ~kernel ~(device : Device.t) ~streamed ~soa_params =
  let refusal = window_refusal ~device ~streamed ~soa_params in
  if Sarek_policy.reporting () then
    Sarek_policy.report
      {
        kernel;
        device = device.name;
        optimisation = Sarek_policy.Launch_window;
        decision =
          (match refusal with
          | None -> Sarek_policy.Applied Sarek_policy.Exact_by_construction
          | Some why -> Sarek_policy.Refused why);
      } ;
  Option.is_none refusal

(* One decision for the kernel: the first refusal, else the first that rests on
   a grant, else the first. *)
let representative = function
  | [] -> None
  | first :: _ as decisions ->
      let refused = function
        | Sarek_policy.Refused _ -> true
        | Sarek_policy.Applied _ -> false
      and granted = function
        | Sarek_policy.Applied (Sarek_policy.Granted _) -> true
        | Sarek_policy.Applied _ | Sarek_policy.Refused _ -> false
      in
      Some
        (match List.find_opt refused decisions with
        | Some d -> d
        | None -> Option.value ~default:first (List.find_opt granted decisions))

let coopmat_gate ~(device : Device.t) (ir : kernel) =
  if Sarek_policy.reporting () then
    let budget = Sarek_policy.budget ~kernels:[ir.kern_name] in
    let decisions =
      List.map
        (Sarek_coopmat.policy_decision_for budget)
        (Sarek_ir_analysis.kernel_coopmat_configs ir)
    in
    Option.iter
      (fun decision ->
        Sarek_policy.report
          {
            kernel = ir.kern_name;
            device = device.name;
            optimisation = Sarek_policy.Matrix_rewrite;
            decision;
          })
      (representative decisions)
