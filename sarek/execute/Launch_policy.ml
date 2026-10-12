(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Launch-path decisions that report to the explain stream
   (specs/optimisation-policy.md, section 3.5) and have no module of their own:
   whether a launch is deferred by the launch window, and the explicit
   cooperative-matrix gate of the device backends. The reporting costs one
   atomic read when nobody listens: no text is built then. *)

open Sarek_ir_types
module Device = Spoc_core.Device

(* "<framework>:<device id>", the target of every per-device decision. *)
let target (device : Device.t) =
  Printf.sprintf "%s:%d" device.framework device.id

let refusal_text : Window_gate.refusal -> string = function
  | Switched_off -> "switched off (Execute.Window.set_enabled)"
  | Restricted -> Sarek_policy.disabled_reason
  | Cpu_device -> "CPU device (host buffers can alias)"
  | Not_on framework -> "not on " ^ framework
  | Streamed -> "stream launch"
  | Soa_params -> "structure-of-arrays arguments"

(* Whether the launch of [ir] on [device] is deferred by the launch window;
   [stream] is the stream it is enqueued on, if any. Reports the decision. *)
let deferred (ir : kernel) ~(device : Device.t) ~stream ~soa_params =
  let refusal =
    Window_gate.refusal
      ~framework:device.framework
      ~is_cpu:device.capabilities.Spoc_framework.Framework_sig.is_cpu
      ~streamed:(Option.is_some stream)
      ~soa_params
  in
  if Sarek_policy.reporting () then
    Sarek_policy.report
      {
        kernel = ir.kern_name;
        target = target device;
        optimisation = Sarek_policy.Launch_window;
        decision =
          (match refusal with
          | None -> Sarek_policy.Applied Sarek_policy.Exact_by_construction
          | Some why -> Sarek_policy.Refused (refusal_text why));
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
    Option.iter
      (Sarek_policy.report_gate ~kernel:ir.kern_name ~target:(target device))
      (representative
         (List.map
            (Sarek_coopmat.policy_decision_for budget)
            (Sarek_ir_analysis.kernel_coopmat_configs ir)))
