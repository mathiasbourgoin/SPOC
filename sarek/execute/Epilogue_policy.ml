(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Which backends run epilogue fusion, and for which kernels
    (specs/epilogue-fusion.md, FR-104; specs/optimisation-policy.md).

    The fused kernel equals the unfused pair only if no kernel of the backend is
    contracted: the device compiler may share a subexpression of P and C (or
    forward P's stored value) in the fused kernel that the unfused C contracts.
    HIP compiles every kernel with [-ffp-contract=off] and every Metal kernel
    starts with [#pragma METAL fp contract(off)]
    (Sarek_ir_metal.metal_fp_contract_pragma), so fusion is exact there. CUDA,
    OpenCL and Vulkan contract inside kernels (docs/fp-contraction-policy.md
    section 14): there a fused kernel may differ from the pair by one
    contraction per element, a deviation the program accepts for the kernels
    concerned with [Cross_kernel_contraction] ({!Sarek_policy.requires}).

    What this module adds to that decision is a per-backend switch that only
    restricts: every backend is on by default, [SAREK_LAUNCH_EPILOGUE=0] and
    {!set_backend_enabled}[ false] switch it off, and [true] only undoes a
    switch-off; it never grants anything. *)

(* The backends {!Sarek_policy.requires} offers epilogue fusion on;
   [SAREK_LAUNCH_EPILOGUE=0] switches all of them off at start-up. *)
let offered = ["HIP"; "Metal"; "CUDA/PTX"; "CUDA/C"; "CUDA"; "OpenCL"; "Vulkan"]

let switched_off : string list Atomic.t =
  Atomic.make
    (match Sys.getenv_opt "SAREK_LAUNCH_EPILOGUE" with
    | Some ("0" | "false" | "off" | "no") -> offered
    | _ -> [])

(* A backend stays off once its plugin reported that it lost the exactness the
   fusion relies on ({!Spoc_framework_registry.Fp_exactness}: Metal without
   conformant compile options), grant or not: fast math is more than
   contraction. *)
let enabled (backend : string) =
  (not (List.mem backend (Atomic.get switched_off)))
  && Spoc_framework_registry.Fp_exactness.is_exact backend

(** Switch epilogue fusion off ([false]) or back on ([true]) for one backend.
    Switching on is not a grant: on CUDA, OpenCL and Vulkan fusion still needs
    {!Sarek_policy.Cross_kernel_contraction} for both kernels. *)
let set_backend_enabled ~(backend : string) (b : bool) =
  let rec update () =
    let old = Atomic.get switched_off in
    let rest = List.filter (fun x -> x <> backend) old in
    let next = if b then rest else backend :: rest in
    if not (Atomic.compare_and_set switched_off old next) then update ()
  in
  update ()

(** Whether the epilogue of [consumer] may be merged into [producer] on
    [backend]: the backend is not switched off or inexact, and
    {!Sarek_policy.decide} allows {!Sarek_policy.Epilogue_fusion} for the meet
    of both kernels' budgets. *)
let decide_pair ~(backend : string) ~(producer : string) ~(consumer : string) :
    Sarek_policy.decision =
  if List.mem backend (Atomic.get switched_off) then
    Sarek_policy.Refused ("epilogue fusion switched off on " ^ backend)
  else if not (Spoc_framework_registry.Fp_exactness.is_exact backend) then
    Sarek_policy.Refused (backend ^ " lost its floating-point exactness")
  else
    Sarek_policy.decide
      (Sarek_policy.effective_for ~kernels:[producer; consumer])
      Sarek_policy.Epilogue_fusion
      ~backend
