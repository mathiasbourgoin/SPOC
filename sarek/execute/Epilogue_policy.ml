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
    restricts: every backend is on by default, {!set_backend_enabled}[ false]
    switches one off and [true] only undoes that; [SAREK_LAUNCH_EPILOGUE=0]
    restricts epilogue fusion for the whole process, for good. Neither grants
    anything. *)

(* Backends switched off with {!set_backend_enabled}; only a switch-off made
   there is undone by [true]. [SAREK_LAUNCH_EPILOGUE=0] is not in this list: it
   is a process restriction of {!Sarek_policy.Epilogue_fusion} (one-way, like
   [SAREK_OPT_DISABLE=epilogue] in M3), so no later [true] brings fusion back. *)
let switched_off : string list Atomic.t = Atomic.make []

let () =
  match Sys.getenv_opt "SAREK_LAUNCH_EPILOGUE" with
  | Some ("0" | "false" | "off" | "no") ->
      Sarek_policy.restrict Sarek_policy.Epilogue_fusion
  | _ -> ()

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

(* The policy half of the decision per (backend, producer, consumer), valid
   for one [Sarek_policy.version]: a repeated submit of a pair computes no
   effective policy. The switch-off list and the exactness mark are not in the
   version and are read at every call. *)
let pair_cache :
    (string * string * string, int * Sarek_policy.decision) Hashtbl.t =
  Hashtbl.create 32

let pair_lock = Mutex.create ()

let policy_decision ~backend ~producer ~consumer =
  let key = (backend, producer, consumer) in
  (* The version is read before the policy, as {!Sarek_policy.version} asks. *)
  let version = Sarek_policy.version () in
  match Mutex.protect pair_lock (fun () -> Hashtbl.find_opt pair_cache key) with
  | Some (v, decision) when v = version -> decision
  | Some _ | None ->
      let decision =
        Sarek_policy.decide
          (Sarek_policy.effective_for ~kernels:[producer; consumer])
          Sarek_policy.Epilogue_fusion
          ~backend
      in
      Mutex.protect pair_lock (fun () ->
          if Hashtbl.length pair_cache >= 4096 then Hashtbl.reset pair_cache ;
          Hashtbl.replace pair_cache key (version, decision)) ;
      decision

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
  else policy_decision ~backend ~producer ~consumer

(** Whether fusion can run on [backend] under the process grants alone: not
    switched off, exact, and allowed by {!Sarek_policy.decide} for the process
    budget (HIP and Metal; the others once [Cross_kernel_contraction] is granted
    at process scope). Kernel-scoped grants are not seen: the decision per pair
    is {!decide_pair}. *)
let available (backend : string) =
  enabled backend
  &&
  match
    Sarek_policy.decide
      (Sarek_policy.effective_for ~kernels:[])
      Sarek_policy.Epilogue_fusion
      ~backend
  with
  | Sarek_policy.Applied _ -> true
  | Sarek_policy.Refused _ -> false
