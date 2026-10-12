(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

include Sarek_policy_vocab

(* Position in [all_grants]: the order of [Budget.grants] and [Budget.digest].
   Total, and checked against [all_grants] by the laws test. *)
let rank = function
  | Deviation Accumulation_order -> 0
  | Deviation (Operand_rounding Operands_f16) -> 1
  | Deviation (Operand_rounding Operands_bf16) -> 2
  | Deviation (Operand_rounding Operands_bf16x3) -> 3
  | Deviation Single_rounding -> 4
  | Deviation Cross_kernel_contraction -> 5
  | Consent Automatic_float_rewrite -> 6

let is_blank reason = String.trim reason = ""

module type BUDGET = sig
  type t

  val exact : t

  val add : grant -> reason:string -> t -> (t, refusal) result

  val permits : t -> grant -> bool

  val reason : t -> grant -> string option

  val meet : t -> t -> t

  val leq : t -> t -> bool

  val grants : t -> (grant * string) list

  val digest : t -> string
end

module Budget_impl = struct
  (* Invariant: sorted by [rank], at most one entry per grant. *)
  type t = (grant * string) list

  let exact = []

  let permits t g = List.mem_assoc g t

  let reason t g = List.assoc_opt g t

  (* Total: the reason is trusted, [add] checks it. The first reason is kept. *)
  let insert g why t =
    if permits t g then t
    else
      let before, after = List.partition (fun (h, _) -> rank h < rank g) t in
      before @ ((g, why) :: after)

  let add g ~reason t =
    if is_blank reason then Error Blank_reason else Ok (insert g reason t)

  let meet a b = List.filter (fun (g, _) -> permits b g) a

  let leq a b = List.for_all (fun (g, _) -> permits b g) a

  let grants t = t

  let digest t = String.concat "," (List.map (fun (g, _) -> grant_name g) t)
end

(* Abstract outside this file: the interface says [module Budget : BUDGET]. *)
module Budget : BUDGET with type t = Budget_impl.t = Budget_impl

(* ------------------------------------------------------------------------ *)
(* The process-wide store                                                    *)
(* ------------------------------------------------------------------------ *)

type entry = {scope : scope; granted : grant; why : string}

(* One-way and process-wide: entries are only ever added, so a decision taken
   under a grant (a source generated without barriers, a capability check
   that passed and was memoised) was taken under a policy that still holds.
   An [Atomic] because launches run on several domains and the store is read
   on every capability check. At most one entry per (scope, grant). *)
let store : entry list Atomic.t = Atomic.make []

(* The name Sarek_lower_ir gives every [%kernel] written without one. *)
let default_kernel_name = "sarek_kern"

(* Bumped by every change of the policy (a new grant, restriction or fused
   registration), after the change is visible: a reader that reads the version
   BEFORE it reads the policy can only err towards recomputing. *)
let version_counter = Atomic.make 0

let version () = Atomic.get version_counter

let bump () = Atomic.incr version_counter

(* Grants taken away for the rest of the process (SAREK_OPT_DISABLE). The
   store keeps the program's statements; the readers below skip a revoked
   grant, so a grant is recorded but never effective. One-way, like the store. *)
let revocations : grant list Atomic.t = Atomic.make []

let revoked g = List.mem g (Atomic.get revocations)

let rec revoke g =
  let current = Atomic.get revocations in
  if not (List.mem g current) then
    if Atomic.compare_and_set revocations current (g :: current) then bump ()
    else revoke g

let rec record scope granted why =
  let current = Atomic.get store in
  if List.exists (fun e -> e.scope = scope && e.granted = granted) current then
    ()
  else if Atomic.compare_and_set store current ({scope; granted; why} :: current)
  then bump ()
  else record scope granted why

let grant scope g ~reason =
  if is_blank reason then Error Blank_reason
  else if scope = Kernel default_kernel_name then Error Default_kernel_name
  else Ok (record scope g reason)

let find snapshot scope g =
  List.find_opt (fun e -> e.scope = scope && e.granted = g) snapshot
  |> Option.map (fun e -> e.why)

let reason ?kernel g =
  if revoked g then None
  else
    let snapshot = Atomic.get store in
    match Option.bind kernel (fun k -> find snapshot (Kernel k) g) with
    | Some why -> Some why
    | None -> find snapshot Process g

(* The grants of one scope added to [b]; reasons already in [b] are kept. *)
let add_scope snapshot scope b =
  List.fold_left
    (fun b e ->
      if e.scope = scope then Budget_impl.insert e.granted e.why b else b)
    b
    snapshot

(* ------------------------------------------------------------------------ *)
(* Fused kernels                                                             *)
(* ------------------------------------------------------------------------ *)

module Names = Map.Make (String)

(* Fused (or derived) kernel name -> the member names registered for it. Only
   grows: a name registered again gains the new members (a content-derived name
   can be built from different members, and then it is covered only by what all
   of them allow). An immutable map behind an atomic: a launch reads one
   snapshot. *)
let fused : string list Names.t Atomic.t = Atomic.make Names.empty

let rec register_fused ~name ~members =
  let current = Atomic.get fused in
  let known = Option.value ~default:[] (Names.find_opt name current) in
  let added = List.filter (fun m -> not (List.mem m known)) members in
  if added <> [] then
    if
      Atomic.compare_and_set
        fused
        current
        (Names.add name (known @ added) current)
    then bump ()
    else register_fused ~name ~members

let register_derived ~name ~from = register_fused ~name ~members:[from]

(* A kernel's own grants first, so its reason wins over the process one. *)
let own_budget snapshot kernel =
  add_scope
    snapshot
    Process
    (add_scope snapshot (Kernel kernel) Budget_impl.exact)

(* The meet over the members of a registered fused kernel (its own [Kernel]
   grants are not read: the name is generated), recursively. A name met again
   while expanding itself (a malformed registration) contributes nothing, which
   is the least permissive answer. *)
let rec kernel_budget snapshot registry visiting kernel =
  match Names.find_opt kernel registry with
  | _ when List.mem kernel visiting -> Budget_impl.exact
  | Some (first :: rest) ->
      let member = kernel_budget snapshot registry (kernel :: visiting) in
      List.fold_left
        (fun acc m -> Budget_impl.meet acc (member m))
        (member first)
        rest
  | Some [] | None -> own_budget snapshot kernel

let without_revoked revoked (b : Budget_impl.t) : Budget_impl.t =
  if revoked = [] then b
  else List.filter (fun (g, _) -> not (List.mem g revoked)) b

let budget ~kernels =
  let snapshot = Atomic.get store in
  let registry = Atomic.get fused in
  let raw =
    match kernels with
    | [] -> add_scope snapshot Process Budget_impl.exact
    | first :: rest ->
        let one = kernel_budget snapshot registry [] in
        List.fold_left
          (fun acc k -> Budget_impl.meet acc (one k))
          (one first)
          rest
  in
  without_revoked (Atomic.get revocations) raw

(* ------------------------------------------------------------------------ *)
(* Restrictions and the common decision                                      *)
(* ------------------------------------------------------------------------ *)

let restrictions : optimisation list Atomic.t = Atomic.make []

let restricted opt = List.mem opt (Atomic.get restrictions)

let rec restrict opt =
  let current = Atomic.get restrictions in
  if not (List.mem opt current) then
    if Atomic.compare_and_set restrictions current (opt :: current) then bump ()
    else restrict opt

type effective = {budget : Budget.t; disabled : optimisation list}

let effective_for ~kernels =
  let current = Atomic.get restrictions in
  {
    budget = budget ~kernels;
    disabled = List.filter (fun o -> List.mem o current) all_optimisations;
  }

let effective ~kernel = effective_for ~kernels:[kernel]

let effective_digest e =
  String.concat
    "|"
    [
      Budget.digest e.budget;
      String.concat "," (List.map optimisation_name e.disabled);
    ]

let requires opt ~backend =
  match opt with
  | Epilogue_fusion -> (
      match backend with
      | "HIP" | "Metal" -> Some []
      | "CUDA/PTX" | "CUDA/C" | "CUDA" | "OpenCL" | "Vulkan" ->
          Some [Deviation Cross_kernel_contraction]
      | _ -> None)
  | Launch_window | Matrix_rewrite | Narrowing_proof | Auto_unroll -> Some []

let disabled_reason = "disabled (SAREK_OPT_DISABLE)"

let decide e opt ~backend =
  if List.mem opt e.disabled then Refused disabled_reason
  else
    match requires opt ~backend with
    | None -> Refused ("not on " ^ backend)
    | Some needed -> (
        match
          List.find_opt (fun g -> not (Budget.permits e.budget g)) needed
        with
        | Some g when revoked g ->
            Refused
              (Printf.sprintf
                 "grant %s revoked (SAREK_OPT_DISABLE)"
                 (grant_name g))
        | Some g ->
            Refused
              (Printf.sprintf "no grant %s (kernel or process)" (grant_name g))
        | None -> (
            match needed with
            | [] -> Applied Exact_by_construction
            | _ ->
                Applied
                  (Granted
                     {
                       grants = needed;
                       reason =
                         String.concat
                           "; "
                           (List.filter_map (Budget.reason e.budget) needed);
                     })))

(* ------------------------------------------------------------------------ *)
(* Decision reporting                                                        *)
(* ------------------------------------------------------------------------ *)

let reporter : (event -> unit) option Atomic.t = Atomic.make None

let last_outcome : (string * string * optimisation, decision) Hashtbl.t =
  Hashtbl.create 16

(* The grants already reported as having no effect, by (kernel, grant, cause). *)
let noted : (string * grant * ignored, unit) Hashtbl.t = Hashtbl.create 16

(* Guards [last_outcome] and [noted]. *)
let outcome_lock = Mutex.create ()

let set_reporter r =
  Mutex.protect outcome_lock (fun () ->
      Hashtbl.reset last_outcome ;
      Hashtbl.reset noted) ;
  Atomic.set reporter r

(* True when [r] differs from the last outcome reported for its key. Under
   [outcome_lock]. *)
let is_new r =
  let key = (r.kernel, r.device, r.optimisation) in
  if Hashtbl.find_opt last_outcome key = Some r.decision then false
  else (
    Hashtbl.replace last_outcome key r.decision ;
    true)

(* The kernel names a record is about: a pair is written ["p + c"]. *)
let kernels_of name =
  let sep = " + " in
  let n = String.length sep in
  let rec split from i acc =
    if i + n > String.length name then
      List.rev (String.sub name from (String.length name - from) :: acc)
    else if String.sub name i n = sep then
      split (i + n) (i + n) (String.sub name from (i - from) :: acc)
    else split from (i + 1) acc
  in
  split 0 0 []

(* What the program granted for [kernel] that cannot take effect, from the
   store as it is now: a [Kernel] grant on a generated (fused or derived)
   name, a grant revoked by the environment. A revoked process grant is
   reported once, under the name ["(process)"]. *)
let ignored_grants kernel =
  let snapshot = Atomic.get store in
  let registry = Atomic.get fused in
  List.concat_map
    (fun e ->
      let generated =
        match e.scope with
        | Kernel k when k = kernel && Names.mem k registry ->
            [(kernel, e.granted, Generated_name)]
        | Kernel _ | Process -> []
      in
      let revoked_here =
        match e.scope with
        | Kernel k when k = kernel && revoked e.granted ->
            [(kernel, e.granted, Revoked)]
        | Process when revoked e.granted -> [("(process)", e.granted, Revoked)]
        | Kernel _ | Process -> []
      in
      generated @ revoked_here)
    snapshot

(* The not yet reported ones for a record, marked as reported. Under
   [outcome_lock]. *)
let fresh_ignored_grants record_kernel =
  List.filter
    (fun key ->
      if Hashtbl.mem noted key then false
      else (
        Hashtbl.replace noted key () ;
        true))
    (List.concat_map ignored_grants (kernels_of record_kernel))

let reporting () = Option.is_some (Atomic.get reporter)

let report r =
  match Atomic.get reporter with
  | None -> ()
  | Some f ->
      let fresh =
        Mutex.protect outcome_lock (fun () ->
            if is_new r then Some (fresh_ignored_grants r.kernel) else None)
      in
      Option.iter
        (fun notes ->
          f (Decision r) ;
          List.iter
            (fun (kernel, grant, cause) ->
              f (Ignored_grant {kernel; grant; cause}))
            notes)
        fresh

let refusal_message = function
  | Blank_reason ->
      "Sarek_policy.grant: the reason must not be empty (it is the record that \
       the deviation or the consent was understood)"
  | Default_kernel_name ->
      "Sarek_policy.grant: Kernel \"sarek_kern\" is the name of every unnamed \
       [%kernel], library kernels included; name the kernel, or grant at \
       Process scope"

(* ------------------------------------------------------------------------ *)
(* The environment                                                           *)
(* ------------------------------------------------------------------------ *)

module Env = struct
  type restriction = Sarek_policy_vocab.restriction =
    | Disable of optimisation
    | Revoke of grant

  type explain = Sarek_policy_vocab.explain =
    | Explain_off
    | Explain_stderr
    | Explain_unknown of string

  include Sarek_policy_env

  let narrow restrictions e =
    let revoked =
      List.filter_map
        (function Revoke g -> Some g | Disable _ -> None)
        restrictions
    in
    {
      budget = without_revoked revoked e.budget;
      disabled =
        List.filter
          (fun o -> List.mem o e.disabled || List.mem (Disable o) restrictions)
          all_optimisations;
    }
end

(* SAREK_OPT_DISABLE and SAREK_EXPLAIN are read here, once, when the process
   starts: a restriction is one-way, so there is nothing to re-read. *)
let apply_environment () =
  let restrictions, unknown =
    Env.parse (Option.value ~default:"" (Sys.getenv_opt "SAREK_OPT_DISABLE"))
  in
  List.iter (fun item -> prerr_endline (Env.unknown_message item)) unknown ;
  List.iter
    (function Env.Disable o -> restrict o | Env.Revoke g -> revoke g)
    restrictions ;
  match Env.parse_explain (Sys.getenv_opt "SAREK_EXPLAIN") with
  | Env.Explain_off -> ()
  | Env.Explain_stderr ->
      set_reporter (Some (fun e -> prerr_endline (Env.explain_line e)))
  | Env.Explain_unknown value ->
      prerr_endline (Env.explain_unknown_message value)

let () = apply_environment ()
