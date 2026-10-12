(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

type operand_precision = Operands_f16 | Operands_bf16 | Operands_bf16x3

type deviation =
  | Accumulation_order
  | Operand_rounding of operand_precision
  | Single_rounding
  | Cross_kernel_contraction

type consent = Automatic_float_rewrite

type grant = Deviation of deviation | Consent of consent

type optimisation =
  | Launch_window
  | Epilogue_fusion
  | Matrix_rewrite
  | Narrowing_proof
  | Auto_unroll

type scope = Process | Kernel of string

type refusal = Blank_reason | Default_kernel_name

type evidence = Proof of string | Granted of string | Exact_by_construction

type decision = Applied of evidence | Refused of string

let grant_name = function
  | Deviation Accumulation_order -> "accumulation-order"
  | Deviation (Operand_rounding Operands_f16) -> "operand-rounding-f16"
  | Deviation (Operand_rounding Operands_bf16) -> "operand-rounding-bf16"
  | Deviation (Operand_rounding Operands_bf16x3) -> "operand-rounding-bf16x3"
  | Deviation Single_rounding -> "single-rounding"
  | Deviation Cross_kernel_contraction -> "cross-kernel-contraction"
  | Consent Automatic_float_rewrite -> "automatic-float-rewrite"

let optimisation_name = function
  | Launch_window -> "window"
  | Epilogue_fusion -> "epilogue"
  | Matrix_rewrite -> "matrix"
  | Narrowing_proof -> "narrowing-proof"
  | Auto_unroll -> "unroll"

let all_grants =
  [
    Deviation Accumulation_order;
    Deviation (Operand_rounding Operands_f16);
    Deviation (Operand_rounding Operands_bf16);
    Deviation (Operand_rounding Operands_bf16x3);
    Deviation Single_rounding;
    Deviation Cross_kernel_contraction;
    Consent Automatic_float_rewrite;
  ]

let all_optimisations =
  [Launch_window; Epilogue_fusion; Matrix_rewrite; Narrowing_proof; Auto_unroll]

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

(* Fused kernel name -> the member names registered for it. Only grows: a name
   registered again gains the new members (a content-derived name can be built
   from different members, and then it is covered only by what all allow). *)
let fused : (string * string list) list Atomic.t = Atomic.make []

let rec register_fused ~name ~members =
  let current = Atomic.get fused in
  let known = Option.value ~default:[] (List.assoc_opt name current) in
  let added = List.filter (fun m -> not (List.mem m known)) members in
  if added <> [] then
    let next = (name, known @ added) :: List.remove_assoc name current in
    if Atomic.compare_and_set fused current next then bump ()
    else register_fused ~name ~members

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
  match List.assoc_opt kernel registry with
  | _ when List.mem kernel visiting -> Budget_impl.exact
  | Some (first :: rest) ->
      let member = kernel_budget snapshot registry (kernel :: visiting) in
      List.fold_left
        (fun acc m -> Budget_impl.meet acc (member m))
        (member first)
        rest
  | Some [] | None -> own_budget snapshot kernel

let budget ~kernels =
  let snapshot = Atomic.get store in
  let registry = Atomic.get fused in
  match kernels with
  | [] -> add_scope snapshot Process Budget_impl.exact
  | first :: rest ->
      let one = kernel_budget snapshot registry [] in
      List.fold_left
        (fun acc k -> Budget_impl.meet acc (one k))
        (one first)
        rest

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

let decide e opt ~backend =
  if List.mem opt e.disabled then Refused "disabled (SAREK_OPT_DISABLE)"
  else
    match requires opt ~backend with
    | None -> Refused ("not on " ^ backend)
    | Some needed -> (
        match
          List.find_opt (fun g -> not (Budget.permits e.budget g)) needed
        with
        | Some g ->
            Refused
              (Printf.sprintf "no grant %s (kernel or process)" (grant_name g))
        | None -> (
            match needed with
            | [] -> Applied Exact_by_construction
            | _ ->
                Applied
                  (Granted
                     (String.concat
                        "; "
                        (List.filter_map (Budget.reason e.budget) needed)))))

(* ------------------------------------------------------------------------ *)
(* Decision reporting                                                        *)
(* ------------------------------------------------------------------------ *)

type record = {
  kernel : string;
  device : string;
  optimisation : optimisation;
  decision : decision;
}

let reporter : (record -> unit) option Atomic.t = Atomic.make None

let last_outcome : (string * string * optimisation, decision) Hashtbl.t =
  Hashtbl.create 16

let outcome_lock = Mutex.create ()

let set_reporter r =
  Mutex.protect outcome_lock (fun () -> Hashtbl.reset last_outcome) ;
  Atomic.set reporter r

(* True when [r] differs from the last outcome reported for its key. *)
let is_new r =
  let key = (r.kernel, r.device, r.optimisation) in
  Mutex.protect outcome_lock (fun () ->
      if Hashtbl.find_opt last_outcome key = Some r.decision then false
      else (
        Hashtbl.replace last_outcome key r.decision ;
        true))

let report r =
  match Atomic.get reporter with None -> () | Some f -> if is_new r then f r

let refusal_message = function
  | Blank_reason ->
      "Sarek_policy.grant: the reason must not be empty (it is the record that \
       the deviation or the consent was understood)"
  | Default_kernel_name ->
      "Sarek_policy.grant: Kernel \"sarek_kern\" is the name of every unnamed \
       [%kernel], library kernels included; name the kernel, or grant at \
       Process scope"
