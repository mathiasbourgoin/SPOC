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

let rec record scope granted why =
  let current = Atomic.get store in
  if List.exists (fun e -> e.scope = scope && e.granted = granted) current then
    ()
  else if
    not
      (Atomic.compare_and_set store current ({scope; granted; why} :: current))
  then record scope granted why

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

(* A kernel's own grants first, so its reason wins over the process one. *)
let kernel_budget snapshot kernel =
  add_scope
    snapshot
    Process
    (add_scope snapshot (Kernel kernel) Budget_impl.exact)

let budget ~kernels =
  let snapshot = Atomic.get store in
  match kernels with
  | [] -> add_scope snapshot Process Budget_impl.exact
  | first :: rest ->
      List.fold_left
        (fun acc k -> Budget_impl.meet acc (kernel_budget snapshot k))
        (kernel_budget snapshot first)
        rest

let refusal_message = function
  | Blank_reason ->
      "Sarek_policy.grant: the reason must not be empty (it is the record that \
       the deviation or the consent was understood)"
  | Default_kernel_name ->
      "Sarek_policy.grant: Kernel \"sarek_kern\" is the name of every unnamed \
       [%kernel], library kernels included; name the kernel, or grant at \
       Process scope"
