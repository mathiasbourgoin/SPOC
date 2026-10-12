(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** One typed store for every precision grant and consent
    (specs/optimisation-policy.md).

    The program states, once, which deviations from the interpreter's result it
    accepts ({!deviation}) and which rewrites of its own code it allows
    ({!consent}); this module records those statements and answers "is this
    grant in force, and why". It sits in the lowest layer ([sarek_ir], no
    dependencies) so that code generation, the capability gate and the launch
    path can all ask it.

    {b What it is not.} It is not a switch board: nothing here turns an
    optimisation on or off, and no environment variable grants anything (SPOC
    and Sarek environment variables only restrict). It does not change what any
    caller computes: {!Sarek_coopmat.accept_relaxed_accumulation} and the other
    opt-ins are thin wrappers over {!grant}, with the same refusals and the same
    messages as before it existed.

    {b Who consults which scope} (milestone M2). Every decision point that knows
    the kernel it is deciding for reads the grants of that kernel through
    {!budget} (or {!effective}): the f32 to f16 narrowing, the coopmat verdicts
    ({!Sarek_coopmat.policy_verdict_for}), the matrix rewrite and the epilogue
    fusion. A [Kernel]-scoped grant therefore changes the decision for that
    kernel and for no other. Callers with no kernel read the process scope. The
    text a code generator emits, and what the launch memo stores, depend on the
    policy: a grant given after a kernel's source was generated takes effect at
    its next launch (the memo compares {!version}, then the digest of the
    kernel's {!effective} policy).

    {b Restrictions.} {!restrict} is the one way to take an optimisation away:
    process-wide and one-way, like a grant. It is a diagnostic switch for SPOC's
    own environment parsing and for tests, not a scoped API.

    {b The environment} ([SAREK_OPT_DISABLE], [SAREK_EXPLAIN]) is read once,
    when this module is initialised (at program start, before any query), by
    {!Env}. [SAREK_OPT_DISABLE] only restricts: it disables optimisations and
    {e revokes} grants for the whole process (a revoked grant is still recorded
    by {!grant}, but no budget, reader or decision sees it). It never grants.
    [SAREK_EXPLAIN=1] installs a reporter that writes one line per decision to
    stderr ({!Env.explain_line}). *)

(** {1 Vocabulary} *)

(** The operand type a float32 matrix product is rounded to before it runs on
    matrix units. [Operands_bf16x3] splits each operand into three bfloat16
    terms for near-float32 accuracy. Re-exported, with its constructors, by
    {!Sarek_coopmat}. *)
type operand_precision = Operands_f16 | Operands_bf16 | Operands_bf16x3

(** A named way a result may differ from the interpreter's. Each carries its own
    documented bound where one exists; there is no common numeric scale. *)
type deviation =
  | Accumulation_order
      (** the hardware chooses the order of the additions
          ({!Sarek_coopmat.accept_relaxed_accumulation}) *)
  | Operand_rounding of operand_precision
      (** float32 operands are rounded to a narrower type on load
          ({!Sarek_coopmat.accept_operand_rounding}) *)
  | Single_rounding
      (** an f32 to f16 narrowing may round once where the interpreter rounds
          twice ({!Sarek_f16_policy.accept_single_rounding}) *)
  | Cross_kernel_contraction
      (** a device compiler may contract a multiply of one fused kernel with an
          add of the next. Grantable now; read by epilogue fusion from M2. *)

(** A permission to rewrite the program's own code, separate from accepting any
    accuracy deviation: granting one never grants the other. *)
type consent =
  | Automatic_float_rewrite
      (** the automatic float matrix rewrite of [Execute.run_vectors]
          ({!Sarek_coopmat.accept_automatic_float_rewrite}) *)

type grant = Deviation of deviation | Consent of consent

(** Every transformation a policy decision can apply to a launch. Vocabulary for
    the explain report; the decision points are wired to it from M2. *)
type optimisation =
  | Launch_window
  | Epilogue_fusion
  | Matrix_rewrite
  | Narrowing_proof
  | Auto_unroll

(** Where a grant applies. [Kernel name] matches
    {!Sarek_ir_types.kernel.kern_name} exactly; a kernel built by a fusion pass
    has the fused name, which is never matched by a grant: its budget is the
    meet over its parts ({!register_fused}). *)
type scope = Process | Kernel of string

(** Why {!grant} (or {!Budget.add}) refused. [Default_kernel_name]:
    [Kernel "sarek_kern"], the name every unnamed [[%kernel]] gets, would cover
    all of them. *)
type refusal = Blank_reason | Default_kernel_name

(** What justifies an applied optimisation, for the explain report. [Granted]:
    the grants it needed (each held by the kernel) and their reasons joined by
    ["; "]. *)
type evidence =
  | Proof of string
  | Granted of {grants : grant list; reason : string}
  | Exact_by_construction

(** The outcome of one decision point, for the explain report. *)
type decision = Applied of evidence | Refused of string

(** Stable names, used in reports and in cache keys: ["accumulation-order"],
    ["operand-rounding-f16"], ["operand-rounding-bf16"],
    ["operand-rounding-bf16x3"], ["single-rounding"],
    ["cross-kernel-contraction"], ["automatic-float-rewrite"]. *)
val grant_name : grant -> string

(** ["window"], ["epilogue"], ["matrix"], ["narrowing-proof"], ["unroll"]. *)
val optimisation_name : optimisation -> string

(** The seven grants, in the order {!Budget.grants} lists them. *)
val all_grants : grant list

val all_optimisations : optimisation list

(** {1 Budgets} *)

module type BUDGET = sig
  (** A set of grants, each with the reason it was given. *)
  type t

  (** No grant: the bottom of {!leq}. *)
  val exact : t

  (** [add g ~reason b] grants [g]. A blank [reason] is [Error Blank_reason],
      the only refusal [add] can return ([Default_kernel_name] concerns the
      kernel scope of {!grant}). The error type is shared with {!grant} on
      purpose. The first reason for a grant is kept; adding it again is a no-op.
  *)
  val add : grant -> reason:string -> t -> (t, refusal) result

  val permits : t -> grant -> bool

  val reason : t -> grant -> string option

  (** Intersection: a grant survives only if both operands hold it. The reason
      is the left operand's. The budget of a fused kernel is the meet of its
      members' budgets. *)
  val meet : t -> t -> t

  (** [leq a b]: every grant of [a] is in [b]. [exact] is below everything and
      {!meet} is the greatest lower bound. *)
  val leq : t -> t -> bool

  (** The grants and their reasons, in {!all_grants} order. *)
  val grants : t -> (grant * string) list

  (** A stable string of the grant set: equal iff the sets are equal, whatever
      the reasons and the order in which grants were added. Meant for cache
      keys, where a policy change must change the key. *)
  val digest : t -> string
end

module Budget : BUDGET

(** {1 The process-wide store} *)

(** [grant scope g ~reason] records that the program accepts [g] for [scope].
    Process-wide and one-way: there is no withdrawal, so a check that passed
    (and that a caller memoised) can never become stale. Thread-safe (an
    [Atomic] updated by compare-and-set). The first non-empty [reason] per
    [(scope, g)] is kept; later calls for the same pair are no-ops. A blank
    [reason] is [Error Blank_reason] and [Kernel "sarek_kern"] is
    [Error Default_kernel_name]; a refusal records nothing. *)
val grant : scope -> grant -> reason:string -> (unit, refusal) result

(** [reason ?kernel g]: with [~kernel], that kernel's own grant first, then the
    process grant; without, the process grant only. [None] when not granted. *)
val reason : ?kernel:string -> grant -> string option

(** [budget ~kernels]: the budget of a launch whose kernel is built from
    [kernels]. For each kernel, its process grants plus its own; the result is
    the {!Budget.meet} over the kernels, so a grant held by one member only is
    absent. [[]] gives the process grants only. Reads one consistent snapshot of
    the store. *)
val budget : kernels:string list -> Budget.t

(** The sentence for a refusal of {!grant}. *)
val refusal_message : refusal -> string

(** {1 Fused kernels} *)

(** [register_fused ~name ~members]: a fusion pass built the kernel [name] from
    the kernels [members] (themselves possibly fused kernels). From then on
    [budget ~kernels:[name]] is the {!Budget.meet} over the budgets of the
    members, recursively, so a fused kernel holds a grant only if every member
    does, and a grant given to a member after the registration is seen.

    A [Kernel] grant on the fused kernel's own name is {b not} counted: fusion
    names are generated (content digests, counters) and are not a statement of
    the program. Registering a name again adds the new members to those already
    registered (a content-derived name can come from different members; it is
    then covered only by what all of them allow). A registration with no member
    is ignored, and a name that would contain itself contributes no grant.
    Thread-safe; bumps {!version} when it changes the registry. *)
val register_fused : name:string -> members:string list -> unit

(** [register_derived ~name ~from]: the kernel [name] is [from] rewritten (the
    matrix rewrite renames its output [<from>_sarek_mm]), not a statement of the
    program, and has exactly [from]'s budget (recursively, if [from] is itself
    fused or derived) so that a [Kernel] grant of [from] is read for the
    rewritten kernel at the capability gate and by the interpreter.
    [register_fused ~name ~members:[from]]; same registry, same rules. *)
val register_derived : name:string -> from:string -> unit

(** {1 Restrictions} *)

(** [restrict opt] takes [opt] away for the rest of the process. One-way and
    thread-safe; restricting twice is a no-op. Only SPOC's environment parsing
    ([SAREK_OPT_DISABLE]) and tests call it. A restriction never invalidates a
    check a caller memoised, because a memoised check is re-keyed on {!version}.
    Bumps {!version} when [opt] was not yet restricted. *)
val restrict : optimisation -> unit

val restricted : optimisation -> bool

(** [revoked g]: [SAREK_OPT_DISABLE] revoked [g] for the process. A revoked
    grant is still recorded by {!grant} (the program's statement), but
    {!reason}, {!budget}, {!effective} and {!decide} never see it. There is no
    way to revoke in code: only the environment does, once, at start-up. *)
val revoked : grant -> bool

(** {1 The effective policy of a launch} *)

(** What a decision point reads: the grants in force for the kernel being
    decided, and the optimisations taken away. [disabled] is in
    {!all_optimisations} order. *)
type effective = {budget : Budget.t; disabled : optimisation list}

(** [effective ~kernel]: [budget ~kernels:[kernel]] and the current
    restrictions. *)
val effective : kernel:string -> effective

(** As {!effective} for a launch built from [kernels] (the meet of their
    budgets): the epilogue decision of a producer and its consumer. *)
val effective_for : kernels:string list -> effective

(** A stable string of the whole effective policy: {!Budget.digest} and the
    disabled optimisations. Equal iff the policies are equal. *)
val effective_digest : effective -> string

(** What an optimisation needs on a backend, as grants that must be held;
    [Some []] is exact there and [None] is not offered there.
    - [Epilogue_fusion]: [Some []] on ["HIP"] and ["Metal"] (every kernel is
      compiled without contraction); [Some [Deviation Cross_kernel_contraction]]
      on ["CUDA/PTX"], ["CUDA/C"], ["CUDA"], ["OpenCL"] and ["Vulkan"]; [None]
      elsewhere (Native, Interpreter, WebGPU never defer).
    - [Launch_window], [Narrowing_proof], [Auto_unroll]: [Some []] everywhere.
    - [Matrix_rewrite]: [Some []] everywhere; its float lowerings check their
      own grants (the accuracy opt-ins and the consent), unchanged.

    A Metal that marked itself inexact ([Fp_exactness]) is refused by the
    epilogue even with the grant: that fact lives above this module. *)
val requires : optimisation -> backend:string -> grant list option

(** ["disabled (SAREK_OPT_DISABLE)"]: the reason of a [Refused] by a
    restriction, for decision points that refuse without {!decide}. *)
val disabled_reason : string

(** The common decision, in this order:
    - [Refused "disabled (SAREK_OPT_DISABLE)"] if [opt] is in [e.disabled];
    - [Refused ("not on " ^ backend)] if {!requires} is [None];
    - [Refused ("no grant " ^ name ^ " (kernel or process)")] for the first
      required grant [e.budget] does not hold, or
      [Refused ("grant " ^ name ^ " revoked (SAREK_OPT_DISABLE)")] when it does
      not hold because the environment revoked it;
    - [Applied Exact_by_construction] if nothing is required;
    - [Applied (Granted {grants; reason})] otherwise: the required grants and
      their reasons joined by ["; "]. *)
val decide : effective -> optimisation -> backend:string -> decision

(** {1 Change counter and decision reporting} *)

(** Increases with every change of the policy: a grant that was not yet
    recorded, a restriction not yet in force, a fused registration that adds a
    member. Refusals, duplicates and reads leave it alone. A cache that memoised
    something derived from the policy stores [version ()] read {b before} it
    read the policy; one atomic read tells it whether anything changed since. *)
val version : unit -> int

(** One decision at a decision point, for the explain report. *)
type record = {
  kernel : string;
  device : string;
  optimisation : optimisation;
  decision : decision;
}

(** Why a grant the program made has no effect. [Generated_name]: a [Kernel]
    grant whose name is a fused, derived or static-fusion kernel's (generated,
    never matched). [Revoked]: the environment revoked the grant. *)
type ignored = Generated_name | Revoked

(** What the sink of {!report} receives: a decision, or a statement of the
    program that the policy does not obey. [Ignored_grant] comes right after the
    first decision that sees the kernel, once per [(kernel, grant, cause)]; a
    grant revoked at [Process] scope is reported once, with the kernel name
    ["(process)"]. *)
type event =
  | Decision of record
  | Ignored_grant of {kernel : string; grant : grant; cause : ignored}

(** Install (or, with [None], remove) the sink of {!report}. The default is no
    sink, except that [SAREK_EXPLAIN=1] installs one at start-up that writes
    {!Env.explain_line} to stderr; a sink installed by the program replaces it.
    Installing one forgets which outcomes were already reported. *)
val set_reporter : (event -> unit) option -> unit

(** [true] when a sink is installed: lets a decision point skip building a
    record (and its strings) when nobody listens. *)
val reporting : unit -> bool

(** [report r]: with a sink installed, calls it if [r.decision] differs from the
    last one reported for [(r.kernel, r.device, r.optimisation)]; otherwise does
    nothing. Without a sink it is one atomic read. Safe to call from any domain.
*)
val report : record -> unit

(** {1 The environment} *)

(** [SAREK_OPT_DISABLE], [SAREK_EXPLAIN]. The pure part, so that tests need no
    environment; the module reads the variables once at initialisation and
    applies them with {!restrict} and an internal revocation. Nothing here can
    grant. *)
module Env : sig
  (** One item of [SAREK_OPT_DISABLE]: take an optimisation away, or revoke a
      grant for the whole process (the budget never holds it, whatever the
      program grants). *)
  type restriction = Disable of optimisation | Revoke of grant

  (** [parse value]: the restrictions of a [SAREK_OPT_DISABLE] value, in order
      of first appearance, and the unknown items (trimmed, lowercased).
      Comma-separated; case-insensitive; surrounding spaces and empty items are
      ignored; a repeated item counts once. The items are {!optimisation_name}
      (disable) and {!grant_name} (revoke); there is no ["all"]. *)
  val parse : string -> restriction list * string list

  (** The one-line warning for an unknown item: names the variable, the item and
      the valid items. Written to stderr at start-up, once per item; the item is
      ignored. *)
  val unknown_message : string -> string

  (** [narrow restrictions e]: [e] with the optimisations disabled and the
      grants revoked. Never adds a grant or removes a restriction: for every
      [e], the budget of the result is below that of [e] ({!Budget.leq}) and
      {!decide} applies nothing on the result that it did not apply on [e]. *)
  val narrow : restriction list -> effective -> effective

  (** What [SAREK_EXPLAIN] says. Exactly [1] turns the report on; unset, empty
      or [0] is off; anything else is off with a warning
      ({!explain_unknown_message}). *)
  type explain = Explain_off | Explain_stderr | Explain_unknown of string

  val parse_explain : string option -> explain

  val explain_unknown_message : string -> string

  (** One line, no newline, space-separated; names have their spaces replaced by
      ['_'] and an epilogue pair ["p + c"] is written ["p+c"]:
      {v
sarek-explain <kernel> <device> <optimisation> applied proof=<text>
sarek-explain <kernel> <device> <optimisation> applied granted=<grant>[,<grant>] reason="<reason>"
sarek-explain <kernel> <device> <optimisation> applied exact
sarek-explain <kernel> <device> <optimisation> refused <reason>
sarek-explain <kernel> - policy refused grant-on-generated-name <grant>
sarek-explain <kernel> - policy refused grant-revoked <grant>
      v}
      [<optimisation>] and [<grant>] are {!optimisation_name} and {!grant_name};
      [<reason>] has its quotes and backslashes escaped. *)
  val explain_line : event -> string
end
