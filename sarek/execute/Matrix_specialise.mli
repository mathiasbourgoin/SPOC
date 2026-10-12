(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Launch-time specialisation of matrix products
    (specs/matrix-pattern-dynamic.md).

    [Execute.run_vectors] asks {!choose}, per launch, whether to run a
    matrix-unit rewrite of the kernel it was given. The answer depends on the
    kernel (a GEMM nest the rewrite can bound, {!Sarek_matrix_mapping}), the
    device (HIP gfx11, Vulkan, CUDA/C, Metal, or the interpreter; see below),
    the restrictions of {!Sarek_policy}, the accuracy opt-ins of {!Sarek_coopmat}, and the actual sizes:
    the aligned float16 size class loads fragments straight from global memory
    ({!Sarek_matrix_direct}), every other case is staged through shared memory
    ({!Sarek_matrix_rewrite}). A rewritten launch writes exactly the outputs the
    original launch writes.

    Static plans and rewritten kernels are cached by the IR's physical identity
    (like [Execute]'s launch memo, #424) and by (lowering, size class, subgroup
    width, fragment edge); the device and the policy are re-checked on every launch,
    which costs a few list lookups. A kernel with no matrix product costs one
    memo lookup. *)

open Sarek_ir_types

(** [lowering_for ~budget ~explicit case precision]: the lowering the policy
    allows for an element case, or why none. int32 is always lowered; float
    cases need the operand rounding for [precision] (f32) and, unless
    [explicit], the automatic-rewrite consent, both read from [budget] (the
    grants of the kernel: {!Sarek_policy.budget}; a grant revoked by
    [SAREK_OPT_DISABLE] is not in it). The accuracy
    opt-in of the configuration ([accept_relaxed_accumulation]) is checked
    afterwards, with the device, by {!choose}. *)
val lowering_for :
  budget:Sarek_policy.Budget.t ->
  explicit:bool ->
  Sarek_matrix_advice.element_case ->
  Sarek_coopmat.operand_precision option ->
  (Sarek_matrix_rewrite.lowering, string) result

(** {1 Outcome} *)

(** [Staged]: tiles through shared memory, any size. [Direct w]: aligned
    float16, fragments from global memory, [w] waves per block. *)
type variant = Staged | Direct of int

type specialised = {
  ir : kernel;  (** the rewritten kernel, cached *)
  lowering : Sarek_matrix_rewrite.lowering;
  variant : variant;
  config : Sarek_coopmat.config;
  extra : int list;
      (** the values of {!Sarek_matrix_scheme.extra_names}, appended to the
          original arguments *)
  block : int * int * int;
  grid : int * int * int;
  via_interpreter : bool;
      (** launch through the interpreter's own entry point *)
  evidence : Sarek_policy.evidence;
      (** what justified the rewrite: int32 is a [Proof] (bit-identical), a
          float lowering is [Granted] with the grants it was admitted under;
          the explain report's [applied] line *)
}

type outcome = Specialised of specialised | Original of string  (** why *)

val variant_name : variant -> string

val outcome_text : outcome -> string

(** [choose ?precision ?explicit ?stream ~device ~ir ~launch ()]. [launch] gives
    the launch facts (scalar arguments, dimensions); it is only called for a
    kernel that is a matrix product. [precision] selects the operand rounding of
    a float32 product; without it, float16 then bfloat16, whichever was accepted
    (the bf16x3 split, more accurate but not faster, only when asked for).
    [explicit] (default [false]): the caller asked for the rewrite
    ({!Sarek_matrix.run}); only then is the interpreter rewritten — by default
    it stays the independent oracle — and the float rewrites then need the
    accuracy opt-ins only, not {!Sarek_coopmat.accept_automatic_float_rewrite}.
    A float kernel declined for lack of that opt-in yields an [Original] whose
    reason names it. [stream]: the launch is on a stream (the interpreter path
    has none). Devices other than HIP, Vulkan, CUDA/C, Metal and the interpreter
    are declined before any analysis; Vulkan is rewritten automatically on
    64-wide subgroups only, and CUDA/C (like the interpreter) on an explicit
    request only; Metal lowers the float16 operands only, on 8x8x8 fragments (it
    has no integer [simdgroup_matrix]: an integer product is declined with that
    reason); an exception in the analysis yields [Original].

    The cache is keyed by the IR's physical identity, as [Execute]'s launch
    memo: a kernel rebuilt for every launch is analysed every time. *)
val choose :
  ?precision:Sarek_coopmat.operand_precision ->
  ?explicit:bool ->
  ?stream:bool ->
  device:Spoc_core.Device.t ->
  ir:kernel ->
  launch:(unit -> Sarek_ir_eval_int.launch option) ->
  unit ->
  outcome

(** {1 Report}

    Every decision about a kernel that is a matrix product is an
    {!Sarek_policy.Matrix_rewrite} record for {!Sarek_policy.report} (kernel,
    device name, and [Applied evidence] or [Refused why]); not for other
    kernels or devices that can never rewrite. [SAREK_EXPLAIN=1] prints them. *)

(** Test hook: while [true], the analysis of a kernel not yet cached raises, as
    a defect in it would. Launches must still succeed ([Original]). *)
val fail_analysis_for_tests : bool -> unit
[@@alert test_only "for tests only: makes matrix analysis fail"]

(** Test hook: the subgroup width the rewrite is laid out for on the interpreter
    (default 32). The interpreter's cooperative-matrix subgroup is fixed at 32
    lanes, so a 64-lane wave runs as two interpreter subgroups: this checks the
    staging guard, the strides and the block size against the oracle, not the
    wave-to-subgroup mapping (that is covered by the Vulkan e2e run). Honoured
    on the interpreter only. *)
val subgroup_override_for_tests : int option -> unit
[@@alert
  test_only "for tests only: lays the interpreter rewrite out for 64 lanes"]

(** Test hook: the fragment edge (m = n = k, 8 or 16) the rewrite is laid out
    for on the interpreter (default 16). On a device it is read from the
    advertised configurations (16 where offered, else 8, which is Metal).
    Honoured on the interpreter only. *)
val fragment_override_for_tests : int option -> unit
[@@alert test_only "for tests only: lays the interpreter rewrite out for 8x8x8"]

(** Forget every cached plan and rewritten kernel. *)
val clear : unit -> unit
