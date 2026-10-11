(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Launch-time specialisation of matrix products
    (specs/matrix-pattern-dynamic.md).

    [Execute.run_vectors] asks {!choose}, per launch, whether to run a
    matrix-unit rewrite of the kernel it was given. The answer depends on the
    kernel (a GEMM nest the rewrite can bound, {!Sarek_matrix_mapping}), the
    device (HIP gfx11 with 32-wide waves, or the interpreter), the {!mode}, the
    accuracy opt-ins of {!Sarek_coopmat}, and the actual sizes: the aligned
    float16 size class loads fragments straight from global memory
    ({!Sarek_matrix_direct}), every other case is staged through shared memory
    ({!Sarek_matrix_rewrite}). A rewritten launch writes exactly the outputs the
    original launch writes.

    Static plans and rewritten kernels are cached by the IR's physical identity
    (like [Execute]'s launch memo, #424) and by (lowering, size class); the
    device and mode are re-checked on every launch, which costs a few list
    lookups. A kernel with no matrix product costs one memo lookup. *)

open Sarek_ir_types

(** {1 Mode} *)

(** [Off]: never rewrite. [Exact]: only rewrites that keep results bit-identical
    (int32). [Auto] (default): [Exact] plus, for a program's own float kernels,
    the float rewrites the program opted into: the explicit
    {!Sarek_coopmat.accept_automatic_float_rewrite} AND the accuracy opt-ins the
    configuration requires ({!Sarek_coopmat.accept_relaxed_accumulation},
    {!Sarek_coopmat.accept_operand_rounding}); a float rewrite never happens
    without all of them. The accuracy opt-ins alone admit matrix units only for
    code that asks for them ([Sarek_gemm.Coopmat], {!choose} with
    [~explicit:true]). Initialised from [SAREK_MATRIX_REWRITE], parsed by
    {!parse_mode}: an unrecognised value warns on stderr and gives [Auto]. *)
type mode = Off | Exact | Auto

val mode_name : mode -> string

(** [parse_mode value] reads the value of [SAREK_MATRIX_REWRITE] (pure).
    Case-insensitive: [off], [0], [false], [no] give [Off]; [exact] gives
    [Exact]; [auto], [on], [1], [true], [yes] give [Auto]; unset or empty gives
    [Auto]. Any other value gives [Auto] and a warning text (naming the
    variable, the value and the accepted values) that the caller prints. *)
val parse_mode : string option -> mode * string option

(** [lowering_for ~mode ~explicit case precision]: the lowering the policy
    allows for an element case, or why none. int32 is always lowered; float
    cases need mode [Auto], the accepted operand rounding for [precision] (f32),
    and, unless [explicit], {!Sarek_coopmat.accept_automatic_float_rewrite}. The
    accuracy opt-in of the configuration ([accept_relaxed_accumulation]) is
    checked afterwards, with the device, by {!choose}. *)
val lowering_for :
  mode:mode ->
  explicit:bool ->
  Sarek_matrix_advice.element_case ->
  Sarek_coopmat.operand_precision option ->
  (Sarek_matrix_rewrite.lowering, string) result

val set_mode : mode -> unit

val mode : unit -> mode

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
    has none). Devices other than HIP and the interpreter are declined before
    any analysis; an exception in the analysis yields [Original].

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

(** {1 Report} *)

type report = {kernel : string; device : string; outcome : outcome}

(** Called with every decision about a kernel that is a matrix product (not for
    other kernels, nor in mode [Off]). With [SAREK_MATRIX_REPORT=1], each
    distinct (kernel, device, outcome) is also printed once on stderr. *)
val set_reporter : (report -> unit) option -> unit

(** Test hook: while [true], the analysis of a kernel not yet cached raises, as
    a defect in it would. Launches must still succeed ([Original]). *)
val fail_analysis_for_tests : bool -> unit
[@@alert test_only "for tests only: makes matrix analysis fail"]

(** Test hook: the subgroup width the rewrite is laid out for on the interpreter
    (default 32), so that the 64-wide templates are checked against the oracle.
    Honoured on the interpreter only. *)
val subgroup_override_for_tests : int option -> unit
[@@alert
  test_only "for tests only: lays the interpreter rewrite out for 64 lanes"]

(** Forget every cached plan and rewritten kernel. *)
val clear : unit -> unit
