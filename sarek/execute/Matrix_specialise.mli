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
    (int32). [Auto] (default): [Exact] plus the float rewrites the program opted
    into ({!Sarek_coopmat.accept_relaxed_accumulation},
    {!Sarek_coopmat.accept_operand_rounding}); a float rewrite never happens
    without them. Initialised from [SAREK_MATRIX_REWRITE] ([off], [exact],
    anything else: [auto]). *)
type mode = Off | Exact | Auto

val mode_name : mode -> string

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
    it stays the independent oracle. [stream]: the launch is on a stream (the
    interpreter path has none). Devices other than HIP and the interpreter are
    declined before any analysis; an exception in the analysis yields
    [Original].

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

(** Forget every cached plan and rewritten kernel. *)
val clear : unit -> unit
