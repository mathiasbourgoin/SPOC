(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Run a kernel, on matrix units when it is a recognised matrix product and
    that is allowed (specs/matrix-pattern-rewrite.md).

    {!run} is a drop-in for [Sarek.Execute.run_vectors] that asks for the
    rewrite explicitly: unlike the default path it also rewrites on the
    interpreter (specs/matrix-pattern-dynamic.md). When the kernel is a GEMM
    nest ({!Sarek_matrix_pattern}) whose indices and guards the rewrite can
    bound ({!Sarek_matrix_mapping}), the device offers the configuration and the
    accuracy policy admits it, it launches the rewritten kernel
    ({!Sarek_matrix_rewrite}) over exactly the outputs the original launch
    writes; otherwise it launches the original kernel, unchanged. The returned
    {!outcome} says which, and why.

    Policy: an int32 product is rewritten by default (bit-identical). A
    float16-operand product needs {!Sarek_coopmat.accept_relaxed_accumulation}.
    A float32 product needs [~precision] and
    {!Sarek_coopmat.accept_operand_rounding} for that precision, and
    {!Sarek_coopmat.accept_relaxed_accumulation}. *)

(** Which matrix rewrites [Execute] applies by default
    ({!Sarek_execute.Matrix_specialise.mode}): [Off] none, [Exact] the int32 one
    only, [Auto] (default) the int32 one plus the float ones the program has
    opted into. Initialised from [SAREK_MATRIX_REWRITE]. *)
type mode = Sarek_execute.Matrix_specialise.mode = Off | Exact | Auto

val set_mode : mode -> unit

val mode : unit -> mode

type outcome =
  | Rewritten of {
      lowering : Sarek_matrix_rewrite.lowering;
      config : Sarek_coopmat.config;
    }
  | Original of string  (** why the original kernel ran *)

val run :
  ?precision:Sarek_coopmat.operand_precision ->
  device:Spoc_core.Device.t ->
  ir:Sarek_ir_types.kernel ->
  args:Sarek.Execute.vector_arg list ->
  block:Spoc_framework.Framework_sig.dims ->
  grid:Spoc_framework.Framework_sig.dims ->
  unit ->
  outcome

val outcome_text : outcome -> string
