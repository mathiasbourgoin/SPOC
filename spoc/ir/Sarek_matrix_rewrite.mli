(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The rewritten kernel of a GEMM nest (specs/matrix-pattern-rewrite.md).

    A 64 x 64 output tile per block of 128 invocations (4 waves of 32, each 2 x
    2 fragments of 16 x 16). Each step of 16 along the reduction stages the A
    and B tiles from global memory into shared memory — converted to the
    configuration's operand type, zero outside the region, either layout — then
    loads fragments from shared memory and multiplies. Every output the region
    covers is written with the original's initial value and epilogue; nothing
    else is written.

    The rewritten kernel takes the original's parameters, in order, followed by
    the five int32 parameters of {!extra_names}: the region
    ({!Sarek_matrix_mapping.region}'s [me], [ne], [ze], [d], [t]). *)

open Sarek_ir_types

type lowering = Sarek_matrix_scheme.lowering =
  | Int32_byte_planes
      (** int32: four unsigned byte planes per operand, the ten products of
          order at most 3, u8*u8+s32->s32: bit-identical modulo 2{^ 32} *)
  | F16_operands  (** f16*f16+f32->f32 (operands float16, or rounded to it) *)
  | Bf16_operands  (** float32 operands rounded to bfloat16 *)
  | Bf16x3_operands
      (** float32 operands split into three bfloat16 terms, the six products of
          order at most 2 *)

val lowering_name : lowering -> string

(** The cooperative-matrix configuration a lowering multiplies with, for the
    device gate. *)
val config_of : lowering -> operand_elt:elttype -> Sarek_coopmat_types.config

val extra_names : string list

(** [Error reason] when an operand address, a bound, the initial value or the
    epilogue cannot be expressed in the rewritten launch, or when [subgroup] is
    a width without a template ({!Sarek_matrix_scheme.supported_subgroup}: 32,
    the default, and 64). A block has four waves of [subgroup] lanes. *)
val kernel :
  ?subgroup:int ->
  kernel ->
  Sarek_matrix_pattern.gemm ->
  lowering ->
  (kernel, string) result

(** [(block, grid)] as [(x, y, z)] triples for a region; [subgroup] as for
    {!kernel}. *)
val launch_dims :
  ?subgroup:int ->
  me:int ->
  ne:int ->
  ze:int ->
  unit ->
  (int * int * int) * (int * int * int)
