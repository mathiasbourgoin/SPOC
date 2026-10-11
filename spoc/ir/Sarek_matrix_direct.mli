(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The aligned size class of the matrix rewrite
    (specs/matrix-pattern-dynamic.md).

    When the region is whole 32 x 32 tiles, the reduction a multiple of 16, the
    operands float16 and every matrix row-major, the fragments are loaded
    straight from global memory (as {!Sarek_gemm.Coopmat} does) instead of being
    staged through shared memory: one wave per 32 x 32 tile, 2 x 2 fragments,
    [waves] waves per block, batches on grid y. Same arguments as
    {!Sarek_matrix_rewrite.kernel}; same configuration as the staged float16
    lowering, so the same opt-in applies. *)

open Sarek_ir_types

val config : Sarek_coopmat_types.config

(** The plan admits the variant: float16 operands, row-major A, B and C, a
    float32 output stored as is, the accumulator starting from zero (from C it
    would update C in place through fragments, which the interpreter evaluates
    once per invocation: such a kernel is staged). *)
val eligible : Sarek_matrix_pattern.gemm -> bool

(** The region admits it: [me] and [ne] multiples of 32, [k_extent] a multiple
    of 16, no quotient predicate ([d = 0]). *)
val fits : me:int -> ne:int -> k_extent:int -> d:int -> bool

(** 2 when the number of wave tiles is even, else 1. *)
val waves_for : me:int -> ne:int -> int

val launch_dims :
  me:int ->
  ne:int ->
  ze:int ->
  waves:int ->
  (int * int * int) * (int * int * int)

val kernel :
  kernel -> Sarek_matrix_pattern.gemm -> waves:int -> (kernel, string) result
