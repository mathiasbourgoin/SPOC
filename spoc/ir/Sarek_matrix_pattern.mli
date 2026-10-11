(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Recognition of matrix-product (GEMM) loop nests in Sarek IR.

    A {e GEMM nest} is an accumulator [acc] (a [let mut]) updated by a [for]
    loop over a reduction variable [kv] as [acc := acc + A[ia] * B[ib]] (or
    [fma A[ia] B[ib] acc]) and then stored once as [C[ic] <- e(acc)], where the
    three indices are affine ({!Sarek_ir_affine}) and separate into a row index
    [M] (in A and C), a column index [N] (in B and C), [kv] (in A and B) and
    optional batch indices (in C, and in A or B or both). The full pattern and
    every rejection are specified in specs/matrix-pattern-advice.md.

    This module only recognises. {!Sarek_matrix_advice} says which matrix-unit
    configuration applies; the PPX turns that into a compile-time alert; and
    {!Sarek_fusion} uses it to explain why a GEMM operand is not fused.

    Scope: the kernel body only, not helper functions. An accumulation written
    directly into memory ([c.(i) <- c.(i) + a * b] in the loop) is not a
    candidate. *)

(* [struct include ... end] keeps the type equalities, so
   [Sarek_matrix_pattern.gemm] and [Sarek_matrix_pattern_types.gemm] are the
   same type (the pattern of Sarek_coopmat). *)
include module type of struct
  include Sarek_matrix_pattern_types
end

open Sarek_ir_types

(** Every reduction loop of the kernel body, in body order. *)
val analyze : kernel -> finding list

(** The [Gemm] findings of {!analyze}. *)
val recognise : kernel -> gemm list
