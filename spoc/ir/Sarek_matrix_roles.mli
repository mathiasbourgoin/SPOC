(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Index roles of a GEMM nest (specs/matrix-pattern-advice.md).

    Given the affine forms of the two factors' indices and of the output index,
    decide which atoms form the row index M (in A and C), the column index N (in
    B and C) and the batch indices (in C, and in A or B or both); name the
    factors A and B so that C's column stride is 1 when either naming allows it;
    and derive each operand's strides, layout and offset. *)

open Sarek_ir_types
open Sarek_matrix_pattern_types

(** [index_roles env kv ~k_extent (x, y) (c, ic, ec)] where [k_extent] is the
    constant iteration count when known (a constant leading dimension below it
    means overlapping rows), and [x] and [y] are the two factors as
    [(array, let-inlined index, element type)], [env] classifies [kv] and the
    enclosing loop variables as atoms, and [c]/[ic]/[ec] are the output's array,
    inlined index and element type. Returns A, B, C and
    [(m_index, n_index, batch_atoms)]. *)
val index_roles :
  Sarek_ir_affine.env ->
  var ->
  k_extent:Sarek_ir_affine.poly option ->
  (string * expr * elttype) * (string * expr * elttype) ->
  string * expr * elttype ->
  (operand * operand * operand * (expr * expr * expr list), reason) result
