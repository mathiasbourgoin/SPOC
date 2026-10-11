(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Which outputs a recognised GEMM nest writes
    (specs/matrix-pattern-rewrite.md, "Preconditions").

    A rewritten kernel has its own launch geometry, so it must write exactly the
    outputs the original launch writes: no more (it would overwrite memory the
    original leaves alone), no fewer. {!analyse} recognises how the row, column
    and batch indices derive from the launch, and the guards that bound them;
    {!region} turns that into numbers for one launch. *)

open Sarek_ir_types

type axis = X | Y | Z

type index_kind =
  | Axis of axis
      (** [global_idx_a], or [thread_idx_a + block_dim_a * block_idx_a] *)
  | Loop of {var : var; lo : expr; hi : expr}  (** an enclosing [for] *)
  | Quotient of {linear : string; divisor : expr}
      (** [t / d], [t] the named linear thread id *)
  | Remainder of {linear : string; divisor : expr}  (** [t mod d] *)

type role = Row | Col | Batch | Linear

(** [Upper u]: index < u. [Lower l]: index >= l. *)
type bound = Upper of expr | Lower of expr

type mapping = {
  m : index_kind;
  n : index_kind;
  batch : index_kind option;
  bounds : (role * bound) list;
}

(** The outputs of one launch: rows [0, me), columns [0, ne), batches
    [0, ze); when [d > 0] (a quotient/remainder mapping) also
    [row * d + column < t]. *)
type region = {me : int; ne : int; ze : int; d : int; t : int}

val axis_char : axis -> char

(** [Error reason] when the indices or the guards are outside what the rewrite
    can bound. *)
val analyse : kernel -> Sarek_matrix_pattern.gemm -> (mapping, string) result

(** [Error reason] when a bound cannot be evaluated, a guard excludes the
    origin, or the launch writes nothing. *)
val region : mapping -> Sarek_ir_eval_int.launch -> (region, string) result
