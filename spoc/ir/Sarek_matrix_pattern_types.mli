(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The vocabulary of {!Sarek_matrix_pattern}: what a recognised GEMM nest and a
    rejected reduction loop are. Split out so that the index-role analysis
    ({!Sarek_matrix_roles}) and the recogniser share it; {!Sarek_matrix_pattern}
    re-exports every type with its equalities. *)

open Sarek_ir_types

(** How an operand is laid out in memory, from its two strides: row-major when
    the column stride is 1, column-major when the row stride is 1. Any other
    operand (no unit stride, equal strides, overlapping rows) is not a GEMM
    operand: a matrix unit loads a tile with one base and one leading dimension.
*)
type layout = Row_major | Col_major

(** One matrix of [C = A × B]. Rows and columns are those of the matrix: A is
    [M × K], B is [K × N], C is [M × N]. The strides are uniform expressions;
    [offset] is everything in the index that is neither the row nor the column
    term (constants and batch terms). [index] is the let-inlined index. *)
type operand = {
  array : string;
  elt : elttype;
  index : expr;
  row_stride : expr;
  col_stride : expr;
  layout : layout;
  offset : expr;
}

type update =
  | Mul_add  (** [acc + x * y] *)
  | Fused_multiply_add  (** [fma x y acc] *)

type epilogue =
  | Plain  (** [C[ic] <- acc], possibly through a cast *)
  | Epilogue of expr
      (** [acc] occurs once, beside uniform values and reads of [C[ic]] (e.g.
          [alpha * acc + beta * C[ic]]) *)

type gemm = {
  acc : var;
  init : expr;  (** the accumulator's initial value, let-inlined *)
  k_var : var;
  k_lo : expr;  (** inclusive, uniform *)
  k_hi : expr;  (** inclusive, uniform *)
  m_index : expr;  (** the row atom, e.g. [global_idx_y] or [tid / n] *)
  n_index : expr;  (** the column atom *)
  batch : expr list;  (** batch atoms of C *)
  a : operand;
  b : operand;
  c : operand;
  update : update;
  epilogue : epilogue;
  guards : expr list;
      (** conditions of the enclosing [if]s, outermost first, then the store's
          own guard *)
  enclosing_loops : var list;  (** outermost first *)
}

(** Why a reduction loop is not a GEMM nest. Payloads name the array or the
    sub-expression at fault. *)
type reason =
  | Downto_loop
  | Loop_body_not_single_update
  | Not_a_product
  | Operand_not_an_element of expr
  | Operand_not_global of string
  | Bounds_not_uniform
  | Bounds_data_dependent
  | Index_not_affine of string * expr option
  | Operand_independent_of_reduction of string
  | Output_depends_on_reduction
  | No_row_column_split
  | Stray_index_atom of string
  | Nested_accumulation
  | Accumulator_escapes
  | No_output_store
  | Unsupported_epilogue
  | Output_aliases_operand of string
  | Operand_written of string
  | Overlapping_operand of string
  | Constant_leading_dimension of string
  | Operand_not_unit_stride of string
  | Dependent_indices
  | Unsupported_init

(** A reduction loop (a [for] whose body updates a [let mut] from itself) that
    is not a GEMM nest. *)
type near_miss = {nm_acc : var; nm_loop : var; nm_reason : reason}

type finding = Gemm of gemm | Near_miss of near_miss

val reason_text : reason -> string

val layout_name : layout -> string
