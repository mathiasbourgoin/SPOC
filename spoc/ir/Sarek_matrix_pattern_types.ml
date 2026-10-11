(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The vocabulary of Sarek_matrix_pattern; see its interface. *)

open Sarek_ir_types

type layout = Row_major | Col_major

type operand = {
  array : string;
  elt : elttype;
  index : expr;
  row_stride : expr;
  col_stride : expr;
  layout : layout;
  offset : expr;
}

type update = Mul_add | Fused_multiply_add

type epilogue = Plain | Epilogue of expr

type gemm = {
  acc : var;
  init : expr;
  k_var : var;
  k_lo : expr;
  k_hi : expr;
  m_index : expr;
  n_index : expr;
  batch : expr list;
  a : operand;
  b : operand;
  c : operand;
  update : update;
  epilogue : epilogue;
  guards : expr list;
  enclosing_loops : var list;
}

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
  | Output_written_elsewhere of string
  | Operand_not_unit_stride of string
  | Dependent_indices
  | Unsupported_init

type near_miss = {nm_acc : var; nm_loop : var; nm_reason : reason}

type finding = Gemm of gemm | Near_miss of near_miss

let reason_text = function
  | Downto_loop -> "the reduction loop counts down (downto)"
  | Loop_body_not_single_update ->
      "the reduction loop body does more than update the accumulator"
  | Not_a_product ->
      "the update is not acc + x * y (or fma x y acc) with two factors"
  | Operand_not_an_element _ -> "a factor is not an array element"
  | Operand_not_global arr ->
      Printf.sprintf
        "'%s' is not a global vector (shared/local: an already tiled kernel?)"
        arr
  | Bounds_not_uniform ->
      "the reduction bounds are not uniform (they vary per thread or per loop)"
  | Bounds_data_dependent -> "the reduction bounds read memory"
  | Index_not_affine (arr, _) ->
      Printf.sprintf "the index of '%s' is not affine in the loop variables" arr
  | Operand_independent_of_reduction arr ->
      Printf.sprintf "'%s' is not indexed by the reduction variable" arr
  | Output_depends_on_reduction ->
      "the output index depends on the reduction variable"
  | No_row_column_split ->
      "the operands do not separate into a row (M) and a column (N) index of \
       the output (dot product, matrix-vector product or linearised output)"
  | Stray_index_atom arr ->
      Printf.sprintf
        "the index of '%s' varies with something the output index does not"
        arr
  | Nested_accumulation ->
      "the accumulation is split across another loop or branch"
  | Accumulator_escapes ->
      "the accumulator is used outside the reduction and the final store"
  | No_output_store -> "the accumulator is not stored to a global vector"
  | Unsupported_epilogue ->
      "the stored value combines the accumulator with something other than \
       uniform values and the output element"
  | Output_aliases_operand arr ->
      Printf.sprintf "the output '%s' is also an operand" arr
  | Operand_written arr ->
      Printf.sprintf "the operand '%s' is (or may be) written by the kernel" arr
  | Overlapping_operand arr ->
      Printf.sprintf
        "the elements of '%s' overlap (equal strides: a Toeplitz view, a \
         correlation or a race)"
        arr
  | Output_written_elsewhere arr ->
      Printf.sprintf
        "the output '%s' is written by another statement too, which may \
         replace the product"
        arr
  | Constant_leading_dimension arr ->
      Printf.sprintf
        "'%s' has a constant leading dimension that is below a 16-wide tile or \
         not shown to cover its rows (the extent is not a known constant)"
        arr
  | Operand_not_unit_stride arr ->
      Printf.sprintf "'%s' has no unit stride in either dimension" arr
  | Dependent_indices ->
      "the row and column indices derive from the same thread value without \
       being its quotient and remainder by one divisor"
  | Unsupported_init ->
      "the accumulator starts from a value that is neither uniform nor the \
       output element"

let layout_name = function
  | Row_major -> "row-major"
  | Col_major -> "column-major"
