(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Matrix-product pattern corpus, NEGATIVE half (specs/matrix-pattern-advice.md
 * US-3). Compiled with [-alert @sarek_matrix] (see dune) and no silencing
 * attribute anywhere: a false positive fails the build before the run-time
 * checks of each rejection reason.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std
module P = Sarek_matrix_pattern
module Adv = Sarek_matrix_advice
open Matrix_pattern_check

(* ========================================================================= *)
(* Negative corpus: no silencing attribute anywhere below.                   *)
(* ========================================================================= *)

let n_dot =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (out : float32 vector)
        (k : int32) ->
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.(i) *. b.(i))
      done ;
      out.(0) <- sum]

let n_row_sum =
  [%kernel
    fun (a : float32 vector) (out : float32 vector) (k : int32) ->
      let row = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. a.((row * k) + i)
      done ;
      out.(row) <- sum]

let n_matvec =
  [%kernel
    fun (a : float32 vector)
        (x : float32 vector)
        (y : float32 vector)
        (k : int32) ->
      let row = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + i) *. x.(i))
      done ;
      y.(row) <- sum]

let n_linear_output =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let open Std in
      let tid = global_thread_id in
      let row = tid / n in
      let col = tid mod n in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.(tid) <- sum]

let n_modulo_index =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.(((row * k) + i) mod 7l) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_square_index =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + (i * i)) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_gather =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (perm : int32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + perm.(i)) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_data_bound =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (lens : int32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to lens.(row) - 1l do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_triangular =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to row do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_downto =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = k - 1l downto 0 do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_three_factors =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32)
        (alpha : float32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (alpha *. a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_subtraction =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum -. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_product_reduction =
  [%kernel
    fun (a : float32 vector) (out : float32 vector) (k : int32) ->
      let row = global_idx_x in
      let p = mut 1.0 in
      for i = 0 to k - 1l do
        p := p *. a.((row * k) + i)
      done ;
      out.(row) <- p]

let n_aliased_output =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (n : int32) (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      a.((row * n) + col) <- sum]

let n_operand_written =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum ;
      b.(col) <- 0.0]

let n_guard_reads_acc =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      if sum > 0.0 then c.((row * n) + col) <- sum]

let n_varying_epilogue =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (bias : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum +. bias.(col)]

let n_shared_tiled =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let%shared (tile_a : float32) = 256l in
      let%shared (tile_b : float32) = 256l in
      let tx = thread_idx_x in
      let ty = thread_idx_y in
      let row = ty + (block_dim_y * block_idx_y) in
      let col = tx + (block_dim_x * block_idx_x) in
      let sum = mut 0.0 in
      for t = 0 to (k / 16l) - 1l do
        let%superstep load =
          tile_a.((ty * 16l) + tx) <- a.((row * k) + (t * 16l) + tx) ;
          tile_b.((ty * 16l) + tx) <- b.((((t * 16l) + ty) * n) + col)
        in
        let%superstep _compute =
          for i = 0 to 15l do
            sum := sum +. (tile_a.((ty * 16l) + i) *. tile_b.((i * 16l) + tx))
          done
        in
        ()
      done ;
      if row < m && col < n then c.((row * n) + col) <- sum]

let n_overlapping_output =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.(row + col) <- sum]

let n_toeplitz =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.(row + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_correlation =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.(row + i) *. b.(col + i))
      done ;
      c.((row * n) + col) <- sum]

let n_equal_strides =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + (i * k)) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_no_unit_stride =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k * 2l) + (i * 2l)) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_small_ld =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to 31l do
        sum := sum +. (a.((row * 4l) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_dependent_indices =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let open Std in
      let tid = global_thread_id in
      let row = tid / 4l in
      let col = tid / 8l in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_init_elsewhere =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut c.(0) in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]

let n_stencil =
  [%kernel
    fun (a : float32 vector) (out : float32 vector) ->
      let i = global_idx_x in
      out.(i) <- a.(i - 1l) +. a.(i) +. a.(i + 1l)]

let test_negatives () =
  let is r = Some (fun x -> x = r) in
  let matches f = Some f in
  check_rejected "dot" n_dot (is P.No_row_column_split) ;
  check_rejected "row_sum" n_row_sum (is P.Not_a_product) ;
  check_rejected "matvec" n_matvec (is P.No_row_column_split) ;
  check_rejected "linear_output" n_linear_output (is P.No_row_column_split) ;
  check_rejected
    "modulo"
    n_modulo_index
    (matches (function P.Index_not_affine ("a", _) -> true | _ -> false)) ;
  check_rejected
    "square"
    n_square_index
    (matches (function P.Index_not_affine ("a", _) -> true | _ -> false)) ;
  check_rejected
    "gather"
    n_gather
    (matches (function P.Index_not_affine ("a", _) -> true | _ -> false)) ;
  check_rejected "data_bound" n_data_bound (is P.Bounds_data_dependent) ;
  check_rejected "triangular" n_triangular (is P.Bounds_not_uniform) ;
  check_rejected "downto" n_downto (is P.Downto_loop) ;
  check_rejected
    "three_factors"
    n_three_factors
    (matches (function P.Operand_not_an_element _ -> true | _ -> false)) ;
  check_rejected "subtraction" n_subtraction (is P.Not_a_product) ;
  check_rejected "product_reduction" n_product_reduction (is P.Not_a_product) ;
  check_rejected
    "aliased_output"
    n_aliased_output
    (is (P.Output_aliases_operand "a")) ;
  check_rejected
    "operand_written"
    n_operand_written
    (is (P.Operand_written "b")) ;
  check_rejected "guard_reads_acc" n_guard_reads_acc (is P.No_output_store) ;
  check_rejected
    "varying_epilogue"
    n_varying_epilogue
    (is P.Unsupported_epilogue) ;
  check_rejected
    "shared_tiled"
    n_shared_tiled
    (matches (function P.Operand_not_global _ -> true | _ -> false)) ;
  let overlap arr =
    matches (function P.Overlapping_operand a -> a = arr | _ -> false)
  in
  check_rejected "overlapping_output" n_overlapping_output (overlap "c") ;
  check_rejected "toeplitz" n_toeplitz (overlap "a") ;
  check_rejected "correlation" n_correlation (overlap "a") ;
  check_rejected "equal_strides" n_equal_strides (overlap "a") ;
  check_rejected
    "no_unit_stride"
    n_no_unit_stride
    (is (P.Operand_not_unit_stride "a")) ;
  check_rejected "small_ld" n_small_ld (overlap "a") ;
  check_rejected
    "dependent_indices"
    n_dependent_indices
    (is P.Dependent_indices) ;
  check_rejected "init_elsewhere" n_init_elsewhere (is P.Unsupported_init) ;
  check_rejected "stencil" n_stencil None

let () =
  Alcotest.run
    "matrix_pattern_negative"
    [("corpus", [Alcotest.test_case "negatives" `Quick test_negatives])]
