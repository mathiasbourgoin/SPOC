(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Matrix-product pattern corpus (specs/matrix-pattern-advice.md, US-1..US-3).
 *
 * This file is compiled with [-alert @sarek_matrix] (see dune): every POSITIVE
 * kernel below carries an alert-silencing attribute, every NEGATIVE kernel
 * carries none. A false positive, or a silencing attribute that does not
 * silence, is therefore a build error, before any of the run-time checks.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std
module P = Sarek_matrix_pattern
module Adv = Sarek_matrix_advice

(* ========================================================================= *)
(* Positive corpus                                                            *)
(* ========================================================================= *)

(* 1-D mapping, float32, silenced on the binding. *)
let p_naive_1d =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let open Std in
      let tid = global_thread_id in
      let row = tid / n in
      let col = tid mod n in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]
[@@alert "-sarek_matrix"]

(* float16 operands, fma, 2-D mapping, silenced on the expression. *)
let p_f16_fma =
  [%kernel
    fun (a : float16 vector)
        (b : float16 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum :=
            fma
              (float32_of_float16 a.((row * k) + i))
              (float32_of_float16 b.((i * n) + col))
              sum
        done ;
        c.((row * n) + col) <- sum
      end]
  [@alert "-sarek_matrix"]

(* int32, both operands transposed, row/col over two atoms each. *)
let p_int_transposed =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = thread_idx_y + (block_dim_y * block_idx_y) in
      let col = thread_idx_x + (block_dim_x * block_idx_x) in
      if row < m && col < n then begin
        let sum = mut 0l in
        for i = 0 to k - 1l do
          sum := sum + (a.((i * m) + row) * b.((col * k) + i))
        done ;
        c.((row * n) + col) <- sum
      end]
[@@alert "-sarek_matrix"]

(* Column-major output, product written B * A (reported transposed). *)
let p_col_major_c =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum := sum +. (b.((i * n) + col) *. a.((row * k) + i))
        done ;
        c.((col * m) + row) <- sum
      end]
[@@alert "-sarek_matrix"]

(* A loop nest with no thread index at all. *)
let p_loop_nest =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      for row = 0 to m - 1l do
        for col = 0 to n - 1l do
          let sum = mut 0.0 in
          for i = 0 to k - 1l do
            sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
          done ;
          c.((row * n) + col) <- sum
        done
      done]
[@@alert "-sarek_matrix"]

(* Batched over z (A, B and C), operands through lets, an offset on A and an
   alpha/beta epilogue. *)
let p_batched_epilogue =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32)
        (off : int32)
        (alpha : float32)
        (beta : float32) ->
      let z = global_idx_z in
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          let x = a.(off + (z * m * k) + (row * k) + i) in
          let y = b.((z * k * n) + (i * n) + col) in
          sum := sum +. (x *. y)
        done ;
        let o = (z * m * n) + (row * n) + col in
        c.(o) <- (alpha *. sum) +. (beta *. c.(o))
      end]
[@@alert "-sarek_matrix"]

(* Batched A and C with a broadcast B: z folds into the row index (one GEMM
   with z*m + row as M), since A and C stack their batches contiguously. *)
let p_broadcast_b =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let z = global_idx_z in
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((z * m * k) + (row * k) + i) *. b.((i * n) + col))
      done ;
      c.((z * m * n) + (row * n) + col) <- sum]
[@@alert "-sarek_matrix"]

(* The same with a padded batch stride on A (zs, not m*k): z cannot be part of
   the row index, so it is a batch atom with B broadcast. *)
let p_broadcast_padded =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32)
        (zs : int32) ->
      let z = global_idx_z in
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((z * zs) + (row * k) + i) *. b.((i * n) + col))
      done ;
      c.((z * m * n) + (row * n) + col) <- sum]
[@@alert "-sarek_matrix"]

(* int64 is recognised but has no matrix-unit configuration: no advice, so no
   silencing attribute is needed (and the build proves none is emitted). *)
let p_int64_no_config =
  [%kernel
    fun (a : int64 vector)
        (b : int64 vector)
        (c : int64 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0L in
        for i = 0 to k - 1l do
          sum := sum + (a.((row * k) + i) * b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

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

let n_stencil =
  [%kernel
    fun (a : float32 vector) (out : float32 vector) ->
      let i = global_idx_x in
      out.(i) <- a.(i - 1l) +. a.(i) +. a.(i + 1l)]

(* ========================================================================= *)
(* Checks                                                                     *)
(* ========================================================================= *)

let ir_of (_, kirc) =
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> Alcotest.fail "kernel has no IR"

let only_gemm name kernel =
  match P.analyze (ir_of kernel) with
  | [P.Gemm g] -> g
  | findings ->
      Alcotest.failf
        "%s: expected exactly one GEMM finding, got %d findings"
        name
        (List.length findings)

let layout = Alcotest.testable (Fmt.of_to_string P.layout_name) ( = )

let check_roles name kernel ~a ~b ~c ~layouts:(la, lb, lc) ~batch =
  let g = only_gemm name kernel in
  Alcotest.(check string) (name ^ ": A") a g.a.array ;
  Alcotest.(check string) (name ^ ": B") b g.b.array ;
  Alcotest.(check string) (name ^ ": C") c g.c.array ;
  Alcotest.check layout (name ^ ": A layout") la g.a.layout ;
  Alcotest.check layout (name ^ ": B layout") lb g.b.layout ;
  Alcotest.check layout (name ^ ": C layout") lc g.c.layout ;
  Alcotest.(check int) (name ^ ": batch atoms") batch (List.length g.batch) ;
  g

let case_name = function
  | Adv.F16_inputs -> "f16"
  | Adv.F32_inputs -> "f32"
  | Adv.Int32_inputs -> "int32"
  | Adv.No_case -> "none"

let check_case name g expected =
  Alcotest.(check string)
    (name ^ ": element case")
    expected
    (case_name (Adv.advise g).case)

let test_positives () =
  let rm = P.Row_major and cm = P.Col_major in
  let g =
    check_roles
      "naive_1d"
      p_naive_1d
      ~a:"a"
      ~b:"b"
      ~c:"c"
      ~layouts:(rm, rm, rm)
      ~batch:0
  in
  check_case "naive_1d" g "f32" ;
  let g =
    check_roles
      "f16_fma"
      p_f16_fma
      ~a:"a"
      ~b:"b"
      ~c:"c"
      ~layouts:(rm, rm, rm)
      ~batch:0
  in
  check_case "f16_fma" g "f16" ;
  Alcotest.(check bool) "f16_fma: fused" true (g.update = P.Fused_multiply_add) ;
  let g =
    check_roles
      "int_transposed"
      p_int_transposed
      ~a:"a"
      ~b:"b"
      ~c:"c"
      ~layouts:(cm, cm, rm)
      ~batch:0
  in
  check_case "int_transposed" g "int32" ;
  (* A column-major C = A.B is reported as the row-major C^T = B^T.A^T: the
     operands are named so that C's unit stride is its column stride. *)
  ignore
    (check_roles
       "col_major_c"
       p_col_major_c
       ~a:"b"
       ~b:"a"
       ~c:"c"
       ~layouts:(cm, cm, rm)
       ~batch:0) ;
  let g =
    check_roles
      "loop_nest"
      p_loop_nest
      ~a:"a"
      ~b:"b"
      ~c:"c"
      ~layouts:(rm, rm, rm)
      ~batch:0
  in
  Alcotest.(check int)
    "loop_nest: enclosing loops"
    2
    (List.length g.enclosing_loops) ;
  let g =
    check_roles
      "batched"
      p_batched_epilogue
      ~a:"a"
      ~b:"b"
      ~c:"c"
      ~layouts:(rm, rm, rm)
      ~batch:1
  in
  Alcotest.(check bool)
    "batched: epilogue kept"
    true
    (match g.epilogue with P.Epilogue _ -> true | P.Plain -> false) ;
  let g =
    check_roles
      "broadcast_b"
      p_broadcast_b
      ~a:"a"
      ~b:"b"
      ~c:"c"
      ~layouts:(rm, rm, rm)
      ~batch:0
  in
  Alcotest.(check bool)
    "broadcast_b: z is part of the row index"
    true
    (Astring.String.is_infix
       ~affix:"global_idx_z"
       (Format.asprintf "%a" Sarek_ir_pp.pp_expr g.m_index)) ;
  ignore
    (check_roles
       "broadcast_padded"
       p_broadcast_padded
       ~a:"a"
       ~b:"b"
       ~c:"c"
       ~layouts:(rm, rm, rm)
       ~batch:1) ;
  let g =
    check_roles
      "int64"
      p_int64_no_config
      ~a:"a"
      ~b:"b"
      ~c:"c"
      ~layouts:(rm, rm, rm)
      ~batch:0
  in
  check_case "int64" g "none" ;
  Alcotest.(check bool)
    "int64: no matrix unit"
    false
    (Adv.has_matrix_unit (Adv.advise g))

(* The advice text names what it found and the opt-in that applies. *)
let test_advice_text () =
  let contains s sub = Astring.String.is_infix ~affix:sub s in
  let text kernel =
    let g = only_gemm "text" kernel in
    Adv.render ~kernel_name:"k" g (Adv.advise g)
  in
  let f16 = text p_f16_fma in
  List.iter
    (fun sub ->
      Alcotest.(check bool) ("f16 text has " ^ sub) true (contains f16 sub))
    [
      "kernel 'k'";
      "f16*f16+f32->f32";
      "accept_relaxed_accumulation";
      "HIP gfx11";
      "-sarek_matrix";
    ] ;
  let int32 = text p_int_transposed in
  List.iter
    (fun sub ->
      Alcotest.(check bool) ("int32 text has " ^ sub) true (contains int32 sub))
    ["u8*u8+s32->s32"; "byte planes"; "exact"; "column-major"]

let reasons kernel =
  List.filter_map
    (function P.Near_miss nm -> Some nm.nm_reason | P.Gemm _ -> None)
    (P.analyze (ir_of kernel))

let check_rejected name kernel expected =
  let findings = P.analyze (ir_of kernel) in
  Alcotest.(check bool)
    (name ^ ": no GEMM finding")
    false
    (List.exists
       (function P.Gemm _ -> true | P.Near_miss _ -> false)
       findings) ;
  match expected with
  | None ->
      Alcotest.(check int)
        (name ^ ": no reduction loop")
        0
        (List.length findings)
  | Some pred ->
      Alcotest.(check bool)
        (name ^ ": reason "
        ^ String.concat "; " (List.map P.reason_text (reasons kernel)))
        true
        (List.exists pred (reasons kernel))

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
  check_rejected
    "shared_tiled"
    n_shared_tiled
    (matches (function P.Operand_not_global _ -> true | _ -> false)) ;
  check_rejected "stencil" n_stencil None

let () =
  Alcotest.run
    "matrix_pattern"
    [
      ( "corpus",
        [
          Alcotest.test_case "positives" `Quick test_positives;
          Alcotest.test_case "advice text" `Quick test_advice_text;
          Alcotest.test_case "negatives" `Quick test_negatives;
        ] );
    ]
