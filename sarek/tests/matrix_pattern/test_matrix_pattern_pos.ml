(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Matrix-product pattern corpus, POSITIVE half (specs/matrix-pattern-advice.md
 * US-1, US-2). Compiled with [-alert @sarek_matrix] (see dune): every advised
 * kernel below carries a silencing attribute, so an attribute that does not
 * silence fails the build.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std
module P = Sarek_matrix_pattern
module Adv = Sarek_matrix_advice
open Matrix_pattern_check

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

(* Padded leading dimensions (k + p, n + p): two-monomial strides. *)
let p_padded_ld =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32)
        (p : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * (k + p)) + i) *. b.((i * n) + col))
      done ;
      c.((row * (n + p)) + col) <- sum]
[@@alert "-sarek_matrix"]

(* Accumulating into C: the accumulator starts from the output element. *)
let p_init_from_c =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut c.((row * n) + col) in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum]
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
      "padded_ld"
      p_padded_ld
      ~a:"a"
      ~b:"b"
      ~c:"c"
      ~layouts:(rm, rm, rm)
      ~batch:0
  in
  Alcotest.(check string)
    "padded_ld: m is the row"
    "Sarek_stdlib.Gpu.global_idx_y"
    (Format.asprintf "%a" Sarek_ir_pp.pp_expr g.m_index) ;
  let g =
    check_roles
      "init_from_c"
      p_init_from_c
      ~a:"a"
      ~b:"b"
      ~c:"c"
      ~layouts:(rm, rm, rm)
      ~batch:0
  in
  Alcotest.(check bool)
    "init_from_c: init rendered"
    true
    (Astring.String.is_infix ~affix:"c[" (Adv.render g (Adv.advise g))) ;
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

let () =
  Alcotest.run
    "matrix_pattern_positive"
    [
      ( "corpus",
        [
          Alcotest.test_case "positives" `Quick test_positives;
          Alcotest.test_case "advice text" `Quick test_advice_text;
        ] );
    ]
