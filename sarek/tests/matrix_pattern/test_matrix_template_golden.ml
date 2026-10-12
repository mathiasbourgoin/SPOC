(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The rewritten kernel for 32-wide subgroups (HIP gfx11, CUDA warps) is
 * byte-identical to what it was before the templates were parameterised by
 * subgroup width (H4, specs/matrix-pattern-rewrite.md): the digests below were
 * taken from the unparameterised code and must not move.
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

open Sarek
module Rw = Sarek_matrix_rewrite
open Matrix_pattern_check

let k_int =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0l in
        for i = 0 to k - 1l do
          sum := sum + (a.((row * k) + i) * b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

let k_f32 =
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
          sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

let k_f16 =
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
            sum
            +. float32_of_float16 a.((row * k) + i)
               *. float32_of_float16 b.((i * n) + col)
        done ;
        c.((row * n) + col) <- sum
      end]

let digest_of ?subgroup kernel lowering =
  let ir = ir_of kernel in
  let g = only_gemm "golden" kernel in
  match Rw.kernel ?subgroup ir g lowering with
  | Ok rewritten ->
      Digest.to_hex
        (Digest.string (Format.asprintf "%a" Sarek_ir_pp.pp_kernel rewritten))
  | Error why -> "declined: " ^ why

(* (name, kernel, lowering, subgroup width, digest). The 32-wide digests of the
   first four were taken from the unparameterised code; the other three were
   computed on the code before the fragment parameter existed (b323cc79) and
   equal the output after it (H4b), at fragment 16. *)
let golden =
  [
    ( "int32 byte planes",
      k_int,
      Rw.Int32_byte_planes,
      32,
      "65add5b38a2cfc22766b1ceab9fcec90" );
    ( "f16 operands",
      k_f32,
      Rw.F16_operands,
      32,
      "a421d8fc01a72a5a8a07df6e073bb8c4" );
    ( "bf16 operands",
      k_f32,
      Rw.Bf16_operands,
      32,
      "c6365097e001ff3f05d5bdadaff793a0" );
    ( "bf16x3 operands",
      k_f32,
      Rw.Bf16x3_operands,
      32,
      "0299f08cb99159425bf8120500016c25" );
    ( "int32 byte planes, 64 lanes",
      k_int,
      Rw.Int32_byte_planes,
      64,
      "6fd5f7627d58e77dbbbc35be865c44c1" );
    ( "f16 operands, float16 inputs",
      k_f16,
      Rw.F16_operands,
      32,
      "4e49be53fab5bc4cda5b93a6b557ffa7" );
    ( "f16 operands, float16 inputs, 64 lanes",
      k_f16,
      Rw.F16_operands,
      64,
      "92342692d1bc79773f807f252590d36d" );
  ]

let () =
  Alcotest.run
    "matrix template golden"
    [
      ( "fragment 16",
        List.map
          (fun (name, kernel, lowering, subgroup, expected) ->
            Alcotest.test_case name `Quick (fun () ->
                Alcotest.(check string)
                  name
                  expected
                  (digest_of ~subgroup kernel lowering)))
          golden );
    ]
