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

let digest_of kernel lowering =
  let ir = ir_of kernel in
  let g = only_gemm "golden" kernel in
  match Rw.kernel ir g lowering with
  | Ok rewritten ->
      Digest.to_hex
        (Digest.string (Format.asprintf "%a" Sarek_ir_pp.pp_kernel rewritten))
  | Error why -> "declined: " ^ why

let golden =
  [
    ( "int32 byte planes",
      k_int,
      Rw.Int32_byte_planes,
      "65add5b38a2cfc22766b1ceab9fcec90" );
    ("f16 operands", k_f32, Rw.F16_operands, "a421d8fc01a72a5a8a07df6e073bb8c4");
    ( "bf16 operands",
      k_f32,
      Rw.Bf16_operands,
      "c6365097e001ff3f05d5bdadaff793a0" );
    ( "bf16x3 operands",
      k_f32,
      Rw.Bf16x3_operands,
      "0299f08cb99159425bf8120500016c25" );
  ]

let () =
  Alcotest.run
    "matrix template golden"
    [
      ( "wave32",
        List.map
          (fun (name, kernel, lowering, expected) ->
            Alcotest.test_case name `Quick (fun () ->
                Alcotest.(check string)
                  name
                  expected
                  (digest_of kernel lowering)))
          golden );
    ]
