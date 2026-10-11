(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (specs/matrix-pattern-advice.md, US-1): a naive GEMM with no
   silencing attribute, compiled with -alert @sarek_matrix. The advice is an
   alert, so the build must fail with its text, located at the [for] loop. *)

let gemm =
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
        for kk = 0 to k - 1l do
          sum :=
            sum
            +. float32_of_float16 a.((row * k) + kk)
               *. float32_of_float16 b.((kk * n) + col)
        done ;
        c.((row * n) + col) <- sum
      end]
