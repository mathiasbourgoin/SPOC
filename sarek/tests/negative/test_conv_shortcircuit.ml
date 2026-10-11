(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: The right operand of && runs only on threads whose left operand is
   true: a barrier there is in diverged flow.

   Global review 2026-10-11 C-24, probe p10; specs/convergence-semantics.md.
   Compiled without error before the variance analysis. Expected:
   "Barrier called in diverged control flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) (n : int32) ->
        if
          thread_idx_x > 16l
          &&
          (block_barrier () ;
           true)
        then v.(0l) <- 1l]
  in
  print_endline "This should not print - test should have failed to compile"
