(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A helper returning thread_idx_x guards a barrier. Calls were judged by
   their arguments only.

   Global review 2026-10-11 C-3, probe p04, scenario G-8; specs/convergence-semantics.md.
   Compiled without error before the variance analysis. Expected:
   "Barrier called in diverged control flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) (n : int32) ->
        let tid (u : unit) = thread_idx_x in
        if tid () > 16l then block_barrier ()]
  in
  print_endline "This should not print - test should have failed to compile"
