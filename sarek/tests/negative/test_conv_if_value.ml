(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: The value of a thread-varying if-expression guards a barrier. If and
   match values used to default to uniform.

   Global review 2026-10-11 C-2, probe p03, scenario G-7; specs/convergence-semantics.md.
   Compiled without error before the variance analysis. Expected:
   "Barrier called in diverged control flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) (n : int32) ->
        let x = if thread_idx_x > 16l then 1l else 0l in
        if x = 1l then block_barrier ()]
  in
  print_endline "This should not print - test should have failed to compile"
