(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A mutable variable assigned a thread-varying value guards a barrier. Assignment used
   never to mark the variable varying.

   Global review 2026-10-11 C-1, probe p01, scenario G-5; specs/convergence-semantics.md.
   Compiled without error before the variance analysis. Expected:
   "Barrier called in diverged control flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) (n : int32) ->
        let x = mut 0l in
        x := thread_idx_x ;
        if x > 16l then block_barrier ()]
  in
  print_endline "This should not print - test should have failed to compile"
