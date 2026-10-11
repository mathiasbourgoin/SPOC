(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A uniform value assigned under a thread-varying condition: the variable
   differs between threads afterwards (control dependence).

   Global review 2026-10-11 C-1, probe p02, scenario G-6; specs/convergence-semantics.md.
   Compiled without error before the variance analysis. Expected:
   "Barrier called in diverged control flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) (n : int32) ->
        let x = mut 0l in
        if thread_idx_x > 16l then x := 1l ;
        if x = 1l then block_barrier ()]
  in
  print_endline "This should not print - test should have failed to compile"
