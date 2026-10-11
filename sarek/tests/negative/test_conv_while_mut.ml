(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A loop counter advanced by thread_idx_x: the loop condition becomes
   thread-varying from the second iteration (loop-carried variance).

   Global review 2026-10-11 C-1, probe p11; specs/convergence-semantics.md.
   Compiled without error before the variance analysis. Expected:
   "Barrier called in diverged control flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) (n : int32) ->
        let i = mut 0l in
        while i < n do
          block_barrier () ;
          i := i + thread_idx_x
        done]
  in
  print_endline "This should not print - test should have failed to compile"
