(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A per-thread (Local) array filled with thread_idx_x guards a barrier.
   An array read used to vary only through its index.

   Global review 2026-10-11 C-5, probe p07, scenario G-13; specs/convergence-semantics.md.
   Compiled without error before the variance analysis. Expected:
   "Barrier called in diverged control flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) (n : int32) ->
        let a = create_array 4l Local in
        a.(0l) <- thread_idx_x ;
        if a.(0l) > 16l then block_barrier ()]
  in
  print_endline "This should not print - test should have failed to compile"
