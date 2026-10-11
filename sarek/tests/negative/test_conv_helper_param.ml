(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A helper whose barrier is guarded by its parameter, called with
   thread_idx_x: the parameter may vary, so the barrier is in diverged flow.

   Global review 2026-10-11 C-3, scenario G-10, rule H-1; specs/convergence-semantics.md.
   Compiled without error before the variance analysis. Expected:
   "Barrier called in diverged control flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) (n : int32) ->
        let f (m : int32) = if m > 0l then block_barrier () in
        f thread_idx_x]
  in
  print_endline "This should not print - test should have failed to compile"
