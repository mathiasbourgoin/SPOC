(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A helper that contains a barrier, called under a thread-varying
   condition (rule H-2: rejected at the call).

   Global review 2026-10-11 C-3, probe p05, scenario G-9; specs/convergence-semantics.md.
   Compiled without error before the variance analysis. Expected:
   "Barrier called in diverged control flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) (n : int32) ->
        let sync (u : unit) = block_barrier () in
        if thread_idx_x > 16l then sync ()]
  in
  print_endline "This should not print - test should have failed to compile"
