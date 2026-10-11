(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A helper whose body is a superstep (an implicit barrier), called under
   a thread-varying condition.

   Global review 2026-10-11 C-3, rule H-2; specs/convergence-semantics.md.
   Compiled without error before the variance analysis. Expected:
   "Barrier called in diverged control flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) (n : int32) ->
        let step (u : unit) =
          let%superstep s = v.(0l) <- 1l in
          ()
        in
        if thread_idx_x > 16l then step ()]
  in
  print_endline "This should not print - test should have failed to compile"
