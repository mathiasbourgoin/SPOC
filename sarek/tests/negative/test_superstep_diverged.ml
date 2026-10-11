(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: a superstep reached under a thread-varying condition.

   specs/convergence-semantics.md SS-1. Only the threads with thread_idx_x > 16
   enter the superstep, so only they reach its implicit end-of-step barrier:
   barrier divergence. Expected: "Barrier called in diverged control flow".

   (This file used to expect the rejection of a thread-varying `if` INSIDE a
   superstep body. That case is legal under SS-2 -- structured control flow
   reconverges before the implicit barrier -- and compiled; the target printed
   a non-blocking KNOWN-ISSUE (D-38). It is now a positive case in
   sarek/tests/e2e/test_superstep.ml.) *)

let () =
  let _bad_kernel =
    [%kernel
      fun (input : float32 vector) (output : float32 vector) ->
        let%shared (tile : float32) = () in
        if thread_idx_x > 16l then begin
          let%superstep load =
            tile.(thread_idx_x) <- input.(global_thread_id)
          in
          output.(global_thread_id) <- tile.(thread_idx_x)
        end]
  in
  print_endline "This should not print - test should have failed to compile"
