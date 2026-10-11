(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: An atomic result guards a barrier. The intrinsic is declared
   ThreadVarying, which the analysis did not consult.

   Global review 2026-10-11 C-4, probe p08, scenario G-12; specs/convergence-semantics.md.
   Compiled without error before the variance analysis. Expected:
   "Barrier called in diverged control flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) (n : int32) ->
        let%shared (s : int32) = 4l in
        let old = atomic_add_int32 s 0l 1l in
        if old = 0l then block_barrier ()]
  in
  print_endline "This should not print - test should have failed to compile"
