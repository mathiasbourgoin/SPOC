(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The divergent-barrier refusal also holds under [let open Sarek_stdlib.Std in
   ...], the form every documented kernel uses (#447). *)

let () =
  let _bad_kernel =
    [%kernel
      fun (a : float32 vector) ->
        let open Sarek_stdlib.Std in
        if thread_idx_x > 16l then block_barrier () ;
        a.(thread_idx_x) <- 0.0]
  in
  print_endline "This should not print - test should have failed to compile"
