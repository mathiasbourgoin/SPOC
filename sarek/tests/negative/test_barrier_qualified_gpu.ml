(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The divergent-barrier refusal also holds for the module-qualified spelling
   [Gpu.block_barrier] (#447). *)

let () =
  let _bad_kernel =
    [%kernel
      fun (a : float32 vector) ->
        if thread_idx_x > 16l then Gpu.block_barrier () ;
        a.(thread_idx_x) <- 0.0]
  in
  print_endline "This should not print - test should have failed to compile"
