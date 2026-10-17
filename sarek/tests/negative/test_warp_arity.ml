(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A warp shuffle with one argument: the lane operand is missing. The shared
   arity rule of the typer reports "Wrong number of arguments". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        v.(thread_idx_x) <- warp_shuffle_down v.(thread_idx_x)]
  in
  print_endline "This should not print - test should have failed to compile"
