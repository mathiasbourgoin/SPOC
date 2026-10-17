(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The lane operand of a warp shuffle is an int32: a float32 operand is a type
   error. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : float32 vector) ->
        v.(thread_idx_x) <- warp_shuffle_down v.(thread_idx_x) 1.0]
  in
  print_endline "This should not print - test should have failed to compile"
