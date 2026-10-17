(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The condition is a helper result that depends on the thread (the helper reads thread_idx_x), although its argument is uniform. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let me (k : int32) = thread_idx_x + k in
        if me 0l > 16l then
          v.(thread_idx_x) <- warp_shuffle_xor v.(thread_idx_x) 1l]
  in
  print_endline "This should not print - test should have failed to compile"
