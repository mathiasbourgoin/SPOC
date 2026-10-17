(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The right operand of && runs only for the threads where the left one holds, so a shuffle there is divergent. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        if thread_idx_x > 16 && warp_shuffle_xor v.(thread_idx_x) 1l > 0l then
          v.(thread_idx_x) <- 0l]
  in
  print_endline "This should not print - test should have failed to compile"
