(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The condition is the value of a thread-varying if expression (bound by let), so it is thread-varying. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let c = if thread_idx_x > 16 then 1l else 0l in
        if c > 0l then v.(thread_idx_x) <- warp_shuffle_xor v.(thread_idx_x) 1l]
  in
  print_endline "This should not print - test should have failed to compile"
