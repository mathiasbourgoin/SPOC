(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A shuffling helper called through an alias (let g = reduce) in diverged flow. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let reduce (x : int32) = warp_shuffle_xor x 1l in
        let g = reduce in
        if thread_idx_x > 16l then v.(thread_idx_x) <- g v.(thread_idx_x)]
  in
  print_endline "This should not print - test should have failed to compile"
