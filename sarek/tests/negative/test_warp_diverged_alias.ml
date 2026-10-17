(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A helper reached through an alias (let g = gate) with a thread-varying parameter. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let gate (k : int32) (x : int32) =
          if k > 16l then warp_shuffle_xor x 1l else x
        in
        let g = gate in
        v.(thread_idx_x) <- g thread_idx_x v.(thread_idx_x)]
  in
  print_endline "This should not print - test should have failed to compile"
