(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A helper parameter is thread-varying at the call (the argument is thread_idx_x), so the shuffle under the helper's own `if n > 16` is divergent. Refused with the warp convergence message. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let gate (n : int32) (x : int32) =
          if n > 16l then warp_shuffle_xor x 1l else x
        in
        v.(thread_idx_x) <- gate thread_idx_x v.(thread_idx_x)]
  in
  print_endline "This should not print - test should have failed to compile"
