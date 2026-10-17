(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A warp shuffle moves int32, float32 or float64 values: a bool is refused,
   with a located message naming the primitive and the type. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let b = warp_shuffle (v.(thread_idx_x) > 0l) 1l in
        if b then v.(thread_idx_x) <- 1l]
  in
  print_endline "This should not print - test should have failed to compile"
