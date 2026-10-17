(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A warp shuffle inside a loop whose bound varies per thread: the number of
   iterations differs between lanes, so the lanes would not reach the shuffle
   together. Refused by the convergence check. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let i = mut 0l in
        while i < thread_idx_x do
          v.(thread_idx_x) <- warp_shuffle_xor v.(thread_idx_x) 1l ;
          i := i + 1l
        done]
  in
  print_endline "This should not print - test should have failed to compile"
