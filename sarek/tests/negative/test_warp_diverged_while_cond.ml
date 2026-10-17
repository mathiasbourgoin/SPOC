(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A shuffle in the condition of a thread-varying while loop: from the second
   iteration only the threads still looping evaluate it. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let i = mut thread_idx_x in
        while warp_shuffle_xor i 1l < 20l do
          i := i + 1l
        done ;
        v.(thread_idx_x) <- i]
  in
  print_endline "This should not print - test should have failed to compile"
