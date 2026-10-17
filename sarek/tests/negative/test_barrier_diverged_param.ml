(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A helper whose barrier sits under a test on its parameter, called with a thread-varying argument: only some threads reach the barrier. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let sync (x : int32) =
          if x > 16l then block_barrier () ;
          x
        in
        v.(thread_idx_x) <- sync thread_idx_x]
  in
  print_endline "This should not print - test should have failed to compile"
