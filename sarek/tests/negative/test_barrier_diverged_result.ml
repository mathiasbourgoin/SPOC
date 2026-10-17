(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A helper returns the thread index; the barrier under a test on its result is divergent although no argument of the call varies. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let tid (z : int32) = thread_idx_x + z in
        let t = tid 0l in
        if t > 16l then block_barrier () ;
        v.(thread_idx_x) <- t]
  in
  print_endline "This should not print - test should have failed to compile"
