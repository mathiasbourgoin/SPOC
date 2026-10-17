(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A mutable is assigned under a thread-varying condition, so its later value differs per thread; the barrier under a test on it is divergent. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let m = mut 0l in
        if thread_idx_x > 16 then m := 1l ;
        if m > 0l then block_barrier () ;
        v.(thread_idx_x) <- m]
  in
  print_endline "This should not print - test should have failed to compile"
