(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Two module-level helpers share a name: the call resolves to the later one, whose barrier sits under a test on its parameter. *)

let () =
  let _k =
    [%kernel
      let f (x : int32) : int32 = x + 1l in
      let f (x : int32) : int32 =
        if x > 16l then block_barrier () ;
        x
      in
      fun (v : int32 vector) -> v.(thread_idx_x) <- f thread_idx_x]
  in
  print_endline "This should not print - test should have failed to compile"
