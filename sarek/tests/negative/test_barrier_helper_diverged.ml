(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A block barrier in a helper function that the kernel calls from thread-varying
   control flow: the same transitive rule (H-2) as for warp collectives. Before
   it was implemented a barrier hidden in a helper escaped the divergence
   check. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let sync (x : int32) =
          block_barrier () ;
          x
        in
        if thread_idx_x > 16 then v.(thread_idx_x) <- sync v.(thread_idx_x)]
  in
  print_endline "This should not print - test should have failed to compile"
