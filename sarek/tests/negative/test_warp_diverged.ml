(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile.

   It calls warp_shuffle (a warp collective) inside a thread-varying
   conditional: only the threads above 16 would take part. The warp shuffles
   are lowered on every backend, so the convergence check is what refuses it,
   with "Warp collective 'warp_shuffle' called in diverged control flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        if thread_idx_x > 16 then
          let x = warp_shuffle v.(thread_idx_x) 1l in
          v.(thread_idx_x) <- x]
  in
  print_endline "This should not print - test should have failed to compile"
