(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A warp shuffle in a helper function that the kernel calls from thread-varying
   control flow (rule H-2 of specs/convergence-semantics.md, the transitive
   rule): the call is reached in diverged flow, so the shuffle inside the
   helper is too, and the call site is refused. *)

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let reduce (x : int32) = warp_shuffle_xor x 1l in
        if thread_idx_x > 16 then v.(thread_idx_x) <- reduce v.(thread_idx_x)]
  in
  print_endline "This should not print - test should have failed to compile"
