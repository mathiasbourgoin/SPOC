(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: a [@divergent] superstep reached under a thread-varying
   condition (global review C-7, probe p09).

   specs/convergence-semantics.md SS-1 and SS-4. [@divergent] used to skip the
   entry check, so this compiled and lowered to SBarrier inside SIf: only the
   threads with thread_idx_x > 16 reach the implicit end-of-step barrier. The
   attribute is now inert. Expected: "Barrier called in diverged control
   flow". *)

let () =
  let _bad_kernel =
    [%kernel
      fun (input : float32 vector) (output : float32 vector) ->
        let%shared (tile : float32) = () in
        if thread_idx_x > 16l then begin
          let%superstep[@divergent] load =
            tile.(thread_idx_x) <- input.(global_thread_id)
          in
          output.(global_thread_id) <- tile.(thread_idx_x)
        end]
  in
  print_endline "This should not print - test should have failed to compile"
