(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile.

   It calls warp_shuffle (a warp collective) inside a thread-varying
   conditional. Two checks apply to it, in this order:

   1. The refusal of unlowered intrinsics (Sarek_env.unlowered_intrinsics): no
      backend lowers warp_shuffle yet, so the PPX refuses it at the call with
      "'warp_shuffle' is not lowered by any backend yet". This fires while
      typing and is what this case now asserts.
   2. The convergence check ("Warp collective 'warp_shuffle' called in diverged
      control flow"), which this source used to reach. It is no longer
      reachable from source while warp_shuffle is refused; it stays covered on
      hand-built typed ASTs by sarek/tests/unit/test_convergence.ml
      (test_check_expr_warp_collective_diverged), so warp-scope supersteps
      (roadmap S1) start from a checked base. When warp_shuffle is lowered and
      removed from the refusal table, assert message 2 here again. *)

let () =
  (* Refused as unlowered; the divergence (only threads > 16 participate) is
     what the convergence check would otherwise report. *)
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        if thread_idx_x > 16 then
          let x = warp_shuffle v.(thread_idx_x) 1l in
          v.(thread_idx_x) <- x]
  in
  print_endline "This should not print - test should have failed to compile"
