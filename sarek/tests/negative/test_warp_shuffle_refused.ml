(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A warp collective in FULLY CONVERGED code. No backend lowers warp_shuffle, so
   the PPX refuses it at compile time, located at the call, independently of the
   convergence check. *)

(* open Sarek_stdlib: the generated native code names Gpu/Std, so without it
   this case would also fail for an unbound module, masking the refusal. *)
open Sarek_stdlib

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let i = thread_idx_x in
        v.(i) <- warp_shuffle v.(i) 1l]
  in
  print_endline "This should not print - test should have failed to compile"
