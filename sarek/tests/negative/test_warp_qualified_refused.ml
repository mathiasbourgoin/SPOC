(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The refusal keys on the resolved intrinsic, not the spelling: a
   module-qualified Gpu.warp_ballot is refused like the bare name. *)

(* open Sarek_stdlib: the generated native code names Gpu/Std, so without it
   this case would also fail for an unbound module, masking the refusal. *)
open Sarek_stdlib

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        v.(thread_idx_x) <- Gpu.warp_ballot (v.(thread_idx_x) > 0l)]
  in
  print_endline "This should not print - test should have failed to compile"
