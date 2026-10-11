(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The refusal also holds under [let open Std in ...]. *)

(* open Sarek_stdlib: the generated native code names Gpu/Std, so without it
   this case would also fail for an unbound module, masking the refusal. *)
open Sarek_stdlib

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let open Std in
        v.(thread_idx_x) <- warp_shuffle_xor v.(thread_idx_x) 1l]
  in
  print_endline "This should not print - test should have failed to compile"
