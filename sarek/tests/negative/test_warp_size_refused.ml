(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* warp_size used as a value. It was typed as the constant 32, which is wrong on
   AMD (32 or 64) and WGSL (4-128); no backend lowers it. Refused at PPX time. *)

(* open Sarek_stdlib: the generated native code names Gpu/Std, so without it
   this case would also fail for an unbound module, masking the refusal. *)
open Sarek_stdlib

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        let w = warp_size in
        v.(thread_idx_x) <- w]
  in
  print_endline "This should not print - test should have failed to compile"
