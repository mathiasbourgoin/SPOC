(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* return_unit: CUDA emits [return((void)0);] and skips later barriers, Native
   continues, GLSL has no lowering. Sarek has no early return; refused at PPX
   time with advice to restructure with if/else. *)

(* open Sarek_stdlib: the generated native code names Gpu/Std, so without it
   this case would also fail for an unbound module, masking the refusal. *)
open Sarek_stdlib

let () =
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        if thread_idx_x > 16 then return_unit () ;
        v.(thread_idx_x) <- 1l]
  in
  print_endline "This should not print - test should have failed to compile"
