(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The explicit entry point to the matrix-product rewrite. Since
   specs/matrix-pattern-dynamic.md the decision lives in Execute
   (Matrix_specialise) and applies to every Execute.run_vectors; this module
   reports it and selects a float32 precision. *)

module Rw = Sarek_matrix_rewrite
module Execute = Sarek.Execute
module Spec = Sarek_execute.Matrix_specialise

(* The rewritten kernels call Int32.bits_of_float / float_of_bits (the bf16
   split) and the Gpu intrinsics: make sure their registrations are linked. *)
let () = Sarek_stdlib.force_init ()

type outcome =
  | Rewritten of {lowering : Rw.lowering; config : Sarek_coopmat.config}
  | Original of string

type mode = Spec.mode = Off | Exact | Auto

let set_mode = Spec.set_mode

let mode = Spec.mode

let run ?precision ~device ~ir ~args ~block ~grid () =
  match
    Execute.run_vectors_outcome
      ?precision
      ~explicit:true
      ~device
      ~ir
      ~args
      ~block
      ~grid
      ()
  with
  | Spec.Specialised s -> Rewritten {lowering = s.lowering; config = s.config}
  | Spec.Original why -> Original why

let outcome_text = function
  | Rewritten {lowering; config} ->
      Printf.sprintf
        "rewritten: %s, %s"
        (Rw.lowering_name lowering)
        (Sarek_coopmat.config_name config)
  | Original why -> "original kernel: " ^ why
