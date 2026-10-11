(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

let register () = ()

let source_of (_ : Sarek_ir_types.kernel) = ""

let ptx_of ~name:(_ : string) (_ : string) : string option = None
