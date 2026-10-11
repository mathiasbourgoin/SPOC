(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

let register () = ()

let source_of (_ : Sarek_ir_types.kernel) = ""

let is_zluda ~device_name:(_ : string) ~compute_capability:(_ : int * int) =
  false

let probe ~device_name:(_ : string) ~compute_capability:(_ : int * int)
    ~warp_size:(_ : int) : Sarek_coopmat.device_support option =
  None

let ptx_of ~name:(_ : string) (_ : string) : string option = None
