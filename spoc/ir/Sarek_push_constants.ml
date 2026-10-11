(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Sarek_ir_types

type width = W4 | W8

let bytes_of_width = function W4 -> 4 | W8 -> 8

let width_of_elttype = function TInt64 | TFloat64 -> W8 | _ -> W4

let vulkan_block_limit = 128

let place fields =
  let step (rev_offsets, offset) field =
    let size = bytes_of_width field in
    let misalignment = offset mod size in
    let at =
      if misalignment = 0 then offset else offset + (size - misalignment)
    in
    (at :: rev_offsets, at + size)
  in
  let rev_offsets, total = List.fold_left step ([], 0) fields in
  (List.rev rev_offsets, total)

let fields_of_params params =
  let vectors, scalars =
    List.fold_left
      (fun (vectors, scalars) decl ->
        match decl with
        | DParam ({var_type = TVec _; _}, _) -> (W4 :: vectors, scalars)
        | DParam (v, _) -> (vectors, width_of_elttype v.var_type :: scalars)
        | DLocal _ | DShared _ -> (vectors, scalars))
      ([], [])
      params
  in
  List.rev vectors @ List.rev scalars

let block_bytes params = snd (place (fields_of_params params))

(* Vector lengths come first and the position pairs last, so fusing is not the
   concatenation of the members' own blocks. *)
let fused_block_bytes members =
  let positions = List.concat_map (fun _ -> [W4; W4]) members in
  let fields = fields_of_params (List.concat members) in
  snd (place (fields @ positions))
