(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

let init () =
  if not (Sarek_hip.Hip_shared.is_disabled ()) then Sarek_hip.Hip_plugin.init ()
