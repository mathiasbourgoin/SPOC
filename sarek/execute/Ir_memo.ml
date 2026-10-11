(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Ir_memo - the launch path's tables keyed by an IR's physical identity
 ******************************************************************************)

open Sarek_ir_types

let hashes = Atomic.make 0

(** Every hash of an IR by the launch-path tables; a test hook. *)
let hash_count () = Atomic.get hashes

(** Physical identity of an IR, for [Ephemeron.K1.Make]. *)
module Physical = struct
  type t = kernel

  let equal = ( == )

  (* Any function of the contents is consistent with [==]. [Hashtbl.hash]
     reads a bounded prefix, so this is cheap; spoc_llm names every kernel
     "sarek_kern", so the parameter list is what spreads them. *)
  let hash (k : t) =
    Atomic.incr hashes ;
    Hashtbl.hash (k.kern_name, k.kern_params)
end
