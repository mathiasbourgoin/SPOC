(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Interval arithmetic on int32 values (specs/f16-narrowing-barrier.md §2.1). A
    range contains every value an int32 expression can take; an operation that
    may wrap gives {!full}. *)

open Sarek_ir_types

type range = {lo : int; hi : int}

(** Every int32 value. *)
val full : range

val point : int -> range

val join_range : range -> range -> range

val binop_range : binop -> range -> range -> range

val unop_range : unop -> range -> range
