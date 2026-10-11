(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Host-side evaluation of uniform int32 expressions
    (specs/matrix-pattern-rewrite.md).

    A launch-time decision (the region a kernel covers, a reduction length)
    needs the value of IR expressions built from scalar parameters, constants
    and the launch's block and grid dimensions. Arithmetic is int32 and wraps,
    as on the device. *)

open Sarek_ir_types

(** The facts of one launch: dimensions as [(x, y, z)], and the value of a
    scalar parameter. *)
type launch = {
  grid : int * int * int;
  block : int * int * int;
  scalar : var -> int32 option;
}

(** [None] when the expression is not uniform over the launch (it reads memory,
    a thread index, a loop variable), contains an operator with no int32 meaning
    here, or divides by zero. *)
val eval : launch -> expr -> int32 option

(** Whether an expression reads [block_dim_*], [grid_dim_*] (or a [global_size*]
    / [num_groups*] intrinsic): such a value changes meaning in a kernel
    launched with other dimensions. *)
val mentions_launch_dims : expr -> bool
