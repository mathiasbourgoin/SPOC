(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Evaluate a [for] upper bound once, before the first iteration.

    Sarek [for i = a to b] has OCaml semantics: [b] is evaluated once. The
    C-family, GLSL and WGSL loops are spelled [for (i = a; i <= b; i++)], whose
    condition re-reads [b] every iteration, so a bound that reads something the
    body changes (a [mut] variable, an array element) iterates a different
    number of times than the interpreter, Native and PTX.

    {!hoist} rewrites every [SFor (v, lo, hi, dir, body)] whose [hi] is not
    {e stable} into
    [SLet (sarek_for_hi_N, hi, SFor (v, lo, sarek_for_hi_N, dir, body))] (the
    [sarek_] prefix is reserved, so the name cannot collide with user code). A
    bound is stable, and left untouched, when nothing in the loop can change it:
    literals, variables the loop body never assigns, thread and block indices,
    array lengths, and pure arithmetic or casts of those. Common loops such as
    [for i = 0 to n - 1l] therefore keep their generated source.

    {b Limit.} A bound of [Int32.max_int] ([Upto]) or [Int32.min_int] ([Downto])
    is not supported on any backend: the loop variable would have to step past
    the bound, which overflows [int32] (undefined behaviour in the C dialects,
    wrap-around and an endless loop in PTX and in the interpreter, whose
    [Int32.add] wraps). The loop start is hoisted too, ahead of the bound, when
    it is not pure, so the order start-then-bound is kept. *)

(** Applied to the kernel body and every helper body. Not applied to PTX, which
    evaluates the bound once already. *)
val hoist : Sarek_ir_types.kernel -> Sarek_ir_types.kernel

(** The same rewrite on one statement (exposed for tests). Fresh names are
    numbered from 1 per call. *)
val hoist_stmt : Sarek_ir_types.stmt -> Sarek_ir_types.stmt

(** [stable ~assigned e] is [true] when [e] cannot change while a loop whose
    body assigns the variable names [assigned] runs (exposed for tests). *)
val stable : assigned:string list -> Sarek_ir_types.expr -> bool
