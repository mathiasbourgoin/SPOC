(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Single rounding at f32 -> f16 narrowings: an explicit opt-in
    (specs/f16-narrowing-barrier.md §4).

    The DSL narrows in two steps (every f32 operation rounds to binary32, the
    narrowing rounds that to binary16) and the interpreter does the same. The
    code generator keeps that on every device by emitting a barrier
    ([sarek_f32_barrier]) wherever it cannot prove the two steps give the same
    value as one. The barrier costs: on gfx1100 it stops the compiler fusing the
    producing multiply into the narrowing ([v_fma_mixlo_f16]) and merging the
    narrowed stores.

    After [accept_single_rounding scope ~reason], kernels in [scope] are
    generated without any narrowing barrier. The device may then round a
    narrowed expression once, from its exact value, where the interpreter rounds
    twice. What changes, per narrowing of [v]:
    - [v = a * b] with the narrowing fused: the result is [round16(a * b)]
      instead of [round16(round32(a * b))]; they differ by at most one binary16
      ulp, and only near a binary16 rounding tie; a [-0] product may come out
      [+0] (the fused form adds [+0]);
    - [v = a * b + c] fused into one [v_fma_mix]: the result is
      [round16(a * b + c)] instead of [round16(round32(round32(a * b) + c))];
      the difference is at most one binary16 ulp plus [|a * b| * 2^-24] (the
      product's binary32 rounding error), which is relatively large when
      [a * b + c] cancels. The device result is the more accurate one in both
      cases. The interpreter keeps rounding twice: which fusions a compiler
      performs is not specified, so it cannot be modelled exactly. A
      differential test of such a kernel must compare with this bound, not bit
      for bit.

    Process-wide and one-way, like {!Sarek_coopmat.accept_relaxed_accumulation}
    and separate from it: accepting one never accepts the other. The decision is
    taken when a kernel's source is generated; a kernel generated (and memoised)
    before the call keeps its barriers. *)

(** Which kernels the opt-in covers. [Kernel name] matches
    {!Sarek_ir_types.kernel.kern_name} exactly. A kernel written with
    [[%kernel fun ...]] and no name is named ["sarek_kern"], like every other
    such kernel, and that scope is refused; a kernel built by a fusion pass has
    the fused name, not the names of its parts, so it is not covered by a
    [Kernel] scope. *)
type scope = All_kernels | Kernel of string

(** Why an opt-in was refused. [Default_kernel_name]: [Kernel "sarek_kern"], the
    name every unnamed [[%kernel]] gets, would cover all of them. *)
type refusal = Blank_reason | Default_kernel_name

val refusal_message : refusal -> string

(** [accept_single_rounding scope ~reason]. The first non-empty [reason] for a
    scope is kept; later calls for the same scope are no-ops. Thread-safe. A
    blank [reason] and [Kernel "sarek_kern"] are refused and record nothing. *)
val accept_single_rounding : scope -> reason:string -> (unit, refusal) result

(** The reason recorded for the kernel named [kernel]: its own [Kernel] scope
    first, then [All_kernels]; [None] when neither was accepted. *)
val single_rounding_reason : kernel:string -> string option
