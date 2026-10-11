(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Which f32 -> f16 narrowings need [sarek_f32_barrier]
    (specs/f16-narrowing-barrier.md).

    The DSL narrows in two steps: every f32 operation rounds to binary32, then
    the narrowing rounds that to binary16. The AMDGPU backend may fuse the
    operation feeding a narrowing into it ([v_fma_mixlo_f16], [v_add_f16], ...)
    and round once; the barrier forbids that. It is unnecessary when one
    rounding provably equals two:

    - {b exact}: no operation in the narrowed expression rounds, so the f32
      value [v] is the exact real value of the expression and any evaluation the
      compiler may choose ends in [round16(v)];
    - {b binary16 sum}: the expression is [a + b] or [a - b] of two binary16
      values; binary32 carries 24 >= 2 * 11 + 1 significand bits, so
      [round16(round32(a +/- b)) = round16(a +/- b)].

    The analysis is conservative: anything it cannot prove keeps the barrier. *)

open Sarek_ir_types

(** What is known about the variables and arrays of one function body. *)
type facts

(** Facts for a kernel body: its parameters, locals, shared arrays and every
    binding in its body. *)
val of_kernel : kernel -> facts

(** Facts for a helper function body. Its parameters are not opaque (the helper
    is inlined, so the caller's arithmetic reaches them) and its array reads are
    never treated as opaque leaves. *)
val of_helper : helper_func -> facts

(** No facts: every variable unknown. *)
val empty : facts

(** How to emit the narrowing of the f32 expression [e]:
    - [Barrier]: through [sarek_f32_barrier] (nothing proved);
    - [Plain]: a bare narrowing; one rounding equals two whatever the compiler
      fuses into it;
    - [Signed_product (a, b)]: [e] is [a * b], exact in f32, but possibly a zero
      whose sign the fused form gets wrong ([a * b + 0] turns [-0] into [+0]).
      Emitted as a bare narrowing of [a * b] whose sign bit is then or-ed with
      [sign a xor sign b]: the sign of the exact product, which the fused and
      the unfused result share except on that zero. *)
type form = Barrier | Plain | Signed_product of expr * expr

val form : facts -> expr -> form

(** [needs_barrier facts e] is [form facts e = Barrier]. *)
val needs_barrier : facts -> expr -> bool

(** The bound the analysis proves for an exact f32 value: the value is [k * 2^q]
    with [k] an integer, [|k| <= kmax], [qlo <= q <= qhi]; [binary16] when it is
    moreover a binary16 value, [nonzero] when it is never a zero, [nonneg] when
    it is never negative and never [-0]. Exposed for the property test that
    checks the analysis against a reference evaluation. *)
type bound = {
  kmax : float;
  qlo : int;
  qhi : int;
  binary16 : bool;
  nonzero : bool;
  nonneg : bool;
}

(** [exact facts e]: [Some b] when every operation in [e] is exact in f32 (so
    [e]'s f32 value is its exact real value) and that value satisfies [b];
    [None] when this cannot be shown. *)
val exact : facts -> expr -> bound option
