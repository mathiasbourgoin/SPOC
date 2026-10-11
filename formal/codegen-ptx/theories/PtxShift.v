(** PtxShift.v — right-shift semantics of the PTX backend (finding D-54).

 * WHAT THIS MODELS
 *
 * The production emitter (sarek/codegen/Sarek_ir_ptx_expr.ml, [emit_binop])
 * lowers the three IR right-shift operators as
 *
 *   Ir.Shr    -> shr.s32 / shr.s64   arithmetic (sign-filling), width by the
 *                                    first operand's register class
 *   Ir.Lshr32 -> shr.b32             logical (zero-filling), 32-bit
 *   Ir.Lshr64 -> shr.b64             logical (zero-filling), 64-bit
 *
 * An earlier version of this model treated [Shr] as a logical [Nat.shiftr] on
 * [U32] and had no [Lshr32]/[Lshr64]; it had been written against a former
 * [shr.u32] emission and contradicted production on every negative operand.
 *
 * VALUE DOMAIN
 *
 * Registers are bit patterns: a [w]-bit register holds an integer in
 * [0, 2^w). [to_signed] is the two's-complement reading of a pattern. The
 * arithmetic shift is "shift the signed reading right, flooring" — the
 * definition of OCaml's [asr] and of [shr.s] — and is re-expressed on
 * patterns afterwards. Operands outside [0, 2^w) are reduced modulo 2^w
 * first (the surrounding model uses unbounded [nat] and does not wrap
 * additions, so a stray out-of-range value must not make the shift partial).
 *
 * SHIFT AMOUNT
 *
 * PTX ISA, "shr": the shift amount operand is a [.u32] and "shift amounts
 * greater than the register width N are clamped to N". For the 64-bit forms
 * production narrows a 64-bit amount to 32 bits first ([cvt.u32.u64],
 * truncating), modelled by [to_unsigned 32]. Clamping to N, not N-1, is what
 * makes [shr.s] by >= N yield all sign bits and [shr.b] yield 0.
 *)

From Stdlib Require Import ZArith.
From Stdlib Require Import Lia.

Local Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** * Words *)
(* ------------------------------------------------------------------ *)

Definition word_modulus (w : nat) : Z := 2 ^ Z.of_nat w.

(** The bit pattern of [z] in a [w]-bit register: [z] modulo [2^w]. *)
Definition to_unsigned (w : nat) (z : Z) : Z := z mod word_modulus w.

(** Two's-complement reading of the [w]-bit pattern of [z]. *)
Definition to_signed (w : nat) (z : Z) : Z :=
  let u := to_unsigned w z in
  if u <? 2 ^ (Z.of_nat w - 1) then u else u - word_modulus w.

(** Effective shift count: the [.u32] operand clamped to the register width. *)
Definition shift_amount (w : nat) (k : Z) : Z :=
  Z.min (to_unsigned 32 k) (Z.of_nat w).

(** [shr.s<w>]: arithmetic shift right of the pattern [a] by [k]. *)
Definition ashr_w (w : nat) (a k : Z) : Z :=
  to_unsigned w (Z.shiftr (to_signed w a) (shift_amount w k)).

(** [shr.b<w>]: logical shift right of the pattern [a] by [k]. *)
Definition lshr_w (w : nat) (a k : Z) : Z :=
  Z.shiftr (to_unsigned w a) (shift_amount w k).

(* ------------------------------------------------------------------ *)
(** * The instruction table — what production must agree with *)
(* ------------------------------------------------------------------ *)

(** The three IR right-shift operators (mirrors the [Shr], [Lshr32] and
 *  [Lshr64] constructors of [Sarek_ir_types.binop]). *)
Inductive shr_op := OpShr | OpLshr32 | OpLshr64.

(** PTX instruction flavour: [shr.s] (signed) or [shr.b] (bitwise). *)
Inductive shr_flavour := FlavS | FlavB.

(** May [op] be applied to a first operand of this register class
 *  ([wide] = 64-bit)? [Shr] is polymorphic in width; [Lshr32]/[Lshr64] fix it
 *  (production would emit ill-typed PTX for a mismatch). *)
Definition shr_operand_ok (op : shr_op) (wide : bool) : bool :=
  match op, wide with
  | OpShr, _ => true
  | OpLshr32, false => true
  | OpLshr64, true => true
  | _, _ => false
  end.

(** Flavour production emits for each IR operator. *)
Definition shr_flavour_of (op : shr_op) : shr_flavour :=
  match op with
  | OpShr => FlavS
  | OpLshr32 => FlavB
  | OpLshr64 => FlavB
  end.

(** Register width in bits of the emitted instruction. *)
Definition shr_width_of (op : shr_op) (wide : bool) : nat :=
  match op with
  | OpShr => if wide then 64%nat else 32%nat
  | OpLshr32 => 32%nat
  | OpLshr64 => 64%nat
  end.

(** The instruction [shr.<flavour><width>] the model says production emits. *)
Definition shr_insn (op : shr_op) (wide : bool) : shr_flavour * nat :=
  (shr_flavour_of op, shr_width_of op wide).

(** Result pattern of the IR operator [op] applied to patterns [a], [k]. *)
Definition shr_eval (op : shr_op) (wide : bool) (a k : Z) : Z :=
  let w := shr_width_of op wide in
  match shr_flavour_of op with
  | FlavS => ashr_w w a k
  | FlavB => lshr_w w a k
  end.

(** [nat] bridge used by the [nat]-valued [ptx_val] domain. *)
Definition shr_eval_nat (op : shr_op) (wide : bool) (a k : nat) : nat :=
  Z.to_nat (shr_eval op wide (Z.of_nat a) (Z.of_nat k)).

(* ------------------------------------------------------------------ *)
(** * Facts about words *)
(* ------------------------------------------------------------------ *)

Lemma word_modulus_pos : forall w, 0 < word_modulus w.
Proof. intros w. unfold word_modulus. apply Z.pow_pos_nonneg; lia. Qed.

Lemma to_unsigned_range : forall w z, 0 <= to_unsigned w z < word_modulus w.
Proof.
  intros w z. unfold to_unsigned. apply Z.mod_pos_bound, word_modulus_pos.
Qed.

Lemma word_modulus_half : forall w, (0 < w)%nat ->
  word_modulus w = 2 * 2 ^ (Z.of_nat w - 1).
Proof.
  intros w Hw. unfold word_modulus.
  replace (Z.of_nat w) with (1 + (Z.of_nat w - 1)) at 1 by lia.
  rewrite Z.pow_add_r by lia. simpl. lia.
Qed.

Lemma half_pos : forall w, (0 < w)%nat -> 0 < 2 ^ (Z.of_nat w - 1).
Proof. intros w Hw. apply Z.pow_pos_nonneg; lia. Qed.

Lemma to_signed_range : forall w z, (0 < w)%nat ->
  - 2 ^ (Z.of_nat w - 1) <= to_signed w z < 2 ^ (Z.of_nat w - 1).
Proof.
  intros w z Hw. unfold to_signed.
  pose proof (to_unsigned_range w z) as [H0 H1].
  pose proof (word_modulus_half w Hw) as Hh.
  pose proof (half_pos w Hw) as Hp.
  destruct (to_unsigned w z <? 2 ^ (Z.of_nat w - 1)) eqn:E.
  - apply Z.ltb_lt in E. lia.
  - apply Z.ltb_ge in E. lia.
Qed.

Lemma to_unsigned_id : forall w z, 0 <= z < word_modulus w ->
  to_unsigned w z = z.
Proof. intros w z H. unfold to_unsigned. apply Z.mod_small, H. Qed.

(** [to_signed] inverts [to_unsigned] on the signed range. *)
Lemma to_signed_to_unsigned : forall w s, (0 < w)%nat ->
  - 2 ^ (Z.of_nat w - 1) <= s < 2 ^ (Z.of_nat w - 1) ->
  to_signed w (to_unsigned w s) = s.
Proof.
  intros w s Hw Hs.
  pose proof (word_modulus_half w Hw) as Hh.
  pose proof (half_pos w Hw) as Hp.
  unfold to_signed.
  rewrite (to_unsigned_id w (to_unsigned w s)) by apply to_unsigned_range.
  unfold to_unsigned.
  destruct (Z_lt_le_dec s 0) as [Hneg | Hnn].
  - assert (Hm : (s mod word_modulus w) = s + word_modulus w).
    { symmetry; apply Z.mod_unique_pos with (q := -1); lia. }
    rewrite Hm.
    destruct (s + word_modulus w <? 2 ^ (Z.of_nat w - 1)) eqn:E.
    + apply Z.ltb_lt in E. lia.
    + apply Z.ltb_ge in E. lia.
  - assert (Hm : (s mod word_modulus w) = s).
    { apply Z.mod_small. lia. }
    rewrite Hm.
    destruct (s <? 2 ^ (Z.of_nat w - 1)) eqn:E.
    + reflexivity.
    + apply Z.ltb_ge in E. lia.
Qed.

Lemma shift_amount_nonneg : forall w k, 0 <= shift_amount w k.
Proof.
  intros w k. unfold shift_amount.
  pose proof (to_unsigned_range 32 k).
  apply Z.min_glb; lia.
Qed.

(* ------------------------------------------------------------------ *)
(** * Theorems about the instructions *)
(* ------------------------------------------------------------------ *)

(** Results are [w]-bit patterns. *)
Theorem ashr_range : forall w a k, 0 <= ashr_w w a k < word_modulus w.
Proof. intros. unfold ashr_w. apply to_unsigned_range. Qed.

Theorem lshr_range : forall w a k,
  0 <= lshr_w w a k < word_modulus w.
Proof.
  intros w a k. unfold lshr_w.
  pose proof (to_unsigned_range w a) as [H0 H1].
  pose proof (shift_amount_nonneg w k) as Hk.
  rewrite Z.shiftr_div_pow2 by exact Hk.
  assert (Hp : 0 < 2 ^ shift_amount w k) by (apply Z.pow_pos_nonneg; lia).
  split.
  - apply Z.div_pos; lia.
  - apply Z.le_lt_trans with (m := to_unsigned w a); [|exact H1].
    apply Z.div_le_upper_bound; [exact Hp|].
    assert (1 <= 2 ^ shift_amount w k) by lia. nia.
Qed.

(** [shr.s] shifts the two's-complement reading right, flooring: the
 *  definition of an arithmetic shift, as OCaml's [asr] and [Int32.shift_right]
 *  compute it. *)
Theorem ashr_signed : forall w a k, (0 < w)%nat ->
  to_signed w (ashr_w w a k) = Z.shiftr (to_signed w a) (shift_amount w k).
Proof.
  intros w a k Hw. unfold ashr_w.
  pose proof (to_signed_range w a Hw) as [Hlo Hhi].
  pose proof (shift_amount_nonneg w k) as Hk.
  pose proof (half_pos w Hw) as Hm.
  apply to_signed_to_unsigned; [exact Hw|].
  rewrite Z.shiftr_div_pow2 by exact Hk.
  assert (Hp : 0 < 2 ^ shift_amount w k) by (apply Z.pow_pos_nonneg; lia).
  assert (H1 : 1 <= 2 ^ shift_amount w k) by lia.
  set (m := 2 ^ (Z.of_nat w - 1)) in *.
  set (d := 2 ^ shift_amount w k) in *.
  split.
  - apply Z.div_le_lower_bound; [exact Hp|]. nia.
  - apply Z.div_lt_upper_bound; [exact Hp|]. nia.
Qed.

(** On a non-negative operand (top bit clear) the arithmetic and the logical
 *  shift agree: the fill bit is 0 either way. *)
Theorem ashr_nonneg_is_logical : forall w a k, (0 < w)%nat ->
  to_unsigned w a < 2 ^ (Z.of_nat w - 1) -> ashr_w w a k = lshr_w w a k.
Proof.
  intros w a k Hw Hlt. unfold ashr_w, lshr_w, to_signed.
  apply Z.ltb_lt in Hlt. rewrite Hlt.
  pose proof (to_unsigned_range w a) as [H0 H1].
  pose proof (shift_amount_nonneg w k) as Hk.
  apply to_unsigned_id.
  rewrite Z.shiftr_div_pow2 by exact Hk.
  assert (Hp : 0 < 2 ^ shift_amount w k) by (apply Z.pow_pos_nonneg; lia).
  assert (H1' : 1 <= 2 ^ shift_amount w k) by lia.
  split.
  - apply Z.div_pos; lia.
  - apply Z.le_lt_trans with (m := to_unsigned w a); [|exact H1].
    apply Z.div_le_upper_bound; [exact Hp|]. nia.
Qed.

(** The sign survives an arithmetic shift: a negative operand (top bit set)
 *  gives a result with the top bit set. This is the property the old logical
 *  model violated. *)
Theorem ashr_negative_stays_negative : forall w a k, (0 < w)%nat ->
  2 ^ (Z.of_nat w - 1) <= to_unsigned w a ->
  2 ^ (Z.of_nat w - 1) <= ashr_w w a k.
Proof.
  intros w a k Hw Hge.
  pose proof (to_unsigned_range w a) as [H0 H1].
  pose proof (word_modulus_half w Hw) as Hh.
  assert (Hneg : to_signed w a < 0).
  { unfold to_signed.
    destruct (to_unsigned w a <? 2 ^ (Z.of_nat w - 1)) eqn:E.
    - apply Z.ltb_lt in E. lia.
    - apply Z.ltb_ge in E. lia. }
  pose proof (ashr_signed w a k Hw) as Hs.
  pose proof (shift_amount_nonneg w k) as Hk.
  assert (Hshneg : to_signed w (ashr_w w a k) < 0).
  { rewrite Hs. rewrite Z.shiftr_div_pow2 by exact Hk.
    assert (Hp : 0 < 2 ^ shift_amount w k) by (apply Z.pow_pos_nonneg; lia).
    apply Z.div_lt_upper_bound; [exact Hp|]. nia. }
  unfold to_signed in Hshneg.
  pose proof (ashr_range w a k) as [R0 R1].
  rewrite (to_unsigned_id w (ashr_w w a k)) in Hshneg by (split; assumption).
  destruct (ashr_w w a k <? 2 ^ (Z.of_nat w - 1)) eqn:E.
  - apply Z.ltb_lt in E. lia.
  - apply Z.ltb_ge in E. exact E.
Qed.

(** Arithmetic shift of all-ones (-1) is all-ones, whatever the count. *)
Theorem ashr_all_ones : forall w k, (0 < w)%nat ->
  ashr_w w (word_modulus w - 1) k = word_modulus w - 1.
Proof.
  intros w k Hw. unfold ashr_w.
  pose proof (word_modulus_half w Hw) as Hh.
  pose proof (half_pos w Hw) as Hp.
  assert (Hu : to_unsigned w (word_modulus w - 1) = word_modulus w - 1).
  { apply to_unsigned_id. pose proof (word_modulus_pos w). lia. }
  assert (Hs : to_signed w (word_modulus w - 1) = -1).
  { unfold to_signed. rewrite Hu.
    destruct (word_modulus w - 1 <? 2 ^ (Z.of_nat w - 1)) eqn:E.
    - apply Z.ltb_lt in E. lia.
    - lia. }
  rewrite Hs.
  pose proof (shift_amount_nonneg w k) as Hk.
  rewrite Z.shiftr_div_pow2 by exact Hk.
  assert (Hp2 : 0 < 2 ^ shift_amount w k) by (apply Z.pow_pos_nonneg; lia).
  assert (Hq : -1 / 2 ^ shift_amount w k = -1).
  { symmetry; apply Z.div_unique with (r := 2 ^ shift_amount w k - 1); [|lia]. left. lia. }
  rewrite Hq.
  unfold to_unsigned. symmetry. apply Z.mod_unique_pos with (q := -1); lia.
Qed.

(** PTX clamps the count to the width: a count of at least [w] is equivalent
 *  to a count of [w]. *)
Lemma shift_amount_clamped : forall w k,
  Z.of_nat w <= to_unsigned 32 k -> shift_amount w k = Z.of_nat w.
Proof. intros w k H. unfold shift_amount. apply Z.min_r. exact H. Qed.

(** A logical shift by at least the width clears the register. *)
Theorem lshr_big_amount : forall w a k,
  Z.of_nat w <= to_unsigned 32 k -> lshr_w w a k = 0.
Proof.
  intros w a k H. unfold lshr_w. rewrite (shift_amount_clamped w k H).
  pose proof (to_unsigned_range w a) as [H0 H1].
  rewrite Z.shiftr_div_pow2 by lia.
  apply Z.div_small. unfold word_modulus in H1. lia.
Qed.

(** An arithmetic shift by at least the width fills the register with the sign
 *  bit: 0 for a non-negative operand, all-ones for a negative one. *)
Theorem ashr_big_amount : forall w a k, (0 < w)%nat ->
  Z.of_nat w <= to_unsigned 32 k ->
  ashr_w w a k =
    if to_unsigned w a <? 2 ^ (Z.of_nat w - 1) then 0 else word_modulus w - 1.
Proof.
  intros w a k Hw H. unfold ashr_w. rewrite (shift_amount_clamped w k H).
  pose proof (word_modulus_half w Hw) as Hh.
  pose proof (half_pos w Hw) as Hp.
  pose proof (to_unsigned_range w a) as [R0 R1].
  rewrite Z.shiftr_div_pow2 by lia.
  unfold to_signed.
  destruct (to_unsigned w a <? 2 ^ (Z.of_nat w - 1)) eqn:E.
  - apply Z.ltb_lt in E.
    rewrite (Z.div_small (to_unsigned w a)) by (unfold word_modulus in *; lia).
    reflexivity.
  - apply Z.ltb_ge in E.
    assert (Hq : (to_unsigned w a - word_modulus w) / 2 ^ Z.of_nat w = -1).
    { unfold word_modulus in *.
      symmetry; apply Z.div_unique with (r := to_unsigned w a); [|lia]. left. lia. }
    rewrite Hq.
    unfold to_unsigned. symmetry. apply Z.mod_unique_pos with (q := -1); lia.
Qed.

(* ------------------------------------------------------------------ *)
(** * The old model was wrong: concrete regression pins *)
(* ------------------------------------------------------------------ *)

(** -4 >> 1 on a 32-bit register: arithmetic gives -2 (0xFFFFFFFE), logical
 *  gives 0x7FFFFFFE. The pre-D-54 model computed the second for [Shr]. *)
Example ashr32_minus4_by1 : ashr_w 32 4294967292 1 = 4294967294.
Proof. vm_compute. reflexivity. Qed.

Example lshr32_minus4_by1 : lshr_w 32 4294967292 1 = 2147483646.
Proof. vm_compute. reflexivity. Qed.

Theorem ashr_differs_from_lshr :
  ashr_w 32 4294967292 1 <> lshr_w 32 4294967292 1.
Proof. vm_compute. discriminate. Qed.

(** The same on 64 bits: -4 >> 1. *)
Example ashr64_minus4_by1 :
  ashr_w 64 18446744073709551612 1 = 18446744073709551614.
Proof. vm_compute. reflexivity. Qed.

(** PTX count clamping: 0x80000000 >>s 32 is all-ones, >>b 32 is zero. *)
Example ashr32_by_width : ashr_w 32 2147483648 32 = 4294967295.
Proof. vm_compute. reflexivity. Qed.

Example lshr32_by_width : lshr_w 32 2147483648 32 = 0.
Proof. vm_compute. reflexivity. Qed.

(** The instruction table, pinned against production's [emit_binop]. *)
Example shr_insn_shr32 : shr_insn OpShr false = (FlavS, 32%nat).
Proof. reflexivity. Qed.
Example shr_insn_shr64 : shr_insn OpShr true = (FlavS, 64%nat).
Proof. reflexivity. Qed.
Example shr_insn_lshr32 : shr_insn OpLshr32 false = (FlavB, 32%nat).
Proof. reflexivity. Qed.
Example shr_insn_lshr64 : shr_insn OpLshr64 true = (FlavB, 64%nat).
Proof. reflexivity. Qed.
