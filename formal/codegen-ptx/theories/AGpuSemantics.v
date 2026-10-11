(** AGpuSemantics.v — Abstract GPU semantics for Sarek PTX code generation.
 *
 * Concrete, Sarek-targeted model of PTX execution semantics.
 *
 * This module defines the AGPU state machine and the evaluation function
 * [agpu_eval_ir] for the subset of Sarek IR covered by the PTX code generator.
 *
 * Design notes:
 * - [float] is Rocq's primitive IEEE-754 float (from Stdlib.Floats).
 * - [nat] is used for both U32 and U64 values (no overflow modelling needed
 *   for correctness of code-generation — the PTX emitter does not rely on
 *   wrap-around arithmetic at the specification level).
 * - Unsupported / complex constructs return [None]; no [Admitted] is used.
 * - Math intrinsics [sin_f32], [cos_f32], [fma_f32] etc. are declared as
 *   [Parameter]s (uninterpreted constants, not Admitted goals).
 * - Barrier ([SBarrier]) is modelled as an identity on the sequential state;
 *   full barrier semantics requires a concurrent execution model (future work).
 *)

From Stdlib Require Import Strings.String.
From Stdlib Require Import Arith.Arith.
From Stdlib Require Import Bool.Bool.
From Stdlib Require Import ZArith.
From Stdlib Require Import Lia.
From CodegenPtx Require Import PtxShift.
From Stdlib Require Import Floats.

Open Scope string_scope.
(* ZArith (imported for PtxShift's Z-level lemmas) opens Z_scope; the rest of
   this file is nat arithmetic. *)
Open Scope nat_scope.

(* ------------------------------------------------------------------ *)
(** * PTX value domain *)
(* ------------------------------------------------------------------ *)

Inductive ptx_val :=
  | U32  : nat  -> ptx_val
  | U64  : nat  -> ptx_val
  | F32  : float -> ptx_val
  | F64  : float -> ptx_val
  | Pred : bool  -> ptx_val.

(* ------------------------------------------------------------------ *)
(** * Memory spaces *)
(* ------------------------------------------------------------------ *)

Inductive mem_space := Global | Shared.

(* ------------------------------------------------------------------ *)
(** * Thread constants (fixed for the lifetime of a thread) *)
(* ------------------------------------------------------------------ *)

Record thread_const := {
  tidx : nat;   (** threadIdx.x *)
  bidx : nat;   (** blockIdx.x  *)
  bdim : nat;   (** blockDim.x  *)
}.

(* ------------------------------------------------------------------ *)
(** * Memory model *)
(* ------------------------------------------------------------------ *)

Record agpu_mem := {
  global_mem : nat -> ptx_val;   (** word-addressed global memory *)
  shared_mem : nat -> ptx_val;   (** word-addressed shared memory  *)
}.

(* ------------------------------------------------------------------ *)
(** * Abstract GPU thread state *)
(* ------------------------------------------------------------------ *)

Record agpu_state := {
  regs : string -> option ptx_val;   (** register file             *)
  tc   : thread_const;               (** thread constants          *)
  mem  : agpu_mem;                   (** memory hierarchy          *)
}.

(* ------------------------------------------------------------------ *)
(** * Uninterpreted math intrinsics
 *
 * [sqrt] and [abs] are available in Stdlib.Floats.
 * [sin], [cos], [fma] are not in Stdlib — we declare them as Parameters
 * (uninterpreted constants).  These are axiomatic, not admitted goals,
 * and therefore do NOT violate the zero-admits invariant.
 *)
(* ------------------------------------------------------------------ *)

Parameter sin_f32 : float -> float.
Parameter cos_f32 : float -> float.
Parameter fma_f32 : float -> float -> float -> float.
Parameter sin_f64 : float -> float.
Parameter cos_f64 : float -> float.
Parameter fma_f64 : float -> float -> float -> float.

(* ------------------------------------------------------------------ *)
(** * Rocq mirror of the covered Sarek IR [expr] subset
 *
 * This type mirrors [Sarek_ir_types.ml:type expr] for the constructs
 * that [emit_expr] in [Sarek_ir_ptx_expr.ml] handles (non-stub cases).
 * Unsupported constructors are not included; [agpu_eval_ir] returns [None]
 * for any expression it cannot evaluate.
 *)
(* ------------------------------------------------------------------ *)

(** Mirrors [Sarek_ir_types.const] *)
Inductive ir_const :=
  | CInt32  : nat   -> ir_const   (** int32 literal (nat proxy)  *)
  | CInt64  : nat   -> ir_const   (** int64 literal (nat proxy)  *)
  | CFloat32 : float -> ir_const
  | CFloat64 : float -> ir_const
  | CBool   : bool  -> ir_const
  | CUnit   : ir_const.

(** Mirrors [Sarek_ir_types.binop] — all 20 operators ([Shr] is the
 *  arithmetic shift right, [Lshr32]/[Lshr64] the logical ones; see
 *  PtxShift.v) *)
Inductive ir_binop :=
  | Add | Sub | Mul | Div | Mod
  | Eq  | Ne  | Lt  | Le  | Gt | Ge
  | And | Or
  | Shl | Shr | Lshr32 | Lshr64
  | BitAnd | BitOr | BitXor.

(** Mirrors [Sarek_ir_types.elttype] — the subset relevant to AGPU *)
Inductive ir_elttype :=
  | TInt32 | TInt64 | TFloat32 | TFloat64 | TBool.

(** Mirrors [Sarek_ir_types.memspace] for array reads *)
Inductive ir_memspace := MS_Global | MS_Shared.

(** Covered subset of [Sarek_ir_types.expr] *)
Inductive ir_expr :=
  (** Literals *)
  | IEConst     : ir_const  -> ir_expr
  (** Variable read — register name *)
  | IEVar       : string    -> ir_expr
  (** Binary operation *)
  | IEBinop     : ir_binop  -> ir_expr -> ir_expr -> ir_expr
  (** Array read: memory space, address expression, index expression *)
  | IEArrayRead : ir_memspace -> ir_expr -> ir_expr -> ir_expr
  (** Thread-block intrinsics *)
  | IEThreadIdxX  : ir_expr
  | IEBlockIdxX   : ir_expr
  | IEBlockDimX   : ir_expr
  (** Derived: global thread id = blockIdx.x * blockDim.x + threadIdx.x *)
  | IEGlobalIdx   : ir_expr
  (** Barrier: treated as a unit-valued expression (identity on state) *)
  | IEBarrier     : ir_expr
  (** Math intrinsics — f32 variants *)
  | IESin32  : ir_expr -> ir_expr
  | IECos32  : ir_expr -> ir_expr
  | IESqrt32 : ir_expr -> ir_expr
  | IEFabs32 : ir_expr -> ir_expr
  | IEFma32  : ir_expr -> ir_expr -> ir_expr -> ir_expr
  (** Math intrinsics — f64 variants *)
  | IESin64  : ir_expr -> ir_expr
  | IECos64  : ir_expr -> ir_expr
  | IESqrt64 : ir_expr -> ir_expr
  | IEFabs64 : ir_expr -> ir_expr
  | IEFma64  : ir_expr -> ir_expr -> ir_expr -> ir_expr.

(* ------------------------------------------------------------------ *)
(** * Helper: nat arithmetic used in binary-op evaluation *)
(* ------------------------------------------------------------------ *)

(** Boolean-to-nat: true -> 1, false -> 0 *)
Definition bool_to_nat (b : bool) : nat :=
  if b then 1 else 0.

(** Nat comparison result as a U32 *)
Definition nat_cmp_to_u32 (b : bool) : ptx_val := U32 (bool_to_nat b).

(* ------------------------------------------------------------------ *)
(** * Binary operation semantics
 *
 * Returns [None] when the operand types are inconsistent (e.g. adding
 * a float to an int).  The OCaml emitter infers types from register-name
 * prefixes; here we do it structurally.
 *)
(* ------------------------------------------------------------------ *)

(** Right shifts. The operator fixes the instruction ([PtxShift.shr_insn]); the
 *  first operand's register class fixes the width of [Shr]. A 64-bit first
 *  operand accepts a 32- or 64-bit amount (production narrows the latter); a
 *  32-bit one needs a 32-bit amount. [Lshr32]/[Lshr64] on the wrong register
 *  class are ill-typed PTX and evaluate to [None]. *)
Definition agpu_eval_shr (op : shr_op) (v1 v2 : ptx_val) : option ptx_val :=
  match v1, v2 with
  | U32 a, U32 b =>
      if shr_operand_ok op false
      then Some (U32 (shr_eval_nat op false a b)) else None
  | U64 a, U64 b | U64 a, U32 b =>
      if shr_operand_ok op true
      then Some (U64 (shr_eval_nat op true a b)) else None
  | _, _ => None
  end.

Definition agpu_eval_binop (op : ir_binop) (v1 v2 : ptx_val)
    : option ptx_val :=
  match op, v1, v2 with
  (* ---- U32 arithmetic ---- *)
  | Add,    U32 a, U32 b => Some (U32 (a + b))
  | Sub,    U32 a, U32 b => Some (U32 (a - b))
  | Mul,    U32 a, U32 b => Some (U32 (a * b))
  | Div,    U32 a, U32 b => Some (U32 (Nat.div a b))
  | Mod,    U32 a, U32 b => Some (U32 (Nat.modulo a b))
  (* ---- U32 comparisons → U32(0/1) ---- *)
  | Eq,     U32 a, U32 b => Some (nat_cmp_to_u32 (Nat.eqb a b))
  | Ne,     U32 a, U32 b => Some (nat_cmp_to_u32 (negb (Nat.eqb a b)))
  | Lt,     U32 a, U32 b => Some (nat_cmp_to_u32 (Nat.ltb a b))
  | Le,     U32 a, U32 b => Some (nat_cmp_to_u32 (Nat.leb a b))
  | Gt,     U32 a, U32 b => Some (nat_cmp_to_u32 (Nat.ltb b a))
  | Ge,     U32 a, U32 b => Some (nat_cmp_to_u32 (Nat.leb b a))
  (* ---- U32 bitwise / shift ---- *)
  | And,    U32 a, U32 b => Some (U32 (Nat.land a b))
  | Or,     U32 a, U32 b => Some (U32 (Nat.lor  a b))
  | BitAnd, U32 a, U32 b => Some (U32 (Nat.land a b))
  | BitOr,  U32 a, U32 b => Some (U32 (Nat.lor  a b))
  | BitXor, U32 a, U32 b => Some (U32 (Nat.lxor a b))
  | Shl,    U32 a, U32 b => Some (U32 (Nat.shiftl a b))
  (* arithmetic / logical right shifts: shr.s32/.s64 and shr.b32/.b64 *)
  | Shr,    _, _ => agpu_eval_shr OpShr v1 v2
  | Lshr32, _, _ => agpu_eval_shr OpLshr32 v1 v2
  | Lshr64, _, _ => agpu_eval_shr OpLshr64 v1 v2
  (* ---- U64 arithmetic ---- *)
  | Add, U64 a, U64 b => Some (U64 (a + b))
  | Sub, U64 a, U64 b => Some (U64 (a - b))
  | Mul, U64 a, U64 b => Some (U64 (a * b))
  | Div, U64 a, U64 b => Some (U64 (Nat.div a b))
  | Mod, U64 a, U64 b => Some (U64 (Nat.modulo a b))
  (* ---- U64 comparisons ---- *)
  | Eq, U64 a, U64 b => Some (nat_cmp_to_u32 (Nat.eqb a b))
  | Ne, U64 a, U64 b => Some (nat_cmp_to_u32 (negb (Nat.eqb a b)))
  | Lt, U64 a, U64 b => Some (nat_cmp_to_u32 (Nat.ltb a b))
  | Le, U64 a, U64 b => Some (nat_cmp_to_u32 (Nat.leb a b))
  | Gt, U64 a, U64 b => Some (nat_cmp_to_u32 (Nat.ltb b a))
  | Ge, U64 a, U64 b => Some (nat_cmp_to_u32 (Nat.leb b a))
  (* ---- F32 arithmetic ---- *)
  | Add, F32 a, F32 b => Some (F32 (add a b))
  | Sub, F32 a, F32 b => Some (F32 (sub a b))
  | Mul, F32 a, F32 b => Some (F32 (mul a b))
  | Div, F32 a, F32 b => Some (F32 (div a b))
  (* ---- F32 comparisons ---- *)
  | Eq, F32 a, F32 b => Some (nat_cmp_to_u32 (eqb a b))
  | Ne, F32 a, F32 b => Some (nat_cmp_to_u32 (negb (eqb a b)))
  | Lt, F32 a, F32 b => Some (nat_cmp_to_u32 (ltb a b))
  | Le, F32 a, F32 b => Some (nat_cmp_to_u32 (leb a b))
  | Gt, F32 a, F32 b => Some (nat_cmp_to_u32 (ltb b a))
  | Ge, F32 a, F32 b => Some (nat_cmp_to_u32 (leb b a))
  (* ---- F64 arithmetic ---- *)
  | Add, F64 a, F64 b => Some (F64 (add a b))
  | Sub, F64 a, F64 b => Some (F64 (sub a b))
  | Mul, F64 a, F64 b => Some (F64 (mul a b))
  | Div, F64 a, F64 b => Some (F64 (div a b))
  (* ---- F64 comparisons ---- *)
  | Eq, F64 a, F64 b => Some (nat_cmp_to_u32 (eqb a b))
  | Ne, F64 a, F64 b => Some (nat_cmp_to_u32 (negb (eqb a b)))
  | Lt, F64 a, F64 b => Some (nat_cmp_to_u32 (ltb a b))
  | Le, F64 a, F64 b => Some (nat_cmp_to_u32 (leb a b))
  | Gt, F64 a, F64 b => Some (nat_cmp_to_u32 (ltb b a))
  | Ge, F64 a, F64 b => Some (nat_cmp_to_u32 (leb b a))
  (* ---- Unsupported type combinations ---- *)
  | _, _, _ => None
  end.

(* ------------------------------------------------------------------ *)
(** * [agpu_eval_ir] — main evaluation function
 *
 * Signature: [agpu_state -> ir_expr -> option (ptx_val * agpu_state)]
 *
 * State is threaded through (for future stateful extensions; currently
 * only memory reads can alter the observed value).  The state itself is
 * not mutated by expression evaluation (expressions are pure in the IR).
 *)
(* ------------------------------------------------------------------ *)

Fixpoint agpu_eval_ir (st : agpu_state) (e : ir_expr)
    : option (ptx_val * agpu_state) :=
  match e with

  (* ---- Literals ---- *)
  | IEConst (CInt32  n) => Some (U32 n, st)
  | IEConst (CInt64  n) => Some (U64 n, st)
  | IEConst (CFloat32 f) => Some (F32 f, st)
  | IEConst (CFloat64 f) => Some (F64 f, st)
  | IEConst (CBool b)   => Some (U32 (bool_to_nat b), st)
  | IEConst CUnit        => Some (U32 0, st)

  (* ---- Variable read from register file ---- *)
  | IEVar name =>
      match (st.(regs)) name with
      | Some v => Some (v, st)
      | None   => None
      end

  (* ---- Binary operations ---- *)
  | IEBinop op e1 e2 =>
      match agpu_eval_ir st e1 with
      | None          => None
      | Some (v1, st1) =>
          match agpu_eval_ir st1 e2 with
          | None          => None
          | Some (v2, st2) =>
              match agpu_eval_binop op v1 v2 with
              | None   => None
              | Some v => Some (v, st2)
              end
          end
      end

  (* ---- Global memory read: addr + idx -> global_mem[addr + idx]
   *
   * Both base and index must be the same integer type (U32+U32 or U64+U64).
   * Mixed-type combinations (U32 base + U64 idx) return None because the PTX
   * emitter generates instructions with uniform register types.
   *)
  | IEArrayRead MS_Global base_e idx_e =>
      match agpu_eval_ir st base_e with
      | Some (U32 base, st1) =>
          match agpu_eval_ir st1 idx_e with
          | Some (U32 idx, st2) =>
              Some (st2.(mem).(global_mem) (base + idx), st2)
          | _ => None
          end
      | Some (U64 base, st1) =>
          match agpu_eval_ir st1 idx_e with
          | Some (U64 idx, st2) =>
              Some (st2.(mem).(global_mem) (base + idx), st2)
          | _ => None
          end
      | _ => None
      end

  (* ---- Shared memory read ---- *)
  | IEArrayRead MS_Shared base_e idx_e =>
      match agpu_eval_ir st base_e with
      | Some (U32 base, st1) =>
          match agpu_eval_ir st1 idx_e with
          | Some (U32 idx, st2) =>
              Some (st2.(mem).(shared_mem) (base + idx), st2)
          | _ => None
          end
      | Some (U64 base, st1) =>
          match agpu_eval_ir st1 idx_e with
          | Some (U64 idx, st2) =>
              Some (st2.(mem).(shared_mem) (base + idx), st2)
          | _ => None
          end
      | _ => None
      end

  (* ---- Thread-block intrinsics ---- *)
  | IEThreadIdxX => Some (U32 st.(tc).(tidx), st)
  | IEBlockIdxX  => Some (U32 st.(tc).(bidx), st)
  | IEBlockDimX  => Some (U32 st.(tc).(bdim), st)

  (* ---- global_idx = blockIdx.x * blockDim.x + threadIdx.x ---- *)
  | IEGlobalIdx =>
      let gid := st.(tc).(bidx) * st.(tc).(bdim) + st.(tc).(tidx) in
      Some (U32 gid, st)

  (* ---- Barrier: identity on state (sequential model) ---- *)
  | IEBarrier => Some (U32 0, st)

  (* ---- Math intrinsics — f32 ---- *)
  | IESin32 e1 =>
      match agpu_eval_ir st e1 with
      | Some (F32 x, st1) => Some (F32 (sin_f32 x), st1)
      | _ => None
      end

  | IECos32 e1 =>
      match agpu_eval_ir st e1 with
      | Some (F32 x, st1) => Some (F32 (cos_f32 x), st1)
      | _ => None
      end

  | IESqrt32 e1 =>
      match agpu_eval_ir st e1 with
      | Some (F32 x, st1) => Some (F32 (sqrt x), st1)
      | _ => None
      end

  | IEFabs32 e1 =>
      match agpu_eval_ir st e1 with
      | Some (F32 x, st1) => Some (F32 (abs x), st1)
      | _ => None
      end

  | IEFma32 ea eb ec =>
      match agpu_eval_ir st ea with
      | Some (F32 a, st1) =>
          match agpu_eval_ir st1 eb with
          | Some (F32 b, st2) =>
              match agpu_eval_ir st2 ec with
              | Some (F32 c, st3) => Some (F32 (fma_f32 a b c), st3)
              | _ => None
              end
          | _ => None
          end
      | _ => None
      end

  (* ---- Math intrinsics — f64 ---- *)
  | IESin64 e1 =>
      match agpu_eval_ir st e1 with
      | Some (F64 x, st1) => Some (F64 (sin_f64 x), st1)
      | _ => None
      end

  | IECos64 e1 =>
      match agpu_eval_ir st e1 with
      | Some (F64 x, st1) => Some (F64 (cos_f64 x), st1)
      | _ => None
      end

  | IESqrt64 e1 =>
      match agpu_eval_ir st e1 with
      | Some (F64 x, st1) => Some (F64 (sqrt x), st1)
      | _ => None
      end

  | IEFabs64 e1 =>
      match agpu_eval_ir st e1 with
      | Some (F64 x, st1) => Some (F64 (abs x), st1)
      | _ => None
      end

  | IEFma64 ea eb ec =>
      match agpu_eval_ir st ea with
      | Some (F64 a, st1) =>
          match agpu_eval_ir st1 eb with
          | Some (F64 b, st2) =>
              match agpu_eval_ir st2 ec with
              | Some (F64 c, st3) => Some (F64 (fma_f64 a b c), st3)
              | _ => None
              end
          | _ => None
          end
      | _ => None
      end

  end.

(* ------------------------------------------------------------------ *)
(** * Sanity lemmas (no admits — proved by [reflexivity])             *)
(* ------------------------------------------------------------------ *)

(** A constant U32 evaluates to U32 in any state. *)
Lemma agpu_eval_const_u32 :
  forall n st, agpu_eval_ir st (IEConst (CInt32 n)) = Some (U32 n, st).
Proof. intros. reflexivity. Qed.

(** A constant F32 evaluates to F32 in any state. *)
Lemma agpu_eval_const_f32 :
  forall f st, agpu_eval_ir st (IEConst (CFloat32 f)) = Some (F32 f, st).
Proof. intros. reflexivity. Qed.

(** threadIdx.x evaluates to the tidx field of the thread constants. *)
Lemma agpu_eval_thread_idx_x :
  forall st, agpu_eval_ir st IEThreadIdxX = Some (U32 st.(tc).(tidx), st).
Proof. intros. reflexivity. Qed.

(** Barrier is an identity on state. *)
Lemma agpu_eval_barrier_is_identity :
  forall st, agpu_eval_ir st IEBarrier = Some (U32 0, st).
Proof. intros. reflexivity. Qed.

(** global_idx computes blockIdx.x * blockDim.x + threadIdx.x. *)
Lemma agpu_eval_global_idx :
  forall st,
    agpu_eval_ir st IEGlobalIdx =
      Some (U32 (st.(tc).(bidx) * st.(tc).(bdim) + st.(tc).(tidx)), st).
Proof. intros. reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** * Right-shift lemmas (finding D-54)

 * [Shr] is the ARITHMETIC shift ([shr.s32]/[shr.s64]); [Lshr32]/[Lshr64] are
 * the logical ones ([shr.b32]/[shr.b64]). The previous model had [Shr] as a
 * logical [Nat.shiftr] and no [Lshr*]. Numeric facts about the instructions
 * themselves live in PtxShift.v (over Z, where 32/64-bit patterns can be
 * written down); these lift them to the [nat]-valued register domain.
 *)
(* ------------------------------------------------------------------ *)

Lemma agpu_eval_shr_is_signed_u32 : forall a b,
  agpu_eval_binop Shr (U32 a) (U32 b) =
    Some (U32 (Z.to_nat (ashr_w 32 (Z.of_nat a) (Z.of_nat b)))).
Proof. intros. reflexivity. Qed.

Lemma agpu_eval_lshr32_is_logical_u32 : forall a b,
  agpu_eval_binop Lshr32 (U32 a) (U32 b) =
    Some (U32 (Z.to_nat (lshr_w 32 (Z.of_nat a) (Z.of_nat b)))).
Proof. intros. reflexivity. Qed.

Lemma agpu_eval_shr_is_signed_u64 : forall a b,
  agpu_eval_binop Shr (U64 a) (U64 b) =
    Some (U64 (Z.to_nat (ashr_w 64 (Z.of_nat a) (Z.of_nat b)))).
Proof. intros. reflexivity. Qed.

Lemma agpu_eval_lshr64_is_logical_u64 : forall a b,
  agpu_eval_binop Lshr64 (U64 a) (U64 b) =
    Some (U64 (Z.to_nat (lshr_w 64 (Z.of_nat a) (Z.of_nat b)))).
Proof. intros. reflexivity. Qed.

(** [Lshr32]/[Lshr64] on the wrong register class are ill-typed PTX. *)
Lemma agpu_eval_lshr32_wide_ill_typed : forall a b,
  agpu_eval_binop Lshr32 (U64 a) (U64 b) = None.
Proof. intros. reflexivity. Qed.

Lemma agpu_eval_lshr64_narrow_ill_typed : forall a b,
  agpu_eval_binop Lshr64 (U32 a) (U32 b) = None.
Proof. intros. reflexivity. Qed.

(** Sign fill, at the [nat] level: a 32-bit operand with the top bit set
 *  (a negative [int32]) stays negative under [Shr] — the property the old
 *  logical [Shr] violated. *)
Theorem agpu_eval_shr_u32_sign_fill : forall a b r,
  (2 ^ 31 <= Z.of_nat a < 2 ^ 32)%Z ->
  agpu_eval_binop Shr (U32 a) (U32 b) = Some (U32 r) ->
  (2 ^ 31 <= Z.of_nat r)%Z.
Proof.
  intros a b r Ha Heval. rewrite agpu_eval_shr_is_signed_u32 in Heval.
  injection Heval as Hr. subst r.
  pose proof (ashr_range 32 (Z.of_nat a) (Z.of_nat b)) as [R0 R1].
  rewrite Z2Nat.id by exact R0.
  assert (Hmod : to_unsigned 32 (Z.of_nat a) = Z.of_nat a).
  { apply to_unsigned_id. unfold word_modulus. simpl. lia. }
  pose proof (ashr_negative_stays_negative 32 (Z.of_nat a) (Z.of_nat b)
                (ltac:(lia)) (ltac:(rewrite Hmod; simpl; lia))) as H.
  simpl in H. exact H.
Qed.

(** Injectivity of the [Z]/[nat] bridge on [U32] results (stated over
 *  variables so that no 32-bit numeral is ever unfolded to Peano form). *)
Lemma some_u32_to_nat_inj : forall x y : Z,
  (0 <= x)%Z -> (0 <= y)%Z ->
  Some (U32 (Z.to_nat x)) = Some (U32 (Z.to_nat y)) -> x = y.
Proof.
  intros x y Hx Hy H. injection H as H. apply Z2Nat.inj; assumption.
Qed.

(** The two shifts differ on negative operands: there is an operand and an
 *  amount for which [Shr] and [Lshr32] disagree. *)
Theorem agpu_shr_not_lshr32 :
  exists a b : nat,
    (2 ^ 31 <= Z.of_nat a < 2 ^ 32)%Z /\
    agpu_eval_binop Shr (U32 a) (U32 b) <> agpu_eval_binop Lshr32 (U32 a) (U32 b).
Proof.
  (* [a] is kept opaque: unfolding [Z.to_nat 4294967292] would build a
     four-billion-deep Peano numeral. *)
  remember (Z.to_nat 4294967292) as a eqn:Ha.
  assert (HX : Z.of_nat a = 4294967292%Z)
    by (subst a; apply Z2Nat.id; lia).
  clear Ha.
  exists a, 1%nat. split.
  - rewrite HX. lia.
  - rewrite agpu_eval_shr_is_signed_u32, agpu_eval_lshr32_is_logical_u32, HX.
    intro Heq.
    pose proof (ashr_range 32 4294967292%Z (Z.of_nat 1)) as [A0 _].
    pose proof (lshr_range 32 4294967292%Z (Z.of_nat 1)) as [B0 _].
    apply (some_u32_to_nat_inj _ _ A0 B0) in Heq.
    change (Z.of_nat 1) with 1%Z in Heq.
    rewrite ashr32_minus4_by1, lshr32_minus4_by1 in Heq.
    discriminate.
Qed.
