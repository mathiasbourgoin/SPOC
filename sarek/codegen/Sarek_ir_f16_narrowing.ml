(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The exactness analysis behind dropping sarek_f32_barrier. The theorem and its
   conditions are in specs/f16-narrowing-barrier.md §2; this file is its
   implementation, and every approximation in it errs towards keeping the
   barrier. *)

open Sarek_ir_types

(* ---- facts about one function body ---- *)

(* How a name gets a value. A name may be bound several times (shadowing,
   sibling scopes, assignments): an expression that reads it may see any of
   them, so every question about a variable is answered for all its bindings. *)
type def =
  | Bound of expr  (** let, mutable let, assignment, initialised local *)
  | Loop of expr * expr  (** for-loop variable: between the two bounds *)
  | Argument  (** scalar kernel argument: an opaque value *)
  | Unknown  (** helper parameter, pattern variable, uninitialised local *)

type facts = {
  defs : (string, def) Hashtbl.t;
  elt : (string, elttype) Hashtbl.t;  (** element type of each array *)
  global : (string, unit) Hashtbl.t;  (** global array parameters *)
  written : (string, unit) Hashtbl.t;
      (** arrays stored to, or passed anywhere as a value *)
  mutable opaque : bool;
      (** an array is written through an expression this analysis does not
          resolve, or inline native code runs: no array is read-only *)
}

let create () =
  {
    defs = Hashtbl.create 32;
    elt = Hashtbl.create 8;
    global = Hashtbl.create 8;
    written = Hashtbl.create 8;
    opaque = false;
  }

let empty = create ()

let add_def f name d = Hashtbl.add f.defs name d

let mark_written f name = Hashtbl.replace f.written name ()

let pattern_vars f = function
  | PConstr (_, names) -> List.iter (fun n -> add_def f n Unknown) names
  | PWild -> ()

(* The array an l-value writes into, if any. *)
let rec note_store f = function
  | LVar v -> add_def f v.var_name Unknown
  | LArrayElem (a, _) | LArrayElemExpr (EVar {var_name = a; _}, _) ->
      mark_written f a
  | LArrayElemExpr _ -> f.opaque <- true
  | LRecordField (lv, _) -> note_store f lv

let note_binding f v e =
  match e with
  | EArrayCreate (elt, _, _) -> Hashtbl.replace f.elt v.var_name elt
  | _ -> add_def f v.var_name (Bound e)

let on_stmt f () = function
  | SLet (v, e, _) | SLetMut (v, e, _) -> note_binding f v e
  | SAssign (LVar v, e) -> add_def f v.var_name (Bound e)
  | SAssign (lv, _) -> note_store f lv
  | SFor (v, lo, hi, _, _) -> add_def f v.var_name (Loop (lo, hi))
  | SMatch (_, cases) -> List.iter (fun (p, _) -> pattern_vars f p) cases
  | SCoopmat (CM_store {dst; _}) -> mark_written f dst
  | _ -> ()

let on_expr f () = function
  | EMatch (_, cases) -> List.iter (fun (p, _) -> pattern_vars f p) cases
  | EVar {var_type = TVec _ | TArray _; var_name; _} ->
      (* An array as a value (a helper argument, an intrinsic operand) may be
         written by whatever receives it. *)
      mark_written f var_name
  | _ -> ()

let scan_body f body =
  let folder =
    {
      Sarek_ir_analysis.fe = on_expr f;
      ft = (fun () _ -> ());
      fs = on_stmt f;
      fnative = (fun () -> f.opaque <- true);
      visit_lvalue = true;
    }
  in
  Sarek_ir_analysis.stmt_fold folder () body

let array_elt (v : var) =
  match v.var_type with TVec elt | TArray (elt, _) -> Some elt | _ -> None

let on_param f = function
  | DParam (v, Some ai) ->
      Hashtbl.replace f.elt v.var_name ai.arr_elttype ;
      if ai.arr_memspace = Global then Hashtbl.replace f.global v.var_name ()
  | DParam (v, None) -> (
      match array_elt v with
      | Some elt -> Hashtbl.replace f.elt v.var_name elt
      | None -> add_def f v.var_name Argument)
  | DLocal (v, Some e) -> note_binding f v e
  | DLocal (v, None) -> add_def f v.var_name Unknown
  | DShared (name, elt, _) -> Hashtbl.replace f.elt name elt

let of_kernel k =
  let f = create () in
  List.iter (on_param f) k.kern_params ;
  List.iter (on_param f) k.kern_locals ;
  scan_body f k.kern_body ;
  f

let of_helper hf =
  let f = create () in
  List.iter
    (fun v ->
      match array_elt v with
      | Some elt -> Hashtbl.replace f.elt v.var_name elt
      | None -> add_def f v.var_name Unknown)
    hf.hf_params ;
  scan_body f hf.hf_body ;
  f

let read_only_f32 f a =
  Hashtbl.find_opt f.elt a = Some TFloat32
  && Hashtbl.mem f.global a
  && (not (Hashtbl.mem f.written a))
  && not f.opaque

(* ---- typing (only what the analysis needs) ---- *)

(* The stdlib module [m] as kernel code names it: bare, or qualified by one of
   the two stdlib libraries (specs/stdlib-bitcasts.md). *)
let stdlib_path m path =
  path = [m] || path = ["Sarek_stdlib"; m] || path = ["Sarek_stdlib_meta"; m]

let is_int_to_float path name =
  (stdlib_path "Gpu" path && (name = "float" || name = "float_of_int"))
  || (stdlib_path "Float32" path && name = "of_int")

let is_bits_to_float path name =
  stdlib_path "Int32" path && name = "float_of_bits"

let const_type = function
  | CInt32 _ -> TInt32
  | CInt64 _ -> TInt64
  | CFloat32 _ -> TFloat32
  | CFloat64 _ -> TFloat64
  | CBool _ -> TBool
  | CUnit -> TUnit

let rec type_of f = function
  | EConst c -> Some (const_type c)
  | EVar v -> Some v.var_type
  | EArrayRead (a, _) -> Hashtbl.find_opt f.elt a
  | EBinop ((Eq | Ne | Lt | Le | Gt | Ge | And | Or), _, _) | EUnop (Not, _) ->
      Some TBool
  | EBinop (_, a, _) | EUnop (_, a) | EIf (_, a, _) -> type_of f a
  | ECast (t, _) -> Some t
  | EIntrinsic (path, name, _)
    when is_int_to_float path name || is_bits_to_float path name ->
      Some TFloat32
  | _ -> None

(* ---- queries ---- *)

(* One query. [fuel] bounds the work: a variable bound n times is explored n
   times, and chains of such variables would multiply. Running out answers
   "unknown", which is always sound. *)
type ctx = {facts : facts; mutable fuel : int}

let spend c =
  c.fuel <- c.fuel - 1 ;
  c.fuel >= 0

(* Every binding of [v], or [None] when [v] is being explored already (a
   cycle), has no binding, or the fuel ran out. *)
let bindings c seen (v : var) =
  if List.mem v.var_name seen || not (spend c) then None
  else
    match Hashtbl.find_all c.facts.defs v.var_name with
    | [] -> None
    | first :: rest -> Some (v.var_name :: seen, first, rest)

(* ---- int32 value ranges ---- *)

open Sarek_ir_int_range

let rec int_range c seen e =
  match e with
  | EConst (CInt32 n) -> point (Int32.to_int n)
  | EConst (CBool _) -> {lo = 0; hi = 1}
  | EVar v when v.var_type = TInt32 -> var_range c seen v
  | EBinop (op, a, b) ->
      binop_range op (int_range c seen a) (int_range c seen b)
  | EUnop (op, a) -> unop_range op (int_range c seen a)
  | EIf (_, a, b) -> join_range (int_range c seen a) (int_range c seen b)
  | _ -> full

and var_range c seen v =
  match bindings c seen v with
  | None -> full
  | Some (seen, first, rest) ->
      let one = function
        | Bound e -> int_range c seen e
        | Loop (lo, hi) ->
            join_range (int_range c seen lo) (int_range c seen hi)
        | Argument | Unknown -> full
      in
      List.fold_left (fun acc d -> join_range acc (one d)) (one first) rest

(* ---- exact f32 values ---- *)

type bound = {
  kmax : float;
  qlo : int;
  qhi : int;
  binary16 : bool;
  nonzero : bool;
  nonneg : bool;
}

let two24 = 16777216.0

let max_f32 = Int32.float_of_bits 0x7f7fffffl

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Any finite f32 value. A leaf: nothing feeds it, so nothing can be fused. *)
let any_f32 =
  {
    kmax = two24;
    qlo = -149;
    qhi = 104;
    binary16 = false;
    nonzero = false;
    nonneg = false;
  }

(* Any finite binary16 value: (1024..2047) * 2^(e - 10) or (0..1023) * 2^-24. *)
let any_f16 =
  {
    kmax = 2047.0;
    qlo = -24;
    qhi = 5;
    binary16 = true;
    nonzero = false;
    nonneg = false;
  }

(* A zero; [nonneg] when it is known to be +0. *)
let zero ~nonneg =
  {kmax = 0.0; qlo = 0; qhi = 0; binary16 = true; nonzero = false; nonneg}

(* The result of an operation is exact in f32 when it fits 24 significant bits,
   stays clear of the subnormal range (so a flush-to-zero mode cannot change
   it) and below the largest finite f32. *)
let representable b =
  b.kmax <= two24 && b.qlo >= -126 && Float.ldexp b.kmax b.qhi <= max_f32

let const_bound x =
  if (not (Float.is_finite x)) || round32 x <> x then None
  else if x = 0.0 then Some (zero ~nonneg:(Float.sign_bit x = false))
  else
    let m, e = Float.frexp (Float.abs x) in
    let rec strip k q =
      if Float.rem k 2.0 = 0.0 then strip (k /. 2.0) (q + 1) else (k, q)
    in
    let k, q = strip (Float.ldexp m 24) (e - 24) in
    let binary16 = k <= 2047.0 && q >= -24 && Float.abs x <= 65504.0 in
    Some
      {kmax = k; qlo = q; qhi = q; binary16; nonzero = true; nonneg = x > 0.0}

let range_bound r =
  let m = max (abs r.lo) (abs r.hi) in
  if float_of_int m > two24 then None
  else
    Some
      {
        kmax = float_of_int m;
        qlo = 0;
        qhi = 0;
        binary16 = m <= 2048;
        nonzero = r.lo > 0 || r.hi < 0;
        (* float 0 is +0 *)
        nonneg = r.lo >= 0;
      }

let mul_bound a b =
  let nonneg = a.nonneg && b.nonneg in
  if a.kmax = 0.0 || b.kmax = 0.0 then zero ~nonneg
  else
    {
      kmax = a.kmax *. b.kmax;
      qlo = a.qlo + b.qlo;
      qhi = a.qhi + b.qhi;
      binary16 = false;
      nonzero = a.nonzero && b.nonzero;
      nonneg;
    }

(* a = ka * 2^qa, b = kb * 2^qb, m = min qa qb:
   a + b = (ka * 2^(qa - m) + kb * 2^(qb - m)) * 2^m, and qa - m <= qa_hi - qb_lo. *)
let add_bound a b =
  let nonneg = a.nonneg && b.nonneg in
  if a.kmax = 0.0 then {b with binary16 = false; nonzero = false; nonneg}
  else if b.kmax = 0.0 then {a with binary16 = false; nonzero = false; nonneg}
  else
    let shifted x y = Float.ldexp x.kmax (max 0 (x.qhi - y.qlo)) in
    {
      kmax = shifted a b +. shifted b a;
      qlo = min a.qlo b.qlo;
      qhi = min a.qhi b.qhi;
      binary16 = false;
      nonzero = false;
      nonneg;
    }

let neg_bound a = {a with nonneg = false}

let join_bound a b =
  {
    kmax = Float.max a.kmax b.kmax;
    qlo = min a.qlo b.qlo;
    qhi = max a.qhi b.qhi;
    binary16 = a.binary16 && b.binary16;
    nonzero = a.nonzero && b.nonzero;
    nonneg = a.nonneg && b.nonneg;
  }

let both f a b = match (a, b) with Some x, Some y -> Some (f x y) | _ -> None

let rec exact_in c seen e =
  match e with
  | EConst (CFloat32 x) -> const_bound x
  | ECast (TFloat32, x) -> widened c seen x
  | EIntrinsic (path, name, [x]) when is_int_to_float path name ->
      int_bound c seen x
  | EIntrinsic (path, name, [_]) when is_bits_to_float path name -> Some any_f32
  | EArrayRead (a, _) when read_only_f32 c.facts a -> Some any_f32
  | EVar v when v.var_type = TFloat32 -> var_bound c seen v
  | EBinop (((Add | Sub | Mul) as op), a, b) -> (
      let combine = if op = Mul then mul_bound else add_bound in
      match both combine (exact_in c seen a) (exact_in c seen b) with
      | Some r when representable r -> Some r
      | _ -> None)
  | EUnop (Neg, a) -> Option.map neg_bound (exact_in c seen a)
  | EIf (_, a, b) -> both join_bound (exact_in c seen a) (exact_in c seen b)
  | _ -> None

and widened c seen x =
  match type_of c.facts x with
  | Some TFloat16 -> Some any_f16
  | Some TInt32 -> int_bound c seen x
  | Some TFloat32 -> exact_in c seen x
  | _ -> None

and int_bound c seen x =
  if type_of c.facts x = Some TInt32 then range_bound (int_range c seen x)
  else None

and var_bound c seen v =
  match bindings c seen v with
  | None -> None
  | Some (seen, first, rest) ->
      let one = function
        | Bound e -> exact_in c seen e
        | Argument -> Some any_f32
        | Loop _ | Unknown -> None
      in
      List.fold_left (fun acc d -> both join_bound acc (one d)) (one first) rest

(* The narrowing may absorb the operation at the top of [e] (looking through
   variables and conditionals). A product absorbed into the narrowing is
   computed as v_fma_mixlo_f16(a, b, +0), so a product that is -0 comes out
   +0; a negation is absorbed into its operand's sign, so -(+0) comes out +0
   too. Measured on gfx1100 (specs/f16-narrowing-barrier.md §2.3). Such a top
   is safe only when it cannot be a zero of the wrong sign. *)
let rec fusion_safe c seen e =
  match e with
  | EBinop (Mul, _, _) -> (
      match exact_in c seen e with
      | Some p -> p.nonzero || p.nonneg
      | None -> false)
  | EUnop (Neg, a) -> (
      match exact_in c seen a with Some b -> b.nonzero | None -> false)
  | EIf (_, a, b) -> fusion_safe c seen a && fusion_safe c seen b
  | ECast (TFloat32, x) when type_of c.facts x = Some TFloat32 ->
      fusion_safe c seen x
  | EVar v when v.var_type = TFloat32 -> (
      match bindings c seen v with
      | None -> false
      | Some (seen, first, rest) ->
          List.for_all
            (function
              | Bound e -> fusion_safe c seen e
              | Argument -> true
              | Loop _ | Unknown -> false)
            (first :: rest))
  | _ -> true

let query facts = {facts; fuel = 10_000}

let exact facts e = exact_in (query facts) [] e

(* a +/- b of two binary16 values: double rounding through f32 is innocuous. *)
let binary16_sum facts = function
  | EBinop ((Add | Sub), a, b) -> (
      let c = query facts in
      match (exact_in c [] a, exact_in c [] b) with
      | Some x, Some y -> x.binary16 && y.binary16
      | _ -> false)
  | _ -> false

type form = Barrier | Plain | Signed_product of expr * expr

let form facts e =
  if binary16_sum facts e then Plain
  else
    match exact facts e with
    | None -> Barrier
    | Some _ when fusion_safe (query facts) [] e -> Plain
    | Some _ -> (
        match e with
        | EBinop (Mul, a, b) -> Signed_product (a, b)
        | _ -> Barrier)

let needs_barrier facts e =
  match form facts e with Barrier -> true | Plain | Signed_product _ -> false
