(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Affine forms over index atoms with uniform polynomial coefficients. See the
   interface for the vocabulary and specs/matrix-pattern-advice.md for the use. *)

open Sarek_ir_types

type monomial = {coef : int; factors : expr list}

type poly = monomial list

type t = {terms : (expr * poly) list; const : poly}

type leaf = Uniform_leaf | Atom_leaf | Rejected_leaf

type env = {classify_var : var -> leaf; definition : var -> expr option}

type error = Not_affine of expr | Data_dependent of expr | Unknown_var of var

(* ---- Polynomials ------------------------------------------------------- *)

let normalize (p : poly) : poly =
  let sorted =
    List.map (fun m -> {m with factors = List.sort compare m.factors}) p
    |> List.sort (fun m1 m2 -> compare m1.factors m2.factors)
  in
  let rec merge = function
    | m1 :: m2 :: rest when m1.factors = m2.factors ->
        merge ({m1 with coef = m1.coef + m2.coef} :: rest)
    | m :: rest -> m :: merge rest
    | [] -> []
  in
  List.filter (fun m -> m.coef <> 0) (merge sorted)

let poly_const n : poly = normalize [{coef = n; factors = []}]

let poly_factor e : poly = [{coef = 1; factors = [e]}]

let poly_add p q = normalize (p @ q)

let poly_mul p q =
  normalize
    (List.concat_map
       (fun m1 ->
         List.map
           (fun m2 ->
             {coef = m1.coef * m2.coef; factors = m1.factors @ m2.factors})
           q)
       p)

(* Exact division by a single monomial: [None] unless every monomial of [p] is
   an integer multiple of [d] whose factors include [d]'s factors. *)
(* [fs] without one occurrence of [f], or [None] when [f] is absent. *)
let rec drop_factor f = function
  | [] -> None
  | g :: gs when g = f -> Some gs
  | g :: gs -> Option.map (fun r -> g :: r) (drop_factor f gs)

let drop_factors fs ds =
  List.fold_left (fun acc f -> Option.bind acc (drop_factor f)) (Some fs) ds

let monomial_div (dm : monomial) m =
  if m.coef mod dm.coef <> 0 then None
  else
    Option.map
      (fun factors -> {coef = m.coef / dm.coef; factors})
      (drop_factors m.factors dm.factors)

let poly_div p (d : poly) =
  match d with
  | [dm] when dm.coef <> 0 ->
      let quotients = List.map (monomial_div dm) p in
      if List.for_all Option.is_some quotients then
        Some (normalize (List.filter_map Fun.id quotients))
      else None
  | _ -> None

let poly_constant = function
  | [] -> Some 0
  | [{coef; factors = []}] -> Some coef
  | _ -> None

let poly_is_zero p = p = []

let poly_is_one p = p = [{coef = 1; factors = []}]

let poly_equal p q = normalize p = normalize q

let int_const n = EConst (CInt32 (Int32.of_int n))

let monomial_to_expr m =
  match (m.coef, m.factors) with
  | n, [] -> int_const n
  | 1, f :: fs -> List.fold_left (fun acc f -> EBinop (Mul, acc, f)) f fs
  | n, fs -> List.fold_left (fun acc f -> EBinop (Mul, acc, f)) (int_const n) fs

let poly_to_expr (p : poly) : expr =
  match p with
  | [] -> int_const 0
  | m :: ms ->
      List.fold_left
        (fun acc m -> EBinop (Add, acc, monomial_to_expr m))
        (monomial_to_expr m)
        ms

(* ---- Affine forms ------------------------------------------------------ *)

let const_form p = {terms = []; const = p}

let atom_form a = {terms = [(a, poly_const 1)]; const = []}

let merge_terms terms =
  let sorted = List.sort (fun (a1, _) (a2, _) -> compare a1 a2) terms in
  let rec go = function
    | (a1, p1) :: (a2, p2) :: rest when a1 = a2 ->
        go ((a1, poly_add p1 p2) :: rest)
    | t :: rest -> t :: go rest
    | [] -> []
  in
  List.filter (fun (_, p) -> not (poly_is_zero p)) (go sorted)

let add f g =
  {terms = merge_terms (f.terms @ g.terms); const = poly_add f.const g.const}

let scale_poly p f =
  {
    terms = merge_terms (List.map (fun (a, q) -> (a, poly_mul p q)) f.terms);
    const = poly_mul p f.const;
  }

let neg f = scale_poly (poly_const (-1)) f

let coefficient f atom =
  match List.assoc_opt atom f.terms with Some p -> p | None -> []

let atoms f = List.map fst f.terms

let of_terms terms = {terms = merge_terms terms; const = []}

let without f excluded =
  {f with terms = List.filter (fun (a, _) -> not (List.mem a excluded)) f.terms}

let is_uniform f = f.terms = []

let to_expr f =
  let terms =
    List.map
      (fun (a, p) ->
        if poly_is_one p then [{coef = 1; factors = [a]}]
        else poly_mul p [{coef = 1; factors = [a]}])
      f.terms
  in
  poly_to_expr (normalize (List.concat terms @ f.const))

(* ---- From expressions -------------------------------------------------- *)

(* Intrinsics whose value is the same for every thread of a launch. *)
let uniform_intrinsics =
  [
    "block_dim_x";
    "block_dim_y";
    "block_dim_z";
    "grid_dim_x";
    "grid_dim_y";
    "grid_dim_z";
  ]

(* The lowering gives a loop variable [var_mutable = true] at its binder and
   [false] at its uses; atoms are compared structurally, so every variable is
   put in one canonical form. *)
let var_atom v = EVar {v with var_mutable = false}

let rec inline env e =
  let go = inline env in
  match e with
  | EVar v -> (
      match env.definition v with Some d -> inline env d | None -> var_atom v)
  | EConst _ | EArrayLen _ -> e
  | EBinop (op, a, b) -> EBinop (op, go a, go b)
  | EUnop (op, a) -> EUnop (op, go a)
  | EArrayRead (arr, idx) -> EArrayRead (arr, go idx)
  | EArrayReadExpr (b, idx) -> EArrayReadExpr (go b, go idx)
  | ERecordField (r, f) -> ERecordField (go r, f)
  | EIntrinsic (p, n, args) -> EIntrinsic (p, n, List.map go args)
  | ECast (t, a) -> ECast (t, go a)
  | ETuple es -> ETuple (List.map go es)
  | EApp (f, args) -> EApp (go f, List.map go args)
  | ERecord (n, fs) -> ERecord (n, List.map (fun (f, x) -> (f, go x)) fs)
  | EVariant (t, c, args) -> EVariant (t, c, List.map go args)
  | EArrayCreate (t, size, ms) -> EArrayCreate (t, go size, ms)
  | EIf (c, a, b) -> EIf (go c, go a, go b)
  | EMatch (s, cases) -> EMatch (go s, List.map (fun (p, x) -> (p, go x)) cases)

(* What an already-inlined expression contains, as one verdict: a rejected
   variable or an array read wins over an atom, which wins over uniform. *)
let pure_intrinsics = ["min"; "max"; "abs"; "fma"]

let is_pure_intrinsic name = List.mem name pure_intrinsics

let rec leaf_kind env e =
  let worst a b =
    match (a, b) with
    | Rejected_leaf, _ | _, Rejected_leaf -> Rejected_leaf
    | Atom_leaf, _ | _, Atom_leaf -> Atom_leaf
    | Uniform_leaf, Uniform_leaf -> Uniform_leaf
  in
  let all es =
    List.fold_left (fun k e -> worst k (leaf_kind env e)) Uniform_leaf es
  in
  match e with
  | EConst _ | EArrayLen _ -> Uniform_leaf
  | EVar v -> env.classify_var v
  | EIntrinsic (_, name, []) ->
      if List.mem name uniform_intrinsics then Uniform_leaf else Atom_leaf
  | EBinop (_, a, b) -> all [a; b]
  | EUnop (_, a) | ECast (_, a) -> leaf_kind env a
  | EIntrinsic (_, name, args) ->
      (* Only pure functions known not to read the thread are uniform on
         uniform arguments; any other intrinsic (a lane id taking unit, say)
         is thread-varying. *)
      let k = all args in
      if k = Uniform_leaf && not (List.mem name pure_intrinsics) then Atom_leaf
      else k
  | _ -> Rejected_leaf

let rec reads_memory = function
  | EArrayRead _ | EArrayReadExpr _ -> true
  | EBinop (_, a, b) -> reads_memory a || reads_memory b
  | EUnop (_, a) | ECast (_, a) -> reads_memory a
  | EIntrinsic (_, _, args) -> List.exists reads_memory args
  | _ -> false

let rec mentions_atom_var env = function
  | EVar v -> env.classify_var v = Atom_leaf
  | EBinop (_, a, b) -> mentions_atom_var env a || mentions_atom_var env b
  | EUnop (_, a) | ECast (_, a) -> mentions_atom_var env a
  | EIntrinsic (_, _, args) -> List.exists (mentions_atom_var env) args
  | _ -> false

(* A sub-expression that is not linear in its operands: an opaque uniform
   factor, an opaque atom (thread-varying but free of every loop variable), or
   a rejection. *)
let opaque env e =
  if reads_memory e then Error (Data_dependent e)
  else if mentions_atom_var env e then Error (Not_affine e)
  else
    match leaf_kind env e with
    | Uniform_leaf -> Ok (const_form (poly_factor e))
    | Atom_leaf -> Ok (atom_form e)
    | Rejected_leaf -> Error (Not_affine e)

let ( let* ) = Result.bind

let rec of_inlined env e =
  match e with
  | EConst (CInt32 n) -> Ok (const_form (poly_const (Int32.to_int n)))
  | EConst (CInt64 n) -> Ok (const_form (poly_const (Int64.to_int n)))
  | EVar v -> (
      match env.classify_var v with
      | Uniform_leaf -> Ok (const_form (poly_factor e))
      | Atom_leaf -> Ok (atom_form e)
      | Rejected_leaf -> Error (Unknown_var v))
  | EBinop (Add, a, b) ->
      let* fa = of_inlined env a in
      let* fb = of_inlined env b in
      Ok (add fa fb)
  | EBinop (Sub, a, b) ->
      let* fa = of_inlined env a in
      let* fb = of_inlined env b in
      Ok (add fa (neg fb))
  | EUnop (Neg, a) ->
      let* fa = of_inlined env a in
      Ok (neg fa)
  | ECast ((TInt32 | TInt64), a) -> of_inlined env a
  | EBinop (Mul, a, b) -> (
      let* fa = of_inlined env a in
      let* fb = of_inlined env b in
      (* A stride that is a sum ([row * (k + p)]) is kept as ONE opaque
         factor rather than distributed: strides are compared and divided
         monomial by monomial, and a padded leading dimension must stay one
         divisor. Two uniform sides still distribute. *)
      let multiplier side f =
        if List.length f.const > 1 then poly_factor (inline env side)
        else f.const
      in
      match (is_uniform fa, is_uniform fb) with
      | true, true -> Ok (scale_poly fa.const fb)
      | true, false -> Ok (scale_poly (multiplier a fa) fb)
      | false, true -> Ok (scale_poly (multiplier b fb) fa)
      | false, false -> opaque env e)
  | _ -> opaque env e

let of_expr env e = of_inlined env (inline env e)

let error_expr = function
  | Not_affine e | Data_dependent e -> Some e
  | Unknown_var _ -> None
