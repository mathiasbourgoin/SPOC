(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Epilogue coverage without enumeration, for affine store indices
    (specs/epilogue-fusion.md).

    Answers the question {!Launch_epilogue} asks of {!Index_slice} (does P store
    each element of I at most once, at a non-negative index, and every element C
    writes?) in time independent of the launch size, when the relevant slice
    ({!Index_slice.relevant}) of both kernels has this form:
    - straight-line [let]s and [if]s, no loop, [return], [match] or variable
      assignment in the slice, and no [else] branch that matters;
    - every [if] condition is a conjunction ([&&], or [c ? d : false]) of [<],
      [<=], [>], [>=], [==] comparisons;
    - every index and compared value is an affine function, with integer
      coefficients, of {e digits} (below), and its value over the whole launch
      stays within int32, so the device's int32 arithmetic is exact arithmetic;
    - P has exactly one store site to I.

    {b Digits.} The six thread and work-group indices start as one digit each,
    ranging over [\[0, extent)]. Dividing a non-negative value by [2^k], taking
    it modulo [2^k] (or [land (2^k - 1)], or shifting it right by [k]) is exact
    when every coefficient but one digit's is a multiple of [2^k], that digit's
    coefficient is 1, the constant is a multiple of [2^k], and the digit's
    extent divides by [2^k] or is at most [2^k]: the digit [d] is then split
    into [d = lo + 2^k * hi], [lo] in [\[0, 2^k)], and the quotient and
    remainder are affine in the new digits. A warp-per-row index
    [row = block * 8 + tid / 32] with the guard [tid mod 32 == 0] has this form.

    {b Solving.} The digits with a non-zero coefficient in one guard must form a
    {e composite}: sorted by magnitude, the coefficients are [g] times
    [1, e1, e1*e2, ...] where [e_i] are the digits' extents. The composite [L]
    ranges bijectively over [\[0, product of extents)] and the guard is an
    interval of [L]. Composites of different guards are equal or share no digit.
    Every other digit, and every composite whose interval is all of it, counts
    as free. P's store set is the image of this box. The index must be
    [sum k_u * u + c] over these units [u]. The image has each element exactly
    once and is one contiguous interval when the units with more than one value,
    sorted by [k], form a mixed radix (first [k] is 1, each next is the previous
    times the previous unit's size). A [k] smaller than the span below it (zero
    included) repeats an element: P stores something twice, and the answer is
    [false]. C's store sets only need bounds.

    {b Integer range.} A digit of extent [e >= 2] whose coefficient is [a] in a
    checked value satisfies [|a| * (e - 1) <= 2^32] (every value is checked to
    stay in int32 over the launch), so [|a| * e <= 2^33]: products of a
    coefficient and a digit value, and composite weights, stay far below OCaml's
    63-bit [max_int]. {!scale} refuses a constant product that could overflow.

    Any other form is {!Unknown}: the caller enumerates instead. *)

open Sarek_ir_types

exception Unknown of string

let unknown fmt = Printf.ksprintf (fun s -> raise (Unknown s)) fmt

(** {1 Digits and affine values} *)

(* (digit, coefficient), sorted by digit, no zero coefficient. *)
type aff = {co : (int * int) list; c : int}

type space = {
  dims : int array;  (** block x y z, grid x y z *)
  extent : (int, int) Hashtbl.t;  (** of each digit; positions are 0..5 *)
  split : (int, int * int * int) Hashtbl.t;
      (** digit -> (lo, hi, m): digit = lo + m * hi *)
  mutable next : int;  (** next fresh digit *)
  env : env;
}

and env = {
  vars : (int, aff) Hashtbl.t;  (** let-bound relevant variables by id *)
  scalars : (string * int32) list;
  mutable used : string list;  (** scalar parameters read *)
}

let lo32 = Int32.to_int Int32.min_int

let hi32 = Int32.to_int Int32.max_int

let const n = {co = []; c = n}

let extent sp d = Hashtbl.find sp.extent d

let merge co =
  List.sort compare co
  |> List.fold_left
       (fun acc (d, a) ->
         match acc with
         | (d', b) :: rest when d = d' -> (d, a + b) :: rest
         | _ -> (d, a) :: acc)
       []
  |> List.filter (fun (_, a) -> a <> 0)
  |> List.rev

(* Rewrite split digits into their parts; drop digits that are always 0. *)
let rec norm sp a =
  let expand (d, k) =
    match Hashtbl.find_opt sp.split d with
    | Some (lo, hi, m) -> [(lo, k); (hi, k * m)]
    | None -> if extent sp d <= 1 then [] else [(d, k)]
  in
  let co = List.concat_map expand a.co in
  if List.exists (fun (d, _) -> Hashtbl.mem sp.split d) co then
    norm sp {a with co}
  else {a with co = merge co}

(* Bounds of [a] over the whole launch. *)
let range sp a =
  List.fold_left
    (fun (lo, hi) (d, k) ->
      let far = k * (extent sp d - 1) in
      if far < 0 then (lo + far, hi) else (lo, hi + far))
    (a.c, a.c)
    a.co

(* Every value the device computes must be its exact value. *)
let checked sp a =
  let a = norm sp a in
  let lo, hi = range sp a in
  if lo < lo32 || hi > hi32 then unknown "int32 overflow possible" else a

let add sp a b = checked sp {co = a.co @ b.co; c = a.c + b.c}

let scale sp k a =
  let a = norm sp a in
  let big x = x <> 0 && abs k > max_int / (4 * abs x) in
  if big a.c || List.exists (fun (_, x) -> big x) a.co then
    unknown "coefficient too large" ;
  checked sp {co = List.map (fun (d, x) -> (d, k * x)) a.co; c = k * a.c}

let as_const sp a = match norm sp a with {co = []; c} -> Some c | _ -> None

let mul sp a b =
  match (as_const sp a, as_const sp b) with
  | Some k, _ -> scale sp k b
  | _, Some k -> scale sp k a
  | None, None -> unknown "product of two thread-dependent values"

let fresh sp e =
  let d = sp.next in
  sp.next <- d + 1 ;
  Hashtbl.replace sp.extent d e ;
  d

let digit d = {co = [(d, 1)]; c = 0}

(* [a = m * q + r] with [0 <= r < m], both affine (see the module comment). *)
let divmod sp a m =
  let a = norm sp a in
  if fst (range sp a) < 0 then unknown "division of a possibly negative value" ;
  if a.c mod m <> 0 then unknown "constant not a multiple of %d" m ;
  let low, high = List.partition (fun (_, k) -> k mod m <> 0) a.co in
  let q = {co = List.map (fun (d, k) -> (d, k / m)) high; c = a.c / m} in
  match low with
  | [] -> (checked sp q, const 0)
  | [(d, 1)] when extent sp d <= m -> (checked sp q, digit d)
  | [(d, 1)] when extent sp d mod m = 0 ->
      let lo = fresh sp m and hi = fresh sp (extent sp d / m) in
      Hashtbl.replace sp.split d (lo, hi, m) ;
      (add sp q (digit hi), digit lo)
  | _ -> unknown "division by %d does not split a digit" m

let power_of_two sp b =
  match as_const sp b with
  | Some m when m > 0 && m land (m - 1) = 0 -> m
  | _ -> unknown "not a positive power-of-two constant"

let shift sp b =
  match as_const sp b with
  | Some k when k >= 0 && k <= 30 -> 1 lsl k
  | _ -> unknown "shift amount not a constant in [0, 30]"

(* [land (2^k - 1)] on either side. *)
let mask sp a b =
  let low_bits x =
    match as_const sp x with
    | Some m when m >= 0 && m land (m + 1) = 0 -> Some (m + 1)
    | _ -> None
  in
  match (low_bits b, low_bits a) with
  | Some m, _ -> snd (divmod sp a m)
  | None, Some m -> snd (divmod sp b m)
  | None, None -> unknown "mask is not 2^k - 1"

let position sp k = checked sp (digit k)

(* Global index along an axis: block_idx * block_dim + thread_idx. *)
let global sp axis =
  add sp (position sp axis) (scale sp sp.dims.(axis) (position sp (axis + 3)))

let intrinsic sp name =
  let dim k = const sp.dims.(k) in
  match name with
  | "thread_idx_x" | "thread_id_x" -> position sp 0
  | "thread_idx_y" | "thread_id_y" -> position sp 1
  | "thread_idx_z" | "thread_id_z" -> position sp 2
  | "block_idx_x" | "block_id_x" -> position sp 3
  | "block_idx_y" | "block_id_y" -> position sp 4
  | "block_idx_z" | "block_id_z" -> position sp 5
  | "block_dim_x" -> dim 0
  | "block_dim_y" -> dim 1
  | "block_dim_z" -> dim 2
  | "grid_dim_x" -> dim 3
  | "grid_dim_y" -> dim 4
  | "grid_dim_z" -> dim 5
  | "global_thread_id" | "global_idx" | "global_idx_x" -> global sp 0
  | "global_idx_y" -> global sp 1
  | "global_idx_z" -> global sp 2
  | "global_size" | "global_size_x" ->
      checked sp (const (sp.dims.(0) * sp.dims.(3)))
  | n -> unknown "intrinsic %s" n

let scalar sp name =
  match List.assoc_opt name sp.env.scalars with
  | Some n ->
      if not (List.mem name sp.env.used) then sp.env.used <- name :: sp.env.used ;
      const (Int32.to_int n)
  | None -> unknown "variable %s" name

let rec eval sp e =
  let ev = eval sp in
  match e with
  | EConst (CInt32 n) -> const (Int32.to_int n)
  | EVar v -> (
      match Hashtbl.find_opt sp.env.vars v.var_id with
      | Some a -> a
      | None -> scalar sp v.var_name)
  | EIntrinsic (_, name, []) -> intrinsic sp name
  | EBinop (Add, a, b) -> add sp (ev a) (ev b)
  | EBinop (Sub, a, b) -> add sp (ev a) (scale sp (-1) (ev b))
  | EBinop (Mul, a, b) -> mul sp (ev a) (ev b)
  | EBinop (Div, a, b) -> fst (divmod sp (ev a) (power_of_two sp (ev b)))
  | EBinop (Mod, a, b) -> snd (divmod sp (ev a) (power_of_two sp (ev b)))
  | EBinop ((Shr | Lshr32), a, b) -> fst (divmod sp (ev a) (shift sp (ev b)))
  | EBinop (Shl, a, b) -> scale sp (shift sp (ev b)) (ev a)
  | EBinop (BitAnd, a, b) -> mask sp (ev a) (ev b)
  | EUnop (Neg, a) -> scale sp (-1) (ev a)
  | ECast (TInt32, a) -> ev a
  | _ -> unknown "not an affine integer expression"

(** {1 Guards} *)

(* A guard is a list of affine values [f], each meaning [f <= 0]. *)
let rec guard sp cond =
  let le a b = [add sp a (scale sp (-1) b)] in
  let one = const 1 in
  match cond with
  | EConst (CBool true) -> []
  | EBinop (And, a, b) | EIf (a, b, EConst (CBool false)) ->
      guard sp a @ guard sp b
  | EBinop (Lt, a, b) -> le (add sp (eval sp a) one) (eval sp b)
  | EBinop (Le, a, b) -> le (eval sp a) (eval sp b)
  | EBinop (Gt, a, b) -> le (add sp (eval sp b) one) (eval sp a)
  | EBinop (Ge, a, b) -> le (eval sp b) (eval sp a)
  | EBinop (Eq, a, b) ->
      let a = eval sp a and b = eval sp b in
      le a b @ le b a
  | _ -> unknown "guard is not a conjunction of comparisons"

(** {1 Store sites} *)

type site = {guards : aff list; index : aff}

let rec sites sp ~is_target rel guards st acc =
  let go = sites sp ~is_target rel guards in
  if not (Index_slice.matters ~is_target rel st) then acc
  else
    match st with
    | SAssign (LArrayElem (a, i), _) when is_target a ->
        {guards; index = eval sp i} :: acc
    | SLet (v, e, b) | SLetMut (v, e, b) ->
        if Index_slice.Ids.mem v.var_id rel then
          Hashtbl.replace sp.env.vars v.var_id (eval sp e) ;
        go b acc
    | SSeq ss -> List.fold_left (fun acc s -> go s acc) acc ss
    | SIf (c, t, f) ->
        if Option.fold ~none:false ~some:(Index_slice.matters ~is_target rel) f
        then unknown "else branch around a store" ;
        sites sp ~is_target rel (guard sp c @ guards) t acc
    | SBlock b | SPragma (_, b) -> go b acc
    | _ -> unknown "loop, return, assignment or match around a store"

let space ~(dims : Index_slice.dims) ~scalars =
  let bx, by, bz = dims.block and gx, gy, gz = dims.grid in
  let dims = [|bx; by; bz; gx; gy; gz|] in
  let extent = Hashtbl.create 16 in
  Array.iteri (fun k e -> Hashtbl.replace extent k e) dims ;
  {
    dims;
    extent;
    split = Hashtbl.create 4;
    next = 6;
    env = {vars = Hashtbl.create 8; scalars; used = []};
  }

let store_sites ~is_target ~scalars ~dims body =
  let sp = space ~dims ~scalars in
  let rel = Index_slice.relevant ~is_target body in
  let ss = sites sp ~is_target rel [] body [] in
  (sp, ss)

(** {1 Composites} *)

(* Digits with their weights (first weight 1, ascending), and an inclusive
   interval of [sum weight * digit]. *)
type composite = {digits : (int * int) list; lo : int; hi : int}

let size_of sp digits =
  List.fold_left (fun n (d, _) -> n * extent sp d) 1 digits

(* [f] as [g * L] plus its constant, over one composite [L], [g] signed. *)
let composite_of sp f =
  let sorted = List.sort (fun (_, a) (_, b) -> compare (abs a) (abs b)) f.co in
  match sorted with
  | [] -> None
  | (_, g) :: _ ->
      let step (w, acc) (d, a) =
        if a <> g * w then unknown "guard digits are not one composite" ;
        (w * extent sp d, (d, w) :: acc)
      in
      let _, digits = List.fold_left step (1, []) sorted in
      Some (List.rev digits, g)

let cdiv n d = if n >= 0 then (n + d - 1) / d else -(-n / d)

let fdiv n d = if n >= 0 then n / d else -((-n + d - 1) / d)

let shares a b = List.exists (fun (d, _) -> List.mem_assoc d b) a

(* Narrow the composites [cs] by [g * L + c <= 0]; [None] when impossible. *)
let constrain sp cs f =
  match composite_of sp f with
  | None -> if f.c <= 0 then Some cs else None
  | Some (digits, g) ->
      let lo, hi =
        if g > 0 then (0, fdiv (-f.c) g) else (cdiv f.c (-g), max_int)
      in
      let base =
        match List.filter (fun k -> shares k.digits digits) cs with
        | [] -> {digits; lo = 0; hi = size_of sp digits - 1}
        | [k] when k.digits = digits -> k
        | _ -> unknown "guards on overlapping composites"
      in
      let k = {base with lo = max base.lo lo; hi = min base.hi hi} in
      let others = List.filter (fun k -> not (shares k.digits digits)) cs in
      if k.lo > k.hi then None else Some (k :: others)

(* The guarded composites of a site that do not span their whole range;
   [None] when the site never stores. *)
let box sp s =
  let rec go cs = function
    | [] -> Some cs
    | f :: fs -> (
        match constrain sp cs (norm sp f) with
        | None -> None
        | Some cs -> go cs fs)
  in
  Option.map
    (List.filter (fun k -> k.lo > 0 || k.hi < size_of sp k.digits - 1))
    (go [] s.guards)

(* Digits that take more than one value. *)
let live sp =
  Hashtbl.fold
    (fun d e acc ->
      if e > 1 && not (Hashtbl.mem sp.split d) then d :: acc else acc)
    sp.extent
    []

(* The index as units (coefficient, lo, hi) plus its constant: one per
   guarded composite, one per other live digit. *)
let units sp s =
  match box sp s with
  | None -> None
  | Some guarded ->
      let index = norm sp s.index in
      let coef d = Option.value ~default:0 (List.assoc_opt d index.co) in
      let of_composite k =
        let d0, _ = List.hd k.digits in
        let kk = coef d0 in
        if List.exists (fun (d, w) -> coef d <> kk * w) k.digits then
          unknown "index not proportional to a guarded composite" ;
        (kk, k.lo, k.hi)
      in
      let free =
        List.filter
          (fun d ->
            not (List.exists (fun k -> List.mem_assoc d k.digits) guarded))
          (live sp)
        |> List.map (fun d -> (coef d, 0, extent sp d - 1))
      in
      Some (free @ List.map of_composite guarded, index.c)

(** {1 Store sets} *)

let bounds_of (us, c) =
  List.fold_left
    (fun (lo, hi) (k, x, y) ->
      (lo + min (k * x) (k * y), hi + max (k * x) (k * y)))
    (c, c)
    us

(* Bounds of the indices a site stores, or [None] when it never stores. *)
let bounds sp s = Option.map bounds_of (units sp s)

type stores = Exactly_once of (int * int) option | Repeats

(* The mixed-radix test of the module comment. *)
let exactly_once sp s =
  match units sp s with
  | None -> Exactly_once None
  | Some u -> (
      let varying =
        List.filter (fun (_, x, y) -> x < y) (fst u)
        |> List.map (fun (k, x, y) -> (k, y - x + 1))
        |> List.sort compare
      in
      let step span (a, n) =
        if a < 0 then unknown "negative coefficient"
        else if a < span then raise_notrace Exit
        else if a > span then unknown "gaps between stored elements"
        else span * n
      in
      match List.fold_left step 1 varying with
      | _ -> Exactly_once (Some (bounds_of u))
      | exception Exit -> Repeats)

(** [covers] returns the verdict and the scalar parameters it read ([("p", n)]
    for P's, [("c", n)] for C's), as {!Launch_epilogue}'s enumeration does.
    Raises {!Unknown} when either kernel is outside the affine form. *)
let covers ~p_body ~p_array ~p_scalars ~p_dims ~c_body ~c_written ~c_scalars
    ~c_dims =
  let psp, psites =
    store_sites
      ~is_target:(fun a -> a = p_array)
      ~scalars:p_scalars
      ~dims:p_dims
      p_body
  in
  let csp, csites =
    store_sites
      ~is_target:(fun a -> List.mem a c_written)
      ~scalars:c_scalars
      ~dims:c_dims
      c_body
  in
  let inside (lo, hi) s =
    match bounds csp s with None -> true | Some (x, y) -> lo <= x && y <= hi
  in
  let ok =
    match psites with
    | [s] -> (
        match exactly_once psp s with
        | Repeats -> false
        | Exactly_once None ->
            List.for_all (fun s -> bounds csp s = None) csites
        | Exactly_once (Some (lo, hi)) ->
            lo >= 0 && List.for_all (inside (lo, hi)) csites)
    | _ -> unknown "P has %d store sites" (List.length psites)
  in
  ( ok,
    List.map (fun n -> ("p", n)) psp.env.used
    @ List.map (fun n -> ("c", n)) csp.env.used )
