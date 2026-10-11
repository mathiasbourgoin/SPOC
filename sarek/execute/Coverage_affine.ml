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
    - every [if] condition is a conjunction of [<], [<=], [>], [>=] comparisons;
    - every index and compared value is an affine function, with integer
      coefficients, of the thread and work-group indices, and its value over the
      whole launch stays within int32 (so the device's int32 arithmetic is exact
      arithmetic);
    - P has exactly one store site to I.

    Along each axis the pair (work-group index, thread index) is one {e lane}
    [L = b * block_dim + t], a bijection onto [\[0, grid_dim * block_dim)].
    Every value above is then [sum_l a_l * L_l + c]; a guard on one lane is an
    interval of that lane, and P's store set is the image of a box of lanes.
    That image has each element exactly once and is the contiguous interval
    [[lo, hi]] when the lanes with more than one value, sorted by coefficient,
    form a mixed radix: the first coefficient is 1 and each next one is the
    previous coefficient times the previous lane's extent. A lane whose
    coefficient is smaller than the span of the lanes below it repeats elements:
    P stores some element twice, and the answer is [false]. C's store sets only
    need bounds.

    Any other form is {!Unknown}: the caller enumerates instead. *)

open Sarek_ir_types

exception Unknown of string

let unknown fmt = Printf.ksprintf (fun s -> raise (Unknown s)) fmt

(** {1 Affine values over thread and work-group indices} *)

(* Positions 0..5: thread x y z, work-group x y z. *)
type aff = {co : int array; c : int}

type space = {extent : int array;  (** of each position *) env : env}

and env = {
  vars : (int, aff) Hashtbl.t;  (** let-bound relevant variables by id *)
  scalars : (string * int32) list;
  mutable used : string list;  (** scalar parameters read *)
}

let lo32 = Int32.to_int Int32.min_int

let hi32 = Int32.to_int Int32.max_int

let const n = {co = Array.make 6 0; c = n}

let position sp k =
  let co = Array.make 6 0 in
  if sp.extent.(k) > 1 then co.(k) <- 1 ;
  {co; c = 0}

(* Bounds of [a] over the whole launch. *)
let range sp a =
  let lo = ref a.c and hi = ref a.c in
  Array.iteri
    (fun k co ->
      let far = co * (sp.extent.(k) - 1) in
      if far < 0 then lo := !lo + far else hi := !hi + far)
    a.co ;
  (!lo, !hi)

(* Every value the device computes must be its exact value. *)
let checked sp a =
  let lo, hi = range sp a in
  if lo < lo32 || hi > hi32 then unknown "int32 overflow possible" else a

let add sp a b = checked sp {co = Array.map2 ( + ) a.co b.co; c = a.c + b.c}

let scale sp k a =
  let big x = x <> 0 && abs k > max_int / (4 * abs x) in
  if big a.c || Array.exists big a.co then unknown "coefficient too large" ;
  checked sp {co = Array.map (fun x -> k * x) a.co; c = k * a.c}

let is_const a = Array.for_all (( = ) 0) a.co

let mul sp a b =
  if is_const a then scale sp a.c b
  else if is_const b then scale sp b.c a
  else unknown "product of two thread-dependent values"

(* Lane x of [sp]: block_idx_x * block_dim_x + thread_idx_x. *)
let lane sp axis =
  add sp (position sp axis) (scale sp sp.extent.(axis) (position sp (axis + 3)))

let intrinsic sp name =
  let dim k = const sp.extent.(k) in
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
  | "global_thread_id" | "global_idx" | "global_idx_x" -> lane sp 0
  | "global_idx_y" -> lane sp 1
  | "global_idx_z" -> lane sp 2
  | "global_size" | "global_size_x" ->
      checked sp (const (sp.extent.(0) * sp.extent.(3)))
  | n -> unknown "intrinsic %s" n

let scalar sp name =
  match List.assoc_opt name sp.env.scalars with
  | Some n ->
      if not (List.mem name sp.env.used) then sp.env.used <- name :: sp.env.used ;
      const (Int32.to_int n)
  | None -> unknown "variable %s" name

let rec eval sp e =
  match e with
  | EConst (CInt32 n) -> const (Int32.to_int n)
  | EVar v -> (
      match Hashtbl.find_opt sp.env.vars v.var_id with
      | Some a -> a
      | None -> scalar sp v.var_name)
  | EIntrinsic (_, name, []) -> intrinsic sp name
  | EBinop (Add, a, b) -> add sp (eval sp a) (eval sp b)
  | EBinop (Sub, a, b) -> add sp (eval sp a) (scale sp (-1) (eval sp b))
  | EBinop (Mul, a, b) -> mul sp (eval sp a) (eval sp b)
  | EUnop (Neg, a) -> scale sp (-1) (eval sp a)
  | ECast (TInt32, a) -> eval sp a
  | _ -> unknown "not an affine integer expression"

(** {1 Lanes} *)

(* [a] as (coefficient per lane, constant). *)
let lanes sp a =
  let coef axis =
    let t = a.co.(axis) and b = a.co.(axis + 3) in
    if sp.extent.(axis + 3) = 1 then t
    else if sp.extent.(axis) = 1 then b
    else if b = t * sp.extent.(axis) then t
    else unknown "thread and work-group indices not in one lane"
  in
  (Array.init 3 coef, a.c)

let lane_extent sp axis = sp.extent.(axis) * sp.extent.(axis + 3)

(* Ceiling and floor of [n / d], [d > 0]. *)
let cdiv n d = if n >= 0 then (n + d - 1) / d else -(-n / d)

let fdiv n d = if n >= 0 then n / d else -((-n + d - 1) / d)

(** {1 Guards} *)

(* A guard is a box of lanes: inclusive [lo, hi] per lane. *)
type box = (int * int) array

(* Narrow [box] by [f <= 0]. *)
let restrict sp (box : box) f =
  let co, c = lanes sp f in
  match List.filter (fun l -> co.(l) <> 0) [0; 1; 2] with
  | [] -> if c <= 0 then box else Array.map (fun _ -> (1, 0)) box
  | [l] ->
      let lo, hi = box.(l) and a = co.(l) in
      let box = Array.copy box in
      if a > 0 then box.(l) <- (lo, min hi (fdiv (-c) a))
      else box.(l) <- (max lo (cdiv c (-a)), hi) ;
      box
  | _ -> unknown "guard on several lanes"

let rec guard sp box cond =
  let le a b = restrict sp box (add sp a (scale sp (-1) b)) in
  let one = const 1 in
  match cond with
  | EConst (CBool true) -> box
  | EBinop (And, a, b) -> guard sp (guard sp box a) b
  | EBinop (Lt, a, b) -> le (add sp (eval sp a) one) (eval sp b)
  | EBinop (Le, a, b) -> le (eval sp a) (eval sp b)
  | EBinop (Gt, a, b) -> le (add sp (eval sp b) one) (eval sp a)
  | EBinop (Ge, a, b) -> le (eval sp b) (eval sp a)
  | _ -> unknown "guard is not a conjunction of comparisons"

(** {1 Store sites} *)

type site = {box : box; index : aff}

let rec sites sp ~is_target rel box st acc =
  let go = sites sp ~is_target rel box in
  if not (Index_slice.matters ~is_target rel st) then acc
  else
    match st with
    | SAssign (LArrayElem (a, i), _) when is_target a ->
        {box; index = eval sp i} :: acc
    | SLet (v, e, b) | SLetMut (v, e, b) ->
        if Index_slice.Ids.mem v.var_id rel then
          Hashtbl.replace sp.env.vars v.var_id (eval sp e) ;
        go b acc
    | SSeq ss -> List.fold_left (fun acc s -> go s acc) acc ss
    | SIf (c, t, f) ->
        if Option.fold ~none:false ~some:(Index_slice.matters ~is_target rel) f
        then unknown "else branch around a store" ;
        sites sp ~is_target rel (guard sp box c) t acc
    | SBlock b | SPragma (_, b) -> go b acc
    | _ -> unknown "loop, return, assignment or match around a store"

let space ~(dims : Index_slice.dims) ~scalars =
  let bx, by, bz = dims.block and gx, gy, gz = dims.grid in
  {
    extent = [|bx; by; bz; gx; gy; gz|];
    env = {vars = Hashtbl.create 8; scalars; used = []};
  }

let store_sites ~is_target ~scalars ~dims body =
  let sp = space ~dims ~scalars in
  let rel = Index_slice.relevant ~is_target body in
  let box = Array.init 3 (fun l -> (0, lane_extent sp l - 1)) in
  let ss = sites sp ~is_target rel box body [] in
  (sp, ss)

(** {1 Store sets} *)

let empty (box : box) = Array.exists (fun (lo, hi) -> lo > hi) box

(* Bounds of the indices a site stores, or [None] when it never stores. *)
let bounds sp s =
  if empty s.box then None
  else
    let co, c = lanes sp s.index in
    let lo = ref c and hi = ref c in
    Array.iteri
      (fun l a ->
        let x, y = s.box.(l) in
        lo := !lo + min (a * x) (a * y) ;
        hi := !hi + max (a * x) (a * y))
      co ;
    Some (!lo, !hi)

type stores = Exactly_once of (int * int) option | Repeats

(* The mixed-radix test of the module comment. *)
let exactly_once sp s =
  match bounds sp s with
  | None -> Exactly_once None
  | Some b -> (
      let co, _ = lanes sp s.index in
      let varying =
        List.filter (fun l -> fst s.box.(l) < snd s.box.(l)) [0; 1; 2]
        |> List.map (fun l -> (co.(l), snd s.box.(l) - fst s.box.(l) + 1))
        |> List.sort compare
      in
      let step span (a, n) =
        if a < 0 then unknown "negative lane coefficient"
        else if a < span then raise_notrace Exit
        else if a > span then unknown "gaps between stored elements"
        else span * n
      in
      match List.fold_left step 1 varying with
      | _ -> Exactly_once (Some b)
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
