(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - block-scope variance analysis (specs/convergence-semantics.md §3)
 *
 * A whole-kernel fixpoint over the analysis IR (Sarek_variance_ir). Every
 * binder gets an abstract value:
 *
 *   Varying        may differ between the threads of a block
 *   Deps P         uniform as long as the helper parameters in P are
 *                  (Deps {} is uniform)
 *
 * Deps lets a helper's result be uniform at one call and may-vary at another:
 * a call substitutes each parameter by the value of its argument. A helper
 * parameter MAY VARY when some call passes an argument that may vary
 * (context-insensitive). Assignments are joins (flow-insensitive per
 * variable) and carry the control dependence of the conditions between the
 * variable's declaration and the assignment (C-1). If/match values join their
 * condition (C-2), declared intrinsic variance is honoured (C-4), private
 * array reads may vary (C-5), helpers have result summaries (C-3).
 ******************************************************************************)

open Sarek_variance_ir
module IS = Set.Make (Int)

type value = Varying | Deps of IS.t

let uniform = Deps IS.empty

let join a b =
  match (a, b) with
  | Varying, _ | _, Varying -> Varying
  | Deps x, Deps y -> Deps (IS.union x y)

let join_all = List.fold_left join uniform

let leq a b =
  match (a, b) with
  | _, Varying -> true
  | Varying, Deps _ -> false
  | Deps x, Deps y -> IS.subset x y

type state = {
  prog : program;
  vals : value array;  (** per key *)
  priv : bool array;  (** key may denote a private (per-thread) array *)
  vparam : bool array;  (** helper parameter that some call makes varying *)
  results : (fkey, value) Hashtbl.t;
  call_ctl : (fkey, value) Hashtbl.t;  (** join of the control at call sites *)
  decl : (int * int) array;  (** key -> (frame, depth) of its declaration *)
  mutable changed : bool;
  mutable updating : bool;
  mutable next_frame : int;
}

let may_vary st = function
  | Varying -> true
  | Deps ps -> IS.exists (fun p -> st.vparam.(p)) ps

(* Control: a stack of condition values (innermost first) inside a frame. *)
type ctl = {frame : int; conds : value list}

let ctl_join c = join_all c.conds

let push c v = {c with conds = v :: c.conds}

let set_val st k v =
  if st.updating && not (leq v st.vals.(k)) then begin
    st.vals.(k) <- join v st.vals.(k) ;
    st.changed <- true
  end

let set_flag st arr k =
  if st.updating && not arr.(k) then begin
    arr.(k) <- true ;
    st.changed <- true
  end

let set_tbl st tbl f v =
  let old = Option.value ~default:uniform (Hashtbl.find_opt tbl f) in
  if st.updating && not (leq v old) then begin
    Hashtbl.replace tbl f (join v old) ;
    st.changed <- true
  end

let declare st ctl k =
  if st.updating then st.decl.(k) <- (ctl.frame, List.length ctl.conds)

(** Control between [k]'s declaration and here (all of it across frames). *)
let ctl_since st ctl k =
  let frame, depth = st.decl.(k) in
  if frame <> ctl.frame then ctl_join ctl
  else
    let n = List.length ctl.conds - depth in
    join_all (List.filteri (fun i _ -> i < n) ctl.conds)

(** May [n] denote a private array? Anything not traced is assumed so. *)
let may_private st (n : node) =
  match n.d with
  | Var k -> st.priv.(k)
  | New_array (p, _) -> p
  | Lit | Fun_ref _ -> false
  | _ -> true

let subst (f : fn) (args : value list) (v : value) =
  match v with
  | Varying -> Varying
  | Deps ps ->
      IS.fold
        (fun p acc ->
          let rec find params args =
            match (params, args) with
            | (k, _) :: _, a :: _ when k = p -> Some a
            | _ :: ps, _ :: as_ -> find ps as_
            | _ -> None
          in
          match find f.fparams args with
          | Some a -> join a acc
          | None -> join (Deps (IS.singleton p)) acc)
        ps
        uniform

let rec eval st ctl (n : node) : value =
  match n.d with
  | Lit | Fun_ref _ -> uniform
  | Src | Native -> Varying
  | Var k -> st.vals.(k)
  | Join ns -> join_all (List.map (eval st ctl) ns)
  | Store ns ->
      List.iter (fun n -> ignore (eval st ctl n)) ns ;
      uniform
  | Short (a, b) ->
      let va = eval st ctl a in
      join va (eval st (push ctl va) b)
  | Prim_call (varies, _, args) ->
      let v = join_all (List.map (eval st ctl) args) in
      if varies then Varying else v
  | New_array (_, size) ->
      ignore (eval st ctl size) ;
      uniform
  | Arr_get (a, i) ->
      let va = eval st ctl a and vi = eval st ctl i in
      if may_private st a then Varying else join va vi
  | If (c, t, e) ->
      let vc = eval st ctl c in
      let inner = push ctl vc in
      let ve = match e with Some e -> eval st inner e | None -> uniform in
      join vc (join (eval st inner t) ve)
  | Match (s, cases) -> eval_match st ctl s cases
  | While (c, b) ->
      let vc = eval st ctl c in
      join vc (eval st (push ctl vc) b)
  | For (k, lo, hi, b) ->
      let vb = join (eval st ctl lo) (eval st ctl hi) in
      declare st ctl k ;
      set_val st k vb ;
      join vb (eval st (push ctl vb) b)
  | Seq ns -> join_all (List.map (eval st ctl) ns)
  | Let (k, kind, v, b) -> eval_let st ctl k kind v b
  | Assign (k, kind, e) ->
      let ve = eval st ctl e in
      set_val st k (join ve (ctl_since st ctl k)) ;
      if kind = Array_value && may_private st e then set_flag st st.priv k ;
      uniform
  | Call (sel, callees, args) -> eval_call st ctl sel callees args
  | Let_rec (_, cont) -> eval st ctl cont
  | Superstep (b, k) -> join (eval st ctl b) (eval st ctl k)
  | Return e -> eval st ctl e

and eval_match st ctl s cases =
  let vs = eval st ctl s in
  let inner = push ctl vs in
  List.fold_left
    (fun acc (keys, body) ->
      List.iter
        (fun k ->
          declare st inner k ;
          set_val st k vs)
        keys ;
      join acc (eval st inner body))
    vs
    cases

and eval_let st ctl k kind v b =
  let vv = eval st ctl v in
  declare st ctl k ;
  set_val st k vv ;
  if kind = Array_value && may_private st v then set_flag st st.priv k ;
  eval st ctl b

and eval_call st ctl sel callees args =
  let vsel = eval st ctl sel in
  let vargs = List.map (eval st ctl) args in
  let one = function
    | Fun f -> (
        match Hashtbl.find_opt st.prog.funs f with
        | None -> Varying
        | Some fn ->
            note_call st ctl f fn args vargs ;
            let r =
              Option.value ~default:uniform (Hashtbl.find_opt st.results f)
            in
            subst fn vargs r)
    | Prim name -> if prim_varies name then Varying else join_all vargs
    | Unknown -> Varying
  in
  join vsel (join_all (List.map one callees))

and note_call st ctl f (fn : fn) args vargs =
  set_tbl st st.call_ctl f (join (ctl_join ctl) (Deps IS.empty)) ;
  let rec go params args vargs =
    match (params, args, vargs) with
    | (k, kind) :: ps, a :: as_, va :: vs ->
        if may_vary st va then set_flag st st.vparam k ;
        if kind = Array_value && may_private st a then set_flag st st.priv k ;
        go ps as_ vs
    | _ -> ()
  in
  go fn.fparams args vargs

let new_frame st conds =
  let f = st.next_frame in
  st.next_frame <- f + 1 ;
  {frame = f; conds}

let analyse_fun st f (fn : fn) =
  List.iter (fun (k, _) -> st.decl.(k) <- (-1, 0)) fn.fparams ;
  let start =
    if fn.flocal then
      [Option.value ~default:uniform (Hashtbl.find_opt st.call_ctl f)]
    else []
  in
  let r = eval st (new_frame st start) fn.fbody in
  set_tbl st st.results f r

(** Run the fixpoint. [seed] keys are may-vary sources. *)
let analyse ?(seed = []) (prog : program) : state =
  let n = max 1 prog.nkeys in
  let st =
    {
      prog;
      vals = Array.make n uniform;
      priv = Array.make n false;
      vparam = Array.make n false;
      results = Hashtbl.create 8;
      call_ctl = Hashtbl.create 8;
      decl = Array.make n (-1, 0);
      changed = true;
      updating = true;
      next_frame = 0;
    }
  in
  List.iter (fun k -> st.vals.(k) <- Deps (IS.singleton k)) prog.params ;
  List.iter (fun k -> st.vals.(k) <- Varying) seed ;
  while st.changed do
    st.changed <- false ;
    st.next_frame <- 0 ;
    Hashtbl.iter (analyse_fun st) prog.funs ;
    ignore (eval st (new_frame st []) prog.body)
  done ;
  st.updating <- false ;
  st

(** Value of a node once the fixpoint is reached (no state change). *)
let value_of st n = eval st {frame = -2; conds = []} n

let node_may_vary st n = may_vary st (value_of st n)
