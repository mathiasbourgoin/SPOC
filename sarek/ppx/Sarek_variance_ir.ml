(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - analysis IR for the convergence check
 *
 * The typed AST is converted once into a small IR that keeps only what the
 * variance analysis (Sarek_variance) and the convergence check
 * (Sarek_convergence) need, with every name resolved LEXICALLY to a unique
 * key. Typer variable ids are not used: parameters of a local [let rec] all
 * carry id 0 in [TEVar], so ids do not identify binders.
 *
 * The IR is close to the abstract [expr] of
 * formal/convergence-safety/theories/ConvergenceSpec.v; the correspondence is
 * documented in specs/convergence-semantics.md §3 and in that project's
 * ASSUMPTIONS.md.
 ******************************************************************************)

open Sarek_typed_ast

type key = int

type fkey = int

(** A synchronisation point: block barrier or warp collective (by name). *)
type sync = Barrier | Warp of string

(** What an array-typed binder may denote. *)
type arr_kind = Not_array | Array_value | Shared_array

type callee =
  | Fun of fkey  (** a kernel-local or module helper *)
  | Prim of string  (** an intrinsic used as a function value *)
  | Unknown  (** an expression the conversion cannot resolve *)

type node = {d : desc; loc : Sarek_ast.loc}

and desc =
  | Lit  (** uniform leaf *)
  | Src  (** a may-vary source: thread ids, varying intrinsics *)
  | Native  (** inline native code: may vary, not inspected *)
  | Var of key
  | Fun_ref of fkey  (** a helper used as a value (uniform) *)
  | Join of node list  (** pure operation: value is the join of the parts *)
  | Short of node * node
      (** [a && b] / [a || b]: [b] runs only if [a] allows *)
  | Prim_call of bool * sync option * node list
      (** declared may-vary, synchronisation, arguments *)
  | New_array of bool * node  (** private?, size *)
  | Arr_get of node * node
  | If of node * node * node option
  | Match of node * (key list * node) list
  | While of node * node
  | For of key * node * node * node
  | Seq of node list
  | Let of key * arr_kind * node * node
  | Assign of key * arr_kind * node
  | Store of node list  (** memory write: children run, value is unit *)
  | Call of node * callee list * node list  (** selector, callees, args *)
  | Let_rec of fkey * node  (** definition (body in [funs]), continuation *)
  | Superstep of node * node
  | Return of node

type fn = {fparams : (key * arr_kind) list; fbody : node; flocal : bool}

type program = {
  funs : (fkey, fn) Hashtbl.t;
  body : node;
  nkeys : int;  (** keys are [0 .. nkeys - 1] *)
  params : key list;  (** helper parameter keys (all functions) *)
}

(* ------------------------------------------------------------------------- *)
(* Intrinsic classification                                                   *)
(* ------------------------------------------------------------------------- *)

let base_name = function
  | Sarek_env.IntrinsicRef (_, n) -> n
  | Sarek_env.CorePrimitiveRef n -> n

(** May-vary at block scope: declared WarpVarying or ThreadVarying. *)
let prim_varies name = Sarek_core_primitives.is_warp_varying name

let prim_sync name =
  if Sarek_core_primitives.is_convergence_point name then Some Barrier
  else if Sarek_core_primitives.is_warp_convergence_point name then
    Some (Warp name)
  else None

let sync_of_conv ref = function
  | Some Sarek_core_primitives.ConvergencePoint -> Some Barrier
  | Some Sarek_core_primitives.WarpConvergence ->
      Some (Warp (Sarek_env.intrinsic_ref_display_name ref))
  | Some Sarek_core_primitives.NoEffect | None -> None

let arr_kind_of_typ (ty : Sarek_types.typ) =
  match Sarek_types.repr ty with
  | Sarek_types.TArr _ -> Array_value
  | _ -> Not_array

let is_fun_typ (ty : Sarek_types.typ) =
  match Sarek_types.repr ty with Sarek_types.TFun _ -> true | _ -> false

(* ------------------------------------------------------------------------- *)
(* Conversion with lexical scope                                              *)
(* ------------------------------------------------------------------------- *)

module SM = Map.Make (String)

type binding = B_var of key | B_fun of fkey

type st = {
  mutable next_key : int;
  mutable next_fkey : int;
  funs_tbl : (fkey, fn) Hashtbl.t;
  aliases : (key, callee list) Hashtbl.t;
      (** function-typed [let] binders: the callees they may denote *)
  mutable param_keys : key list;
}

let fresh_key st =
  let k = st.next_key in
  st.next_key <- k + 1 ;
  k

let fresh_fkey st =
  let f = st.next_fkey in
  st.next_fkey <- f + 1 ;
  f

let mk d loc = {d; loc}

(** Leaf for an unbound name: an intrinsic constant or a free variable. *)
let free_name_node name loc =
  if prim_varies name then mk Src loc else mk Lit loc

let rec callees_of st scope (te : texpr) : callee list =
  match te.te with
  | TEVar (name, _) -> (
      match SM.find_opt name scope with
      | Some (B_fun f) -> [Fun f]
      | Some (B_var k) -> (
          match Hashtbl.find_opt st.aliases k with
          | Some cs -> cs
          | None -> [Unknown])
      | None -> [Prim name])
  | TEIf (_, t, e) -> (
      callees_of st scope t
      @ match e with Some e -> callees_of st scope e | None -> [])
  | TEMatch (_, cases) ->
      List.concat_map (fun (_, b) -> callees_of st scope b) cases
  | TELet (_, _, _, b) | TELetMut (_, _, _, b) | TEOpen (_, b) ->
      (* A binder inside the selector is out of scope here; the leaves we
         need are named functions, which do not depend on it. *)
      callees_of st scope b
  | TESeq es -> (
      match List.rev es with
      | [] -> [Unknown]
      | last :: _ -> callees_of st scope last)
  | _ -> [Unknown]

let rec pattern_vars (p : tpattern) acc =
  match p.tpat with
  | TPAny -> acc
  | TPVar (n, _) -> n :: acc
  | TPConstr (_, _, None) -> acc
  | TPConstr (_, _, Some p) -> pattern_vars p acc
  | TPTuple ps -> List.fold_left (fun acc p -> pattern_vars p acc) acc ps

let bind_var st scope name =
  let k = fresh_key st in
  (k, SM.add name (B_var k) scope)

let rec conv st scope (te : texpr) : node =
  let loc = te.te_loc in
  let c = conv st scope in
  match te.te with
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEGlobalRef _ ->
      mk Lit loc
  | TENative _ -> mk Native loc
  | TEVar (name, _) -> conv_var scope name loc
  | TEIntrinsicConst ref ->
      if prim_varies (base_name ref) then mk Src loc else mk Lit loc
  | TEIntrinsicFun (ref, conv_kind, args) ->
      mk
        (Prim_call
           ( prim_varies (base_name ref),
             sync_of_conv ref conv_kind,
             List.map c args ))
        loc
  | TEBinop ((Sarek_ast.And | Sarek_ast.Or), a, b) -> mk (Short (c a, c b)) loc
  | TEBinop (_, a, b) | TEVecGet (a, b) -> mk (Join [c a; c b]) loc
  | TEUnop (_, e) | TEFieldGet (e, _, _) -> mk (Join [c e]) loc
  | TETuple es -> mk (Join (List.map c es)) loc
  | TERecord (_, fields) -> mk (Join (List.map (fun (_, e) -> c e) fields)) loc
  | TEConstr (_, _, arg) -> mk (Join (Option.to_list (Option.map c arg))) loc
  | TEArrGet (a, i) -> mk (Arr_get (c a, c i)) loc
  | TEVecSet (v, i, x) | TEArrSet (v, i, x) -> mk (Store [c v; c i; c x]) loc
  | TEFieldSet (r, _, _, x) -> conv_field_set st scope r x loc
  | TEAssign (name, _, e) -> conv_assign st scope name e loc
  | TECreateArray (size, _, mem) ->
      mk (New_array (mem <> Sarek_types.Shared, c size)) loc
  | TEIf (cond, t, e) -> mk (If (c cond, c t, Option.map c e)) loc
  | TEMatch (scrut, cases) ->
      mk (Match (c scrut, List.map (conv_case st scope) cases)) loc
  | TEWhile (cond, b) -> mk (While (c cond, c b)) loc
  | TEFor (name, _, lo, hi, _, b) ->
      let k, scope' = bind_var st scope name in
      mk (For (k, c lo, c hi, conv st scope' b)) loc
  | TESeq es -> mk (Seq (List.map c es)) loc
  | TELet (name, _, v, b) | TELetMut (name, _, v, b) ->
      conv_let st scope name v b loc
  | TELetShared (name, _, _, size, b) -> conv_shared st scope name size b loc
  | TELetRec (name, _, params, fn_body, cont) ->
      conv_let_rec st scope name params fn_body cont loc
  | TEApp (f, args) ->
      mk (Call (c f, callees_of st scope f, List.map c args)) loc
  | TESuperstep (_, _, b, k) -> mk (Superstep (c b, c k)) loc
  | TEReturn e -> mk (Return (c e)) loc
  | TEPragma (_, b) | TEOpen (_, b) -> c b

and conv_assign st scope name (e : texpr) loc =
  match SM.find_opt name scope with
  | Some (B_var k) -> mk (Assign (k, arr_kind_of_typ e.ty, conv st scope e)) loc
  | Some (B_fun _) | None -> mk (Store [conv st scope e]) loc

and conv_shared st scope name size b loc =
  let size_node =
    match size with Some s -> conv st scope s | None -> mk Lit loc
  in
  let k, scope' = bind_var st scope name in
  mk (Let (k, Shared_array, size_node, conv st scope' b)) loc

and conv_var scope name loc =
  match SM.find_opt name scope with
  | Some (B_var k) -> mk (Var k) loc
  | Some (B_fun f) -> mk (Fun_ref f) loc
  | None -> free_name_node name loc

and conv_field_set st scope (r : texpr) x loc =
  let c = conv st scope in
  match r.te with
  | TEVar (name, _) -> (
      match SM.find_opt name scope with
      | Some (B_var k) -> mk (Assign (k, Not_array, c x)) loc
      | _ -> mk (Store [c r; c x]) loc)
  | _ -> mk (Store [c r; c x]) loc

and conv_case st scope ((p : tpattern), body) =
  let names = pattern_vars p [] in
  let keys, scope' =
    List.fold_left
      (fun (ks, sc) n ->
        let k, sc' = bind_var st sc n in
        (k :: ks, sc'))
      ([], scope)
      names
  in
  (keys, conv st scope' body)

and conv_let st scope name (v : texpr) b loc =
  let vnode = conv st scope v in
  let k, scope' = bind_var st scope name in
  if is_fun_typ v.ty then Hashtbl.replace st.aliases k (callees_of st scope v) ;
  mk (Let (k, arr_kind_of_typ v.ty, vnode, conv st scope' b)) loc

and conv_let_rec st scope name params fn_body cont loc =
  let f = fresh_fkey st in
  let scope_fn = SM.add name (B_fun f) scope in
  let fparams, scope_body = bind_params st scope_fn params in
  let fbody = conv st scope_body fn_body in
  Hashtbl.replace st.funs_tbl f {fparams; fbody; flocal = true} ;
  mk (Let_rec (f, conv st scope_fn cont)) loc

and bind_params st scope (params : tparam list) =
  let fparams, scope' =
    List.fold_left
      (fun (acc, sc) (p : tparam) ->
        let k, sc' = bind_var st sc p.tparam_name in
        st.param_keys <- k :: st.param_keys ;
        ((k, arr_kind_of_typ p.tparam_type) :: acc, sc'))
      ([], scope)
      params
  in
  (List.rev fparams, scope')

(* ------------------------------------------------------------------------- *)
(* Entry points                                                               *)
(* ------------------------------------------------------------------------- *)

let new_st () =
  {
    next_key = 0;
    next_fkey = 0;
    funs_tbl = Hashtbl.create 8;
    aliases = Hashtbl.create 8;
    param_keys = [];
  }

let finish st body =
  {funs = st.funs_tbl; body; nkeys = st.next_key; params = st.param_keys}

(** Module items become nested scopes: a constant is a [let], a function a
    [let rec] (its own name is in scope in its body only if recursive). *)
let conv_items st scope items =
  List.fold_left
    (fun (scope, wrap) (item : tmodule_item) ->
      match item with
      | TMConst (name, _, _, e) ->
          let vnode = conv st scope e in
          let k, scope' = bind_var st scope name in
          let w body =
            wrap (mk (Let (k, arr_kind_of_typ e.ty, vnode, body)) e.te_loc)
          in
          (scope', w)
      | TMFun (name, is_rec, params, body) ->
          let f = fresh_fkey st in
          let scope_fn = SM.add name (B_fun f) scope in
          let fparams, scope_body =
            bind_params st (if is_rec then scope_fn else scope) params
          in
          let fbody = conv st scope_body body in
          Hashtbl.replace st.funs_tbl f {fparams; fbody; flocal = false} ;
          (scope_fn, wrap))
    (scope, fun body -> body)
    items

(** Convert a kernel. Kernel parameters are uniform binders. *)
let of_kernel (k : tkernel) : program =
  let st = new_st () in
  let scope =
    List.fold_left
      (fun sc (p : tparam) -> snd (bind_var st sc p.tparam_name))
      SM.empty
      k.tkern_params
  in
  let scope, wrap = conv_items st scope k.tkern_module_items in
  finish st (wrap (conv st scope k.tkern_body))

(** Convert a standalone expression; free names are intrinsics or uniform,
    except those in [varying], which are treated as may-vary sources. *)
let of_expr ?(varying = []) (te : texpr) : program * key list =
  let st = new_st () in
  let keys, scope =
    List.fold_left
      (fun (ks, sc) n ->
        let k, sc' = bind_var st sc n in
        (k :: ks, sc'))
      ([], SM.empty)
      varying
  in
  (finish st (conv st scope te), keys)

(** Sub-nodes evaluated when [n] runs (a [Let_rec]'s body is not: it runs at its
    calls). *)
let children (n : node) : node list =
  match n.d with
  | Lit | Src | Native | Var _ | Fun_ref _ -> []
  | Join ns | Store ns | Seq ns | Prim_call (_, _, ns) -> ns
  | Short (a, b) | Arr_get (a, b) | While (a, b) | Superstep (a, b) -> [a; b]
  | New_array (_, n) | Return n | Let_rec (_, n) | Assign (_, _, n) -> [n]
  | If (c, t, e) -> c :: t :: Option.to_list e
  | Match (s, cases) -> s :: List.map snd cases
  | For (_, lo, hi, b) -> [lo; hi; b]
  | Let (_, _, v, b) -> [v; b]
  | Call (sel, _, args) -> sel :: args
