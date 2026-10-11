(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Renaming of a kernel's names, shared by the fusion rewrites
    (specs/launch-window.md, specs/epilogue-fusion.md).

    Every variable, array and pattern name goes through [name], every var id
    through [id], and every argument-less [Sarek_stdlib.Gpu] intrinsic through
    [position] (which may replace it). Helper functions are called by their own
    names, which are left alone. Record fields and constructors are not names of
    the kernel and are left alone too. *)

open Sarek_ir_types

type ctx = {
  name : string -> string;
  id : int -> int;
  position : string -> expr option;
}

let var ctx (v : var) =
  {v with var_name = ctx.name v.var_name; var_id = ctx.id v.var_id}

let rec expr ctx e =
  let go = expr ctx and n = ctx.name in
  match e with
  | EConst _ -> e
  | EVar v -> EVar (var ctx v)
  | EBinop (op, a, b) -> EBinop (op, go a, go b)
  | EUnop (op, a) -> EUnop (op, go a)
  | EArrayRead (a, i) -> EArrayRead (n a, go i)
  | EArrayReadExpr (a, i) -> EArrayReadExpr (go a, go i)
  | ERecordField (a, f) -> ERecordField (go a, f)
  | EIntrinsic ((["Sarek_stdlib"; "Gpu"] as p), name, args) -> (
      match ctx.position name with
      | Some r when args = [] -> r
      | _ -> EIntrinsic (p, name, List.map go args))
  | EIntrinsic (p, name, args) -> EIntrinsic (p, name, List.map go args)
  | ECast (t, a) -> ECast (t, go a)
  | ETuple es -> ETuple (List.map go es)
  | EApp ((EVar _ as helper), es) ->
      (* A helper function, called by its own (shared, unrenamed) name. *)
      EApp (helper, List.map go es)
  | EApp (f, es) -> EApp (go f, List.map go es)
  | ERecord (t, fs) -> ERecord (t, List.map (fun (f, x) -> (f, go x)) fs)
  | EVariant (t, c, es) -> EVariant (t, c, List.map go es)
  | EArrayLen a -> EArrayLen (n a)
  | EArrayCreate (t, s, m) -> EArrayCreate (t, go s, m)
  | EIf (c, a, b) -> EIf (go c, go a, go b)
  | EMatch (s, cs) ->
      EMatch (go s, List.map (fun (p, x) -> (pattern ctx p, go x)) cs)

and pattern ctx = function
  | PConstr (c, vs) -> PConstr (c, List.map ctx.name vs)
  | PWild -> PWild

let rec lvalue ctx = function
  | LVar v -> LVar (var ctx v)
  | LArrayElem (a, i) -> LArrayElem (ctx.name a, expr ctx i)
  | LArrayElemExpr (a, i) -> LArrayElemExpr (expr ctx a, expr ctx i)
  | LRecordField (lv, f) -> LRecordField (lvalue ctx lv, f)

let rec stmt ctx s =
  let e = expr ctx and st = stmt ctx and v = var ctx in
  match s with
  | SAssign (lv, x) -> SAssign (lvalue ctx lv, e x)
  | SSeq ss -> SSeq (List.map st ss)
  | SIf (c, t, f) -> SIf (e c, st t, Option.map st f)
  | SWhile (c, b) -> SWhile (e c, st b)
  | SFor (i, lo, hi, d, b) -> SFor (v i, e lo, e hi, d, st b)
  | SMatch (x, cs) ->
      SMatch (e x, List.map (fun (p, b) -> (pattern ctx p, st b)) cs)
  | SReturn x -> SReturn (e x)
  | SExpr x -> SExpr (e x)
  | SLet (x, y, b) -> SLet (v x, e y, st b)
  | SLetMut (x, y, b) -> SLetMut (v x, e y, st b)
  | SPragma (p, b) -> SPragma (p, st b)
  | SBlock b -> SBlock (st b)
  | SBarrier | SWarpBarrier | SMemFence | SEmpty | SNative _ | SCoopmat _ -> s

let decl ctx = function
  | DParam (v, info) -> DParam (var ctx v, info)
  | DLocal (v, init) -> DLocal (var ctx v, Option.map (expr ctx) init)
  | DShared (name, t, size) ->
      DShared (ctx.name name, t, Option.map (expr ctx) size)
