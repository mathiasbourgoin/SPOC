(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Horizontal fusion of independent kernels (specs/launch-window.md, FR-011).

    Members [M_0 .. M_{n-1}] with the same block shape become one kernel whose
    grid x is the sum of theirs. Work-group [b] with [off_k <= b < off_k + gx_k]
    runs member [k]'s body unchanged, except that the work-group position it
    observes is rewritten back to its own: [block_idx_x] reads [b - off_k],
    [grid_dim_x] reads [gx_k], and the global index and size intrinsics follow.
    No value flows between members and no arithmetic is added to a member's data
    path, so the result is bit-identical to running the members one after the
    other, on every backend.

    The dispatch condition depends only on [block_idx_x], so it is uniform per
    work-group: barriers inside a member stay legal.

    Every name of member [k] (parameters, locals, shared arrays, loop and
    pattern variables) is prefixed [m<k>_] and its var ids are offset, so
    members cannot capture each other's names. The fused kernel's parameters are
    the members' parameters in order, followed by [hf<k>_off] and [hf<k>_gdx]
    (int32) for each member, a namespace no renamed member name can enter. *)

open Sarek_ir_types

let prefix k name = Printf.sprintf "m%d_%s" k name

(* Var ids of member [k] are shifted into a range of their own. *)
let id_stride = 1 lsl 24

let rename_var k (v : var) =
  {
    v with
    var_name = prefix k v.var_name;
    var_id = v.var_id + ((k + 1) * id_stride);
  }

(* The position parameters live OUTSIDE the members' [m<k>_] namespace: a
   member variable named [off] becomes [m<k>_off] and must not shadow them. *)
let int32_var k name =
  {
    var_name = Printf.sprintf "hf%d_%s" k name;
    var_id = (((k + 1) * id_stride) - 1 - if name = "off" then 0 else 1);
    var_type = TInt32;
    var_mutable = false;
  }

let gpu name = EIntrinsic (["Sarek_stdlib"; "Gpu"], name, [])

(** {1 Renaming and position rewrite} *)

type ctx = {k : int; off : expr; gdx : expr}

(* block_idx_x as the member saw it. *)
let member_block ctx = EBinop (Sub, gpu "block_idx_x", ctx.off)

let position ctx name =
  match name with
  | "block_idx_x" -> Some (member_block ctx)
  | "grid_dim_x" -> Some ctx.gdx
  | "global_thread_id" | "global_idx" | "global_idx_x" ->
      (* Same association as the backends' own expansion: tid + bid * bdim. *)
      Some
        (EBinop
           ( Add,
             gpu "thread_idx_x",
             EBinop (Mul, member_block ctx, gpu "block_dim_x") ))
  | "global_size" | "global_size_x" ->
      Some (EBinop (Mul, ctx.gdx, gpu "block_dim_x"))
  | _ -> None

let rec expr ctx e =
  let go = expr ctx and n = prefix ctx.k in
  match e with
  | EConst _ -> e
  | EVar v -> EVar (rename_var ctx.k v)
  | EBinop (op, a, b) -> EBinop (op, go a, go b)
  | EUnop (op, a) -> EUnop (op, go a)
  | EArrayRead (a, i) -> EArrayRead (n a, go i)
  | EArrayReadExpr (a, i) -> EArrayReadExpr (go a, go i)
  | ERecordField (a, f) -> ERecordField (go a, f)
  | EIntrinsic ((["Sarek_stdlib"; "Gpu"] as p), name, args) -> (
      match position ctx name with
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
  | PConstr (c, vs) -> PConstr (c, List.map (prefix ctx.k) vs)
  | PWild -> PWild

let rec lvalue ctx = function
  | LVar v -> LVar (rename_var ctx.k v)
  | LArrayElem (a, i) -> LArrayElem (prefix ctx.k a, expr ctx i)
  | LArrayElemExpr (a, i) -> LArrayElemExpr (expr ctx a, expr ctx i)
  | LRecordField (lv, f) -> LRecordField (lvalue ctx lv, f)

let rec stmt ctx s =
  let e = expr ctx and st = stmt ctx and v = rename_var ctx.k in
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
  | DParam (v, info) -> DParam (rename_var ctx.k v, info)
  | DLocal (v, init) -> DLocal (rename_var ctx.k v, Option.map (expr ctx) init)
  | DShared (name, t, size) ->
      DShared (prefix ctx.k name, t, Option.map (expr ctx) size)

(** {1 Assembly} *)

exception Incompatible of string

(* Union of named definitions; the same name must carry the same definition. *)
let union_named ~what name_of members =
  List.fold_left
    (fun acc d ->
      match List.find_opt (fun d' -> name_of d' = name_of d) acc with
      | None -> acc @ [d]
      | Some d' ->
          let same = try d = d' with Invalid_argument _ -> false in
          if same then acc
          else
            raise
              (Incompatible
                 (Printf.sprintf "two different %s named %s" what (name_of d))))
    []
    (List.concat members)

type member_parts = {
  binds : (var * expr) list;  (** hoisted shared arrays, renamed *)
  body : stmt;  (** renamed and rewritten *)
  params : decl list;
  locals : decl list;
  off_var : var;
  gdx_var : var;
}

let member_parts k (m : kernel) =
  let off_var = int32_var k "off" and gdx_var = int32_var k "gdx" in
  let ctx = {k; off = EVar off_var; gdx = EVar gdx_var} in
  let binds, body = Kernel_access.shared_prefix m.kern_body in
  {
    binds = List.map (fun (x, y) -> (rename_var k x, expr ctx y)) binds;
    body = stmt ctx body;
    params = List.map (decl ctx) m.kern_params;
    locals = List.map (decl ctx) m.kern_locals;
    off_var;
    gdx_var;
  }

(* if bx < off_0 + gdx_0 then body_0 else if bx < off_1 + gdx_1 then ... else
   body_{n-1}. The last member needs no test: the grid ends with it. *)
let rec dispatch = function
  | [] -> SEmpty
  | [p] -> p.body
  | p :: rest ->
      let bound = EBinop (Add, EVar p.off_var, EVar p.gdx_var) in
      SIf (EBinop (Lt, gpu "block_idx_x", bound), p.body, Some (dispatch rest))

let scalar_param v = DParam (v, None)

(** Fuse [members] (at least two, each with [Kernel_access.horizontal_verdict] =
    [Ok]) into one kernel named [name]. Raises [Incompatible] when two members
    define the same type or helper name differently. *)
let fuse ~(name : string) (members : kernel list) : kernel =
  let parts = List.mapi member_parts members in
  let body =
    List.fold_right
      (fun (x, y) acc -> SLet (x, y, acc))
      (List.concat_map (fun p -> p.binds) parts)
      (dispatch parts)
  in
  let position_params =
    List.concat_map
      (fun p -> [scalar_param p.off_var; scalar_param p.gdx_var])
      parts
  in
  {
    kern_name = name;
    kern_params = List.concat_map (fun p -> p.params) parts @ position_params;
    kern_locals = List.concat_map (fun p -> p.locals) parts;
    kern_body = body;
    kern_types =
      union_named
        ~what:"record types"
        fst
        (List.map (fun m -> m.kern_types) members);
    kern_variants =
      union_named
        ~what:"variant types"
        fst
        (List.map (fun m -> m.kern_variants) members);
    kern_funcs =
      union_named
        ~what:"helper functions"
        (fun h -> h.hf_name)
        (List.map (fun m -> m.kern_funcs) members);
    kern_native_fn = None;
  }
