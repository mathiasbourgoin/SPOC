(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module lowers the typed AST directly to Sarek_ir_ppx, bypassing
 * the legacy Kirc_Ast. This provides a cleaner IR for the V2 execution path.
 ******************************************************************************)

open Sarek_ast
open Sarek_types
open Sarek_typed_ast
module Ir = Sarek_ir_ppx

(** Convert Sarek_types.typ to Sarek_ir_ppx.elttype *)
let rec elttype_of_typ (ty : typ) : Ir.elttype =
  match repr ty with
  | TPrim TInt32 -> Ir.TInt32
  | TPrim TBool -> Ir.TBool
  | TPrim TUnit -> Ir.TUnit
  | TReg "int64" -> Ir.TInt64
  | TReg "float32" -> Ir.TFloat32
  | TReg "float64" -> Ir.TFloat64
  | TVec elem_ty -> Ir.TVec (elttype_of_typ elem_ty)
  | TArr (elem_ty, mem) ->
      let ir_mem =
        match mem with
        | Sarek_types.Global -> Ir.Global
        | Sarek_types.Shared -> Ir.Shared
        | Sarek_types.Local -> Ir.Local
      in
      Ir.TArray (elttype_of_typ elem_ty, ir_mem)
  | TRecord (name, fields) ->
      Ir.TRecord (name, List.map (fun (n, ty) -> (n, elttype_of_typ ty)) fields)
  | TVariant (name, constrs) ->
      Ir.TVariant
        ( name,
          List.map
            (fun (n, ty_opt) ->
              ( n,
                match ty_opt with None -> [] | Some ty -> [elttype_of_typ ty] ))
            constrs )
  | TTuple _ -> Ir.TInt32 (* tuples become int for now *)
  | TFun _ -> Ir.TInt32 (* functions become int *)
  | TVar _ -> Ir.TInt32 (* unresolved should not happen *)
  | TReg _ -> Ir.TInt32 (* other registered types *)

(** Convert Sarek_ast.binop to Sarek_ir_ppx.binop *)
let ir_binop (op : binop) (_ty : typ) : Ir.binop =
  match op with
  | Add -> Ir.Add
  | Sub -> Ir.Sub
  | Mul -> Ir.Mul
  | Div -> Ir.Div
  | Mod -> Ir.Mod
  | Eq -> Ir.Eq
  | Ne -> Ir.Ne
  | Lt -> Ir.Lt
  | Le -> Ir.Le
  | Gt -> Ir.Gt
  | Ge -> Ir.Ge
  | And -> Ir.And
  | Or -> Ir.Or
  | Lsl -> Ir.Shl
  | Lsr -> Ir.Shr
  | Asr -> Ir.Shr (* arithmetic shift right maps to Shr *)
  | Land -> Ir.BitAnd
  | Lor -> Ir.BitOr
  | Lxor -> Ir.BitXor

(** Convert Sarek_ast.unop to Sarek_ir_ppx.unop *)
let ir_unop (op : unop) : Ir.unop =
  match op with Neg -> Ir.Neg | Not -> Ir.Not | Lnot -> Ir.BitNot

(** Create a var from typed var info *)
let make_var name id ty mutable_ : Ir.var =
  {
    var_name = name;
    var_id = id;
    var_type = elttype_of_typ ty;
    var_mutable = mutable_;
  }

(** Convert a typed expression to IR expression *)
let rec lower_expr (te : texpr) : Ir.expr =
  match te.te with
  | TEUnit -> Ir.EConst Ir.CUnit
  | TEBool b -> Ir.EConst (Ir.CBool b)
  | TEInt i -> Ir.EConst (Ir.CInt32 (Int32.of_int i))
  | TEInt32 i -> Ir.EConst (Ir.CInt32 i)
  | TEInt64 i -> Ir.EConst (Ir.CInt64 i)
  | TEFloat f -> Ir.EConst (Ir.CFloat32 f)
  | TEDouble f -> Ir.EConst (Ir.CFloat64 f)
  | TEVar (name, id) ->
      let v = make_var name id te.ty false in
      Ir.EVar v
  | TEVecGet (vec, idx) -> (
      match vec.te with
      | TEVar (name, _) -> Ir.EArrayRead (name, lower_expr idx)
      | _ -> failwith "lower_expr: VecGet on non-variable")
  | TEArrGet (arr, idx) -> (
      match arr.te with
      | TEVar (name, _) -> Ir.EArrayRead (name, lower_expr idx)
      | _ -> failwith "lower_expr: ArrGet on non-variable")
  | TEFieldGet (r, field, _) -> Ir.ERecordField (lower_expr r, field)
  | TEBinop (op, a, b) ->
      Ir.EBinop (ir_binop op te.ty, lower_expr a, lower_expr b)
  | TEUnop (op, a) -> Ir.EUnop (ir_unop op, lower_expr a)
  | TEApp (f, args) -> Ir.EApp (lower_expr f, List.map lower_expr args)
  | TERecord (name, fields) ->
      Ir.ERecord (name, List.map (fun (n, e) -> (n, lower_expr e)) fields)
  | TEConstr (ty_name, constr, arg) ->
      let args = match arg with None -> [] | Some e -> [lower_expr e] in
      Ir.EVariant (ty_name, constr, args)
  | TETuple exprs -> Ir.ETuple (List.map lower_expr exprs)
  | TEGlobalRef (name, ty) ->
      let v = make_var name 0 ty false in
      Ir.EVar v
  | TEIntrinsicConst ref -> (
      match ref with
      | Sarek_env.IntrinsicRef (path, name) -> Ir.EIntrinsic (path, name, [])
      | Sarek_env.CorePrimitiveRef name -> Ir.EIntrinsic ([], name, []))
  | TEIntrinsicFun (ref, _conv, args) -> (
      match ref with
      | Sarek_env.IntrinsicRef (path, name) ->
          Ir.EIntrinsic (path, name, List.map lower_expr args)
      | Sarek_env.CorePrimitiveRef name ->
          Ir.EIntrinsic ([], name, List.map lower_expr args))
  (* These need statement context *)
  | TEVecSet _ | TEArrSet _ | TEFieldSet _ | TEAssign _ | TELet _ | TELetRec _
  | TELetMut _ | TEIf _ | TEFor _ | TEWhile _ | TESeq _ | TEMatch _ | TEReturn _
  | TECreateArray _ | TENative _ | TEPragma _ | TELetShared _ | TESuperstep _
  | TEOpen _ ->
      failwith "lower_expr: expression requires statement context"

(** Convert a typed expression to IR statement *)
let rec lower_stmt (te : texpr) : Ir.stmt =
  match te.te with
  | TEUnit -> Ir.SEmpty
  | TESeq [] -> Ir.SEmpty
  | TESeq [e] -> lower_stmt e
  | TESeq es -> Ir.SSeq (List.map lower_stmt es)
  | TEVecSet (vec, idx, value) -> (
      match vec.te with
      | TEVar (name, id) ->
          let v = make_var name id vec.ty false in
          Ir.SAssign
            (Ir.LArrayElem (v.var_name, lower_expr idx), lower_expr value)
      | _ -> failwith "lower_stmt: VecSet on non-variable")
  | TEArrSet (arr, idx, value) -> (
      match arr.te with
      | TEVar (name, id) ->
          let v = make_var name id arr.ty false in
          Ir.SAssign
            (Ir.LArrayElem (v.var_name, lower_expr idx), lower_expr value)
      | _ -> failwith "lower_stmt: ArrSet on non-variable")
  | TEFieldSet (r, field, _, value) ->
      let lv = lower_lvalue r field in
      Ir.SAssign (lv, lower_expr value)
  | TEAssign (name, id, value) ->
      let v = make_var name id value.ty true in
      Ir.SAssign (Ir.LVar v, lower_expr value)
  | TELet (name, id, value, body) ->
      let v = make_var name id value.ty false in
      Ir.SLet (v, lower_expr value, lower_stmt body)
  | TELetMut (name, id, value, body) ->
      let v = make_var name id value.ty true in
      Ir.SLetMut (v, lower_expr value, lower_stmt body)
  | TELetRec (_name, _id, _params, _fn_body, cont) ->
      (* Inline functions - for now just emit continuation *)
      lower_stmt cont
  | TEIf (cond, then_, else_opt) ->
      Ir.SIf (lower_expr cond, lower_stmt then_, Option.map lower_stmt else_opt)
  | TEFor (var, id, lo, hi, dir, body) ->
      let v = make_var var id (TPrim TInt32) true in
      let ir_dir = match dir with Upto -> Ir.Upto | Downto -> Ir.Downto in
      Ir.SFor (v, lower_expr lo, lower_expr hi, ir_dir, lower_stmt body)
  | TEWhile (cond, body) -> Ir.SWhile (lower_expr cond, lower_stmt body)
  | TEMatch (e, cases) ->
      let ir_cases =
        List.map (fun (pat, body) -> (lower_pattern pat, lower_stmt body)) cases
      in
      Ir.SMatch (lower_expr e, ir_cases)
  | TEReturn e -> Ir.SReturn (lower_expr e)
  | TEPragma (opts, body) -> Ir.SPragma (opts, lower_stmt body)
  | TELetShared (name, id, elem_ty, size_opt, body) ->
      (* Emit shared declaration followed by body *)
      let _decl =
        Ir.DShared (name, elttype_of_typ elem_ty, Option.map lower_expr size_opt)
      in
      (* For now, we inline the decl as SLet with dummy value *)
      let v = make_var name id (TArr (elem_ty, Sarek_types.Shared)) false in
      Ir.SLet (v, Ir.EConst Ir.CUnit, lower_stmt body)
  | TESuperstep (_name, _divergent, step_body, cont) ->
      (* Superstep becomes: body; barrier; continuation *)
      Ir.SSeq [lower_stmt step_body; Ir.SBarrier; lower_stmt cont]
  | TEOpen (_path, body) -> lower_stmt body
  | TECreateArray (_size, _elem_ty, _mem) ->
      (* Array creation as statement - emit as expr *)
      Ir.SExpr (Ir.EConst Ir.CUnit)
  | TENative _ ->
      (* Native blocks are handled specially *)
      Ir.SEmpty
  (* Pure expressions as statements *)
  | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEVecGet _ | TEArrGet _ | TEFieldGet _ | TEBinop _ | TEUnop _
  | TEApp _ | TERecord _ | TEConstr _ | TETuple _ | TEGlobalRef _
  | TEIntrinsicConst _ | TEIntrinsicFun _ ->
      Ir.SExpr (lower_expr te)

and lower_lvalue (r : texpr) (field : string) : Ir.lvalue =
  match r.te with
  | TEVar (name, id) ->
      let v = make_var name id r.ty false in
      Ir.LRecordField (Ir.LVar v, field)
  | TEFieldGet (base, inner_field, _) ->
      Ir.LRecordField (lower_lvalue base inner_field, field)
  | _ -> failwith "lower_lvalue: expected variable or field access"

and lower_pattern (pat : tpattern) : Ir.pattern =
  match pat.tpat with
  | TPAny -> Ir.PWild
  | TPVar (name, _) -> Ir.PConstr ("", [name])
  | TPConstr (_ty_name, constr, arg_pat) ->
      let vars =
        match arg_pat with None -> [] | Some p -> extract_pattern_vars p
      in
      Ir.PConstr (constr, vars)
  | TPTuple pats ->
      Ir.PConstr ("tuple", List.concat_map extract_pattern_vars pats)

and extract_pattern_vars (pat : tpattern) : string list =
  match pat.tpat with
  | TPAny -> ["_"]
  | TPVar (name, _) -> [name]
  | TPConstr (_, _, Some p) -> extract_pattern_vars p
  | TPConstr (_, _, None) -> []
  | TPTuple pats -> List.concat_map extract_pattern_vars pats

(** Convert a kernel parameter to IR declaration *)
let lower_param (p : tparam) : Ir.decl =
  let elt = elttype_of_typ p.tparam_type in
  let v =
    {
      Ir.var_name = p.tparam_name;
      var_id = p.tparam_id;
      var_type = elt;
      var_mutable = false;
    }
  in
  if p.tparam_is_vec then
    let elem_ty =
      match repr p.tparam_type with TVec t -> elttype_of_typ t | _ -> elt
    in
    Ir.DParam (v, Some {arr_elttype = elem_ty; arr_memspace = Ir.Global})
  else Ir.DParam (v, None)

(** Lower a complete kernel *)
let lower_kernel (k : tkernel) : Ir.kernel =
  {
    Ir.kern_name = Option.value k.tkern_name ~default:"kernel";
    kern_params = List.map lower_param k.tkern_params;
    kern_locals = [];
    kern_body = lower_stmt k.tkern_body;
  }
