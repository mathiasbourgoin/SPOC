(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module implements type inference for Sarek kernels.
 * Uses constraint-based Hindley-Milner style inference.
 ******************************************************************************)

open Sarek_ast
open Sarek_types
open Sarek_env
open Sarek_error
open Sarek_typed_ast

(** Unify with error reporting *)
let unify_or_error t1 t2 loc =
  match unify t1 t2 with
  | Ok () -> Ok ()
  | Error (Cannot_unify (t1, t2)) -> Error [Cannot_unify (t1, t2, loc)]
  | Error (Occurs_check (_, t)) -> Error [Recursive_type (t, loc)]

(** Check that a type is numeric (int32, int64, float32, float64) *)
let check_numeric t loc =
  match repr t with
  | TPrim (TInt32 | TInt64 | TFloat32 | TFloat64) -> Ok ()
  | TVar _ -> Ok () (* Will be constrained later *)
  | t -> Error [Type_mismatch {expected = t_int32; got = t; loc}]

(** Check that a type is integer *)
let check_integer t loc =
  match repr t with
  | TPrim (TInt32 | TInt64) -> Ok ()
  | TVar _ -> Ok ()
  | t -> Error [Type_mismatch {expected = t_int32; got = t; loc}]

(** Check that a type is boolean *)
let check_boolean t loc =
  match repr t with
  | TPrim TBool -> Ok ()
  | TVar _ -> unify_or_error t t_bool loc
  | t -> Error [Type_mismatch {expected = t_bool; got = t; loc}]

(** Convert a parsed type expression using the current typing environment *)
let rec type_of_type_expr_env env te =
  match te with
  | Sarek_ast.TEConstr ("vector", [elem]) ->
      TVec (type_of_type_expr_env env elem)
  | Sarek_ast.TEConstr (name, [elem])
    when String.ends_with ~suffix:"vector" name ->
      TVec (type_of_type_expr_env env elem)
  | Sarek_ast.TEConstr ("array", [elem]) ->
      TArr (type_of_type_expr_env env elem, Local)
  | Sarek_ast.TEArrow (a, b) ->
      TFun ([type_of_type_expr_env env a], type_of_type_expr_env env b)
  | Sarek_ast.TETuple ts -> TTuple (List.map (type_of_type_expr_env env) ts)
  | Sarek_ast.TEConstr (name, _) -> (
      match find_type name env with
      | Some (TIRecord {ti_fields; _}) ->
          TRecord (name, List.map (fun (f, t, _) -> (f, t)) ti_fields)
      | Some (TIVariant {ti_constrs; _}) -> TVariant (name, ti_constrs)
      | None -> type_of_type_expr te)
  | _ -> type_of_type_expr te

(** Infer type of a binary operation *)
let infer_binop op t1 t2 loc =
  match op with
  | Add | Sub | Mul | Div ->
      (* Numeric ops: both operands same type, result same type *)
      let* () = unify_or_error t1 t2 loc in
      let* () = check_numeric t1 loc in
      Ok t1
  | Mod ->
      (* Integer only *)
      let* () = unify_or_error t1 t2 loc in
      let* () = check_integer t1 loc in
      Ok t1
  | Eq | Ne ->
      (* Equality: both same type, result bool *)
      let* () = unify_or_error t1 t2 loc in
      Ok t_bool
  | Lt | Le | Gt | Ge ->
      (* Comparison: both same numeric type, result bool *)
      let* () = unify_or_error t1 t2 loc in
      let* () = check_numeric t1 loc in
      Ok t_bool
  | And | Or ->
      (* Boolean ops *)
      let* () = check_boolean t1 loc in
      let* () = check_boolean t2 loc in
      Ok t_bool
  | Land | Lor | Lxor | Lsl | Lsr | Asr ->
      (* Bitwise ops: integer only *)
      let* () = unify_or_error t1 t2 loc in
      let* () = check_integer t1 loc in
      Ok t1

(** Infer type of a unary operation *)
let infer_unop op t loc =
  match op with
  | Neg ->
      let* () = check_numeric t loc in
      Ok t
  | Not ->
      let* () = check_boolean t loc in
      Ok t_bool
  | Lnot ->
      let* () = check_integer t loc in
      Ok t

(** Main type inference function *)
let rec infer (env : t) (expr : expr) : (texpr * t) result =
  let loc = expr.expr_loc in
  match expr.e with
  (* Literals *)
  | EUnit -> Ok (mk_texpr TEUnit t_unit loc, env)
  | EBool b -> Ok (mk_texpr (TEBool b) t_bool loc, env)
  | EInt i -> Ok (mk_texpr (TEInt i) t_int32 loc, env)
  | EInt32 i -> Ok (mk_texpr (TEInt32 i) t_int32 loc, env)
  | EInt64 i -> Ok (mk_texpr (TEInt64 i) t_int64 loc, env)
  | EFloat f -> Ok (mk_texpr (TEFloat f) t_float32 loc, env)
  | EDouble f -> Ok (mk_texpr (TEDouble f) t_float64 loc, env)
  (* Variables *)
  | EVar name -> (
      match lookup name env with
      | LVar info ->
          let id = info.vi_index in
          (* Use the index as ID *)
          Ok (mk_texpr (TEVar (name, id)) info.vi_type loc, env)
      | LIntrinsicConst info ->
          Ok
            ( mk_texpr
                (TEIntrinsicConst (info.const_cuda, info.const_opencl))
                info.const_type
                loc,
              env )
      | LIntrinsicFun info ->
          (* Return a function value that will be applied *)
          let id = fresh_var_id () in
          Ok (mk_texpr (TEVar (name, id)) info.intr_type loc, env)
      | LConstructor (type_name, arg_type) -> (
          match arg_type with
          | None ->
              let ty = TVariant (type_name, [(name, None)]) in
              Ok (mk_texpr (TEConstr (type_name, name, None)) ty loc, env)
          | Some _ ->
              (* Constructor that takes an argument - return as function *)
              let ty = fresh_tvar () in
              (* Will be resolved when applied *)
              let id = fresh_var_id () in
              Ok (mk_texpr (TEVar (name, id)) ty loc, env))
      | LLocalFun (_, typ) ->
          let id = fresh_var_id () in
          Ok (mk_texpr (TEVar (name, id)) typ loc, env)
      | LNotFound -> Error [Unbound_variable (name, loc)])
  (* Vector access: v.[i] *)
  | EVecGet (vec, idx) ->
      let* tv, env = infer env vec in
      let* ti, env = infer env idx in
      let* () = unify_or_error ti.ty t_int32 idx.expr_loc in
      let elem_ty = fresh_tvar () in
      let* () = unify_or_error tv.ty (TVec elem_ty) vec.expr_loc in
      let resolved_elem = repr elem_ty in
      Ok (mk_texpr (TEVecGet (tv, ti)) resolved_elem loc, env)
  (* Vector set: v.[i] <- x *)
  | EVecSet (vec, idx, value) ->
      let* tv, env = infer env vec in
      let* ti, env = infer env idx in
      let* tx, env = infer env value in
      let* () = unify_or_error ti.ty t_int32 idx.expr_loc in
      let elem_ty = fresh_tvar () in
      let* () = unify_or_error tv.ty (TVec elem_ty) vec.expr_loc in
      let* () = unify_or_error tx.ty elem_ty value.expr_loc in
      Ok (mk_texpr (TEVecSet (tv, ti, tx)) t_unit loc, env)
  (* Array/vector access: a.(i) - works for both TArr and TVec *)
  | EArrGet (arr, idx) -> (
      let* ta, env = infer env arr in
      let* ti, env = infer env idx in
      let* () = unify_or_error ti.ty t_int32 idx.expr_loc in
      let elem_ty = fresh_tvar () in
      (* Try to unify with vector first, then array *)
      match unify ta.ty (TVec elem_ty) with
      | Ok () ->
          let resolved_elem = repr elem_ty in
          Ok (mk_texpr (TEVecGet (ta, ti)) resolved_elem loc, env)
      | Error _ ->
          let mem = Local in
          let* () = unify_or_error ta.ty (TArr (elem_ty, mem)) arr.expr_loc in
          let resolved_elem = repr elem_ty in
          Ok (mk_texpr (TEArrGet (ta, ti)) resolved_elem loc, env))
  (* Array/vector set: a.(i) <- x - works for both TArr and TVec *)
  | EArrSet (arr, idx, value) -> (
      let* ta, env = infer env arr in
      let* ti, env = infer env idx in
      let* tx, env = infer env value in
      let* () = unify_or_error ti.ty t_int32 idx.expr_loc in
      let elem_ty = fresh_tvar () in
      (* Try to unify with vector first, then array *)
      match unify ta.ty (TVec elem_ty) with
      | Ok () ->
          let* () = unify_or_error tx.ty elem_ty value.expr_loc in
          Ok (mk_texpr (TEVecSet (ta, ti, tx)) t_unit loc, env)
      | Error _ ->
          let mem = Local in
          let* () = unify_or_error ta.ty (TArr (elem_ty, mem)) arr.expr_loc in
          let* () = unify_or_error tx.ty elem_ty value.expr_loc in
          Ok (mk_texpr (TEArrSet (ta, ti, tx)) t_unit loc, env))
  (* Field access: r.field *)
  | EFieldGet (record, field) -> (
      let* tr, env = infer env record in
      match repr tr.ty with
      | TRecord (_type_name, fields) -> (
          match List.assoc_opt field fields with
          | Some field_ty ->
              let idx =
                match
                  List.mapi (fun i (f, _) -> (f, i)) fields
                  |> List.assoc_opt field
                with
                | Some i -> i
                | None -> 0
              in
              Ok (mk_texpr (TEFieldGet (tr, field, idx)) field_ty loc, env)
          | None -> Error [Field_not_found (field, tr.ty, loc)])
      | t when is_tvar t ->
          (* Type not yet known, defer *)
          let field_ty = fresh_tvar () in
          Ok (mk_texpr (TEFieldGet (tr, field, 0)) field_ty loc, env)
      | t -> Error [Not_a_record (t, loc)])
  (* Field set: r.field <- x *)
  | EFieldSet (record, field, value) -> (
      let* tr, env = infer env record in
      let* tx, env = infer env value in
      match repr tr.ty with
      | TRecord (_type_name, fields) -> (
          match List.assoc_opt field fields with
          | Some field_ty ->
              let* () = unify_or_error tx.ty field_ty value.expr_loc in
              let idx =
                match
                  List.mapi (fun i (f, _) -> (f, i)) fields
                  |> List.assoc_opt field
                with
                | Some i -> i
                | None -> 0
              in
              Ok (mk_texpr (TEFieldSet (tr, field, idx, tx)) t_unit loc, env)
          | None -> Error [Field_not_found (field, tr.ty, loc)])
      | t -> Error [Not_a_record (t, loc)])
  (* Binary operations *)
  | EBinop (op, e1, e2) ->
      let* t1, env = infer env e1 in
      let* t2, env = infer env e2 in
      let* result_ty = infer_binop op t1.ty t2.ty loc in
      Ok (mk_texpr (TEBinop (op, t1, t2)) result_ty loc, env)
  (* Unary operations *)
  | EUnop (op, e) ->
      let* te, env = infer env e in
      let* result_ty = infer_unop op te.ty loc in
      Ok (mk_texpr (TEUnop (op, te)) result_ty loc, env)
  (* Function application *)
  | EApp (fn, args) -> (
      let* tfn, env = infer env fn in
      let* targs, env = infer_list env args in
      match fn.e with
      | EVar name when is_intrinsic_fun name env -> (
          (* Intrinsic function call *)
          match find_intrinsic_fun name env with
          | Some info -> (
              match info.intr_type with
              | TFun (param_tys, ret_ty) ->
                  if List.length param_tys <> List.length targs then
                    Error
                      [
                        Wrong_arity
                          {
                            expected = List.length param_tys;
                            got = List.length targs;
                            loc;
                          };
                      ]
                  else begin
                    let* () = unify_args param_tys targs loc in
                    Ok
                      ( mk_texpr
                          (TEIntrinsicFun
                             (info.intr_cuda, info.intr_opencl, targs))
                          ret_ty
                          loc,
                        env )
                  end
              | t -> Error [Not_a_function (t, fn.expr_loc)])
          | None -> Error [Unbound_variable (name, fn.expr_loc)])
      | _ ->
          (* Regular function application *)
          let ret_ty = fresh_tvar () in
          let expected_fn_ty = TFun (List.map (fun t -> t.ty) targs, ret_ty) in
          let* () = unify_or_error tfn.ty expected_fn_ty fn.expr_loc in
          Ok (mk_texpr (TEApp (tfn, targs)) (repr ret_ty) loc, env))
  (* Mutable assignment *)
  | EAssign (name, value) -> (
      let* tv, env = infer env value in
      match find_var name env with
      | None -> Error [Unbound_variable (name, loc)]
      | Some vi ->
          if (not vi.vi_mutable) || vi.vi_is_param then
            Error [Immutable_variable (name, loc)]
          else
            let* () = unify_or_error tv.ty vi.vi_type value.expr_loc in
            Ok (mk_texpr (TEAssign (name, vi.vi_index, tv)) t_unit loc, env))
  (* Let binding *)
  | ELet (name, ty_annot, value, body) ->
      let env' = enter_level env in
      let* tv, env' = infer env' value in
      let* () =
        match ty_annot with
        | None -> Ok ()
        | Some te ->
            let t = type_of_type_expr_env env te in
            unify_or_error tv.ty t value.expr_loc
      in
      let var_id = fresh_var_id () in
      let vi =
        {
          vi_type = tv.ty;
          vi_mutable = false;
          vi_is_param = false;
          vi_index = var_id;
          vi_is_vec = false;
        }
      in
      let env'' = add_var name vi (exit_level env') in
      let* tb, env'' = infer env'' body in
      Ok (mk_texpr (TELet (name, var_id, tv, tb)) tb.ty loc, env'')
  (* Mutable let binding *)
  | ELetMut (name, ty_annot, value, body) ->
      let* tv, env = infer env value in
      let* () =
        match ty_annot with
        | None -> Ok ()
        | Some te ->
            let t = type_of_type_expr_env env te in
            unify_or_error tv.ty t value.expr_loc
      in
      let var_id = fresh_var_id () in
      let vi =
        {
          vi_type = tv.ty;
          vi_mutable = true;
          vi_is_param = false;
          vi_index = var_id;
          vi_is_vec = false;
        }
      in
      let env' = add_var name vi env in
      let* tb, env' = infer env' body in
      Ok (mk_texpr (TELetMut (name, var_id, tv, tb)) tb.ty loc, env')
  (* If-then-else *)
  | EIf (cond, then_e, else_opt) -> (
      let* tc, env = infer env cond in
      let* () = check_boolean tc.ty cond.expr_loc in
      let* tt, env = infer env then_e in
      match else_opt with
      | None ->
          let* () = unify_or_error tt.ty t_unit then_e.expr_loc in
          Ok (mk_texpr (TEIf (tc, tt, None)) t_unit loc, env)
      | Some else_e ->
          let* te, env = infer env else_e in
          let* () = unify_or_error tt.ty te.ty else_e.expr_loc in
          Ok (mk_texpr (TEIf (tc, tt, Some te)) tt.ty loc, env))
  (* For loop *)
  | EFor (var, lo, hi, dir, body) ->
      let* tlo, env = infer env lo in
      let* thi, env = infer env hi in
      let* () = unify_or_error tlo.ty t_int32 lo.expr_loc in
      let* () = unify_or_error thi.ty t_int32 hi.expr_loc in
      let var_id = fresh_var_id () in
      let vi =
        {
          vi_type = t_int32;
          vi_mutable = false;
          vi_is_param = false;
          vi_index = var_id;
          vi_is_vec = false;
        }
      in
      let env' = add_var var vi env in
      let* tbody, _ = infer env' body in
      Ok (mk_texpr (TEFor (var, var_id, tlo, thi, dir, tbody)) t_unit loc, env)
  (* While loop *)
  | EWhile (cond, body) ->
      let* tc, env = infer env cond in
      let* () = check_boolean tc.ty cond.expr_loc in
      let* tbody, _ = infer env body in
      Ok (mk_texpr (TEWhile (tc, tbody)) t_unit loc, env)
  (* Sequence *)
  | ESeq (e1, e2) ->
      let* t1, env = infer env e1 in
      let* t2, env = infer env e2 in
      Ok (mk_texpr (TESeq [t1; t2]) t2.ty loc, env)
  (* Match *)
  | EMatch (scrutinee, cases) ->
      let* ts, env = infer env scrutinee in
      let* tcases, result_ty, env = infer_match_cases env ts.ty cases loc in
      Ok (mk_texpr (TEMatch (ts, tcases)) result_ty loc, env)
  (* Record construction *)
  | ERecord (name_opt, fields) ->
      let* tfields, env = infer_record_fields env fields in
      let type_name =
        match name_opt with Some n -> n | None -> "anon_record"
      in
      let field_tys = List.map (fun (f, te) -> (f, te.ty)) tfields in
      let* ty =
        match name_opt with
        | Some n -> (
            match find_type n env with
            | Some (TIRecord {ti_fields; _}) ->
                Ok (TRecord (n, List.map (fun (f, t, _) -> (f, t)) ti_fields))
            | Some (TIVariant _) -> Error [Not_a_record (TVariant (n, []), loc)]
            | None -> Ok (TRecord (n, field_tys)))
        | None -> Ok (TRecord (type_name, field_tys))
      in
      Ok (mk_texpr (TERecord (type_name, tfields)) ty loc, env)
  (* Constructor application *)
  | EConstr (name, arg_opt) -> (
      match find_constructor name env with
      | Some (type_name, expected_arg) -> (
          match (arg_opt, expected_arg) with
          | None, None ->
              let ty = TVariant (type_name, [(name, None)]) in
              Ok (mk_texpr (TEConstr (type_name, name, None)) ty loc, env)
          | Some arg, Some expected_ty ->
              let* targ, env = infer env arg in
              let* () = unify_or_error targ.ty expected_ty arg.expr_loc in
              let ty = TVariant (type_name, [(name, Some targ.ty)]) in
              Ok (mk_texpr (TEConstr (type_name, name, Some targ)) ty loc, env)
          | None, Some _ -> Error [Wrong_arity {expected = 1; got = 0; loc}]
          | Some _, None -> Error [Wrong_arity {expected = 0; got = 1; loc}])
      | None -> Error [Unbound_constructor (name, loc)])
  (* Tuple *)
  | ETuple es ->
      let* tes, env = infer_list env es in
      let ty = TTuple (List.map (fun te -> te.ty) tes) in
      Ok (mk_texpr (TETuple tes) ty loc, env)
  (* Return *)
  | EReturn e ->
      let* te, env = infer env e in
      Ok (mk_texpr (TEReturn te) te.ty loc, env)
  (* Create array *)
  | ECreateArray (size, elem_ty, mem) ->
      let* tsize, env = infer env size in
      let* () = unify_or_error tsize.ty t_int32 size.expr_loc in
      let elem_t = type_of_type_expr_env env elem_ty in
      let arr_ty = TArr (elem_t, memspace_of_ast mem) in
      Ok
        ( mk_texpr
            (TECreateArray (tsize, elem_t, memspace_of_ast mem))
            arr_ty
            loc,
          env )
  (* Global reference *)
  | EGlobalRef name ->
      (* Type will be inferred from context or annotated *)
      let ty = fresh_tvar () in
      Ok (mk_texpr (TEGlobalRef (name, ty)) ty loc, env)
  (* Native code *)
  | ENative s ->
      let ty = fresh_tvar () in
      Ok (mk_texpr (TENative s) ty loc, env)
  (* Type annotation *)
  | ETyped (e, ty_expr) ->
      let* te, env = infer env e in
      let ty = type_of_type_expr_env env ty_expr in
      let* () = unify_or_error te.ty ty loc in
      Ok ({te with ty = repr ty}, env)
  (* Module open *)
  | EOpen (_path, e) ->
      (* For now, just infer the inner expression - opens are handled by intrinsics *)
      infer env e

and infer_list env exprs =
  let rec aux env acc = function
    | [] -> Ok (List.rev acc, env)
    | e :: rest ->
        let* te, env = infer env e in
        aux env (te :: acc) rest
  in
  aux env [] exprs

and unify_args param_tys targs loc =
  let rec aux = function
    | [], [] -> Ok ()
    | pt :: pts, ta :: tas ->
        let* () = unify_or_error pt ta.ty ta.te_loc in
        aux (pts, tas)
    | _, _ ->
        Error
          [
            Wrong_arity
              {expected = List.length param_tys; got = List.length targs; loc};
          ]
  in
  aux (param_tys, targs)

and infer_record_fields env fields =
  let rec aux env acc = function
    | [] -> Ok (List.rev acc, env)
    | (name, e) :: rest ->
        let* te, env = infer env e in
        aux env ((name, te) :: acc) rest
  in
  aux env [] fields

and infer_match_cases env scrutinee_ty cases loc =
  match cases with
  | [] -> Error [Invalid_kernel ("empty match", loc)]
  | (pat, body) :: rest ->
      let* tpat, env' = infer_pattern env scrutinee_ty pat in
      let* tbody, _ = infer env' body in
      let result_ty = tbody.ty in
      let* tcases = infer_remaining_cases env scrutinee_ty result_ty rest in
      Ok ((tpat, tbody) :: tcases, result_ty, env)

and infer_remaining_cases env scrutinee_ty result_ty cases =
  let rec aux acc = function
    | [] -> Ok (List.rev acc)
    | (pat, body) :: rest ->
        let* tpat, env' = infer_pattern env scrutinee_ty pat in
        let* tbody, _ = infer env' body in
        let* () = unify_or_error tbody.ty result_ty body.expr_loc in
        aux ((tpat, tbody) :: acc) rest
  in
  aux [] cases

and infer_pattern env scrutinee_ty pat =
  let loc = pat.pat_loc in
  match pat.pat with
  | PAny -> Ok ({tpat = TPAny; tpat_ty = scrutinee_ty; tpat_loc = loc}, env)
  | PVar name ->
      let var_id = fresh_var_id () in
      let vi =
        {
          vi_type = scrutinee_ty;
          vi_mutable = false;
          vi_is_param = false;
          vi_index = var_id;
          vi_is_vec = false;
        }
      in
      let env' = add_var name vi env in
      Ok
        ( {tpat = TPVar (name, var_id); tpat_ty = scrutinee_ty; tpat_loc = loc},
          env' )
  | PConstr (name, arg_pat) -> (
      match find_constructor name env with
      | Some (type_name, arg_ty_opt) -> (
          match (arg_pat, arg_ty_opt) with
          | None, None ->
              Ok
                ( {
                    tpat = TPConstr (type_name, name, None);
                    tpat_ty = scrutinee_ty;
                    tpat_loc = loc;
                  },
                  env )
          | Some ap, Some arg_ty ->
              let* tap, env' = infer_pattern env arg_ty ap in
              Ok
                ( {
                    tpat = TPConstr (type_name, name, Some tap);
                    tpat_ty = scrutinee_ty;
                    tpat_loc = loc;
                  },
                  env' )
          | None, Some _ -> Error [Wrong_arity {expected = 1; got = 0; loc}]
          | Some _, None -> Error [Wrong_arity {expected = 0; got = 1; loc}])
      | None -> Error [Unbound_constructor (name, loc)])
  | PTuple pats ->
      let* ty_list =
        match repr scrutinee_ty with
        | TTuple ts when List.length ts = List.length pats -> Ok ts
        | _ ->
            let ts = List.map (fun _ -> fresh_tvar ()) pats in
            let* () = unify_or_error scrutinee_ty (TTuple ts) loc in
            Ok ts
      in
      let* tpats, env' = infer_patterns env ty_list pats in
      Ok ({tpat = TPTuple tpats; tpat_ty = scrutinee_ty; tpat_loc = loc}, env')

and infer_patterns env tys pats =
  let rec aux env acc = function
    | [], [] -> Ok (List.rev acc, env)
    | ty :: tys, pat :: pats ->
        let* tpat, env' = infer_pattern env ty pat in
        aux env' (tpat :: acc) (tys, pats)
    | _ -> assert false
  in
  aux env [] (tys, pats)

(** Check if a name is an intrinsic function in the environment *)
and is_intrinsic_fun name env =
  match find_intrinsic_fun name env with Some _ -> true | None -> false

(** Check if a type is still an unbound type variable *)
and is_tvar t =
  match repr t with TVar {contents = Unbound _} -> true | _ -> false

(** Type a complete kernel *)
let infer_kernel (env : t) (kernel : Sarek_ast.kernel) : tkernel result =
  (* Register type declarations first *)
  let rec add_type_decls env acc = function
    | [] -> Ok (List.rev acc, env)
    | decl :: rest -> (
        match decl with
        | Sarek_ast.Type_record {tdecl_name; tdecl_fields; tdecl_loc} ->
            let tfields =
              List.map
                (fun (fname, fmut, fty) ->
                  (fname, type_of_type_expr_env env fty, fmut))
                tdecl_fields
            in
            let env' =
              add_type
                tdecl_name
                (TIRecord {ti_name = tdecl_name; ti_fields = tfields})
                env
            in
            let acc_decl =
              TTypeRecord {tdecl_name; tdecl_fields = tfields; tdecl_loc}
            in
            add_type_decls env' (acc_decl :: acc) rest
        | Sarek_ast.Type_variant {tdecl_name; tdecl_constructors; tdecl_loc} ->
            let constrs =
              List.map
                (fun (cname, carg) ->
                  (cname, Option.map (type_of_type_expr_env env) carg))
                tdecl_constructors
            in
            let env' =
              add_type
                tdecl_name
                (TIVariant {ti_name = tdecl_name; ti_constrs = constrs})
                env
            in
            let acc_decl =
              TTypeVariant {tdecl_name; tdecl_constructors = constrs; tdecl_loc}
            in
            add_type_decls env' (acc_decl :: acc) rest)
  in
  let* ttypes, env_with_types = add_type_decls env [] kernel.kern_types in

  (* Type module items next to extend the environment *)
  let rec add_module_items env acc = function
    | [] -> Ok (List.rev acc, env)
    | item :: rest -> (
        match item with
        | Sarek_ast.MConst (name, ty_expr, value) ->
            let ty = type_of_type_expr_env env ty_expr in
            let* tvalue, env' = infer env value in
            let* () = unify_or_error tvalue.ty ty value.expr_loc in
            let var_id = fresh_var_id () in
            let vi =
              {
                vi_type = ty;
                vi_mutable = false;
                vi_is_param = false;
                vi_index = var_id;
                vi_is_vec = false;
              }
            in
            let env'' = add_var name vi env' in
            add_module_items
              env''
              (TMConst (name, var_id, ty, tvalue) :: acc)
              rest
        | Sarek_ast.MFun (name, params, body) ->
            let rec add_fun_params env idx acc = function
              | [] -> Ok (List.rev acc, env)
              | p :: rest ->
                  let ty = type_of_type_expr_env env p.param_type in
                  let is_vec = match ty with TVec _ -> true | _ -> false in
                  let var_id = fresh_var_id () in
                  let vi =
                    {
                      vi_type = ty;
                      vi_mutable = false;
                      vi_is_param = false;
                      vi_index = var_id;
                      vi_is_vec = is_vec;
                    }
                  in
                  let env' = add_var p.param_name vi env in
                  let tparam =
                    {
                      tparam_name = p.param_name;
                      tparam_type = ty;
                      tparam_index = idx;
                      tparam_is_vec = is_vec;
                      tparam_id = var_id;
                    }
                  in
                  add_fun_params env' (idx + 1) (tparam :: acc) rest
            in
            let* tparams, env_fun = add_fun_params env 0 [] params in
            let* tbody, _ = infer env_fun body in
            let fn_ty =
              TFun (List.map (fun p -> p.tparam_type) tparams, tbody.ty)
            in
            let env'' = add_local_fun name fn_ty env in
            add_module_items env'' (TMFun (name, tparams, tbody) :: acc) rest)
  in
  let* tmodule_items, env_after_mods =
    add_module_items env_with_types [] kernel.kern_module_items
  in
  (* Add parameters to environment *)
  let rec add_params env idx acc = function
    | [] -> Ok (List.rev acc, env)
    | p :: rest ->
        let ty = type_of_type_expr_env env p.param_type in
        let is_vec = match ty with TVec _ -> true | _ -> false in
        let var_id = fresh_var_id () in
        let vi =
          {
            vi_type = ty;
            vi_mutable = false;
            vi_is_param = true;
            vi_index = var_id;
            vi_is_vec = is_vec;
          }
        in
        let env' = add_var p.param_name vi env in
        let tparam =
          {
            tparam_name = p.param_name;
            tparam_type = ty;
            tparam_index = idx;
            tparam_is_vec = is_vec;
            tparam_id = var_id;
          }
        in
        add_params env' (idx + 1) (tparam :: acc) rest
  in
  let* tparams, env' = add_params env_after_mods 0 [] kernel.kern_params in
  let* tbody, _ = infer env' kernel.kern_body in
  Ok
    {
      tkern_name = kernel.kern_name;
      tkern_type_decls = ttypes;
      tkern_module_items = tmodule_items;
      tkern_params = tparams;
      tkern_body = tbody;
      tkern_return_type = tbody.ty;
      tkern_loc = kernel.kern_loc;
    }
