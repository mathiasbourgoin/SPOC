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

(** Mangle type names for C compatibility *)
let mangle_type_name name = String.map (function '.' -> '_' | c -> c) name

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
  | TTuple _ -> Ir.TInt32
  | TFun _ -> Ir.TInt32
  | TVar _ -> Ir.TInt32
  | TReg _ -> Ir.TInt32

(** Get C type string for a typ *)
let rec c_type_of_typ ty =
  match repr ty with
  | TPrim TInt32 -> "int"
  | TPrim TBool -> "int"
  | TPrim TUnit -> "void"
  | TReg "int64" -> "long"
  | TReg "float32" -> "float"
  | TReg "float64" -> "double"
  | TRecord (name, _) -> "struct " ^ mangle_type_name name ^ "_sarek"
  | TVariant (name, _) -> "struct " ^ mangle_type_name name ^ "_sarek"
  | TVec t -> c_type_of_typ t ^ " *"
  | TArr (t, _) -> c_type_of_typ t ^ " *"
  | _ -> "int"

(** Generate C struct definition and builder for record types *)
let record_constructor_strings name (fields : (string * typ) list) =
  let name = mangle_type_name name in
  let struct_name = name ^ "_sarek" in
  let struct_fields =
    List.map
      (fun (fname, fty) -> "  " ^ c_type_of_typ fty ^ " " ^ fname ^ ";")
      fields
  in
  let struct_def =
    "struct " ^ struct_name ^ " {\n" ^ String.concat "\n" struct_fields ^ "\n};"
  in
  let params =
    String.concat
      ", "
      (List.map (fun (fname, fty) -> c_type_of_typ fty ^ " " ^ fname) fields)
  in
  let assigns =
    String.concat
      "\n"
      (List.map
         (fun (fname, _) -> "  res." ^ fname ^ " = " ^ fname ^ ";")
         fields)
  in
  let builder =
    "struct " ^ struct_name ^ " build_" ^ struct_name ^ "(" ^ params ^ ") {\n"
    ^ "  struct " ^ struct_name ^ " res;\n" ^ assigns ^ "\n  return res;\n}"
  in
  [struct_def; builder]

(** Generate C struct definitions and builders for variant types *)
let variant_constructor_strings name constrs =
  let name = mangle_type_name name in
  let struct_name = name ^ "_sarek" in
  let constr_structs =
    List.map
      (fun (cname, carg) ->
        let field =
          match carg with
          | None -> "  int " ^ name ^ "_sarek_" ^ cname ^ "_t;"
          | Some ty ->
              "  " ^ c_type_of_typ ty ^ " " ^ name ^ "_sarek_" ^ cname ^ "_t;"
        in
        "struct " ^ name ^ "_sarek_" ^ cname ^ " {\n" ^ field ^ "\n};")
      constrs
  in
  let union_fields =
    List.map
      (fun (cname, _) ->
        "  struct " ^ name ^ "_sarek_" ^ cname ^ " " ^ name ^ "_sarek_" ^ cname
        ^ ";")
      constrs
  in
  let union_def =
    "union " ^ name ^ "_sarek_union {\n"
    ^ String.concat "\n" union_fields
    ^ "\n};"
  in
  let main_struct =
    "struct " ^ struct_name ^ " {\n" ^ "  int " ^ name ^ "_sarek_tag;\n"
    ^ "  union " ^ name ^ "_sarek_union " ^ name ^ "_sarek_union;\n" ^ "};"
  in
  let builders =
    List.mapi
      (fun idx (cname, carg) ->
        let params, assign =
          match carg with
          | None -> ("", "  /* no payload */")
          | Some ty ->
              let pname = "v" in
              ( c_type_of_typ ty ^ " " ^ pname,
                "  res." ^ name ^ "_sarek_union." ^ name ^ "_sarek_" ^ cname
                ^ "." ^ name ^ "_sarek_" ^ cname ^ "_t = " ^ pname ^ ";" )
        in
        "struct " ^ struct_name ^ " build_" ^ name ^ "_" ^ cname ^ "(" ^ params
        ^ ") {\n" ^ "  struct " ^ struct_name ^ " res;\n" ^ "  res." ^ name
        ^ "_sarek_tag = " ^ string_of_int idx ^ ";\n" ^ assign ^ "\n"
        ^ "  return res;\n}")
      constrs
  in
  constr_structs @ (union_def :: main_struct :: builders)

(** Lowering state *)
type state = {
  mutable next_var_id : int;
  fun_map : (string, tparam list * texpr) Hashtbl.t;
  lowering_stack : (string, unit) Hashtbl.t;
  lowered_funs : (string, Ir.stmt * string) Hashtbl.t;
}

let create_state fun_map =
  {
    next_var_id = 0;
    fun_map;
    lowering_stack = Hashtbl.create 8;
    lowered_funs = Hashtbl.create 8;
  }

let fresh_id state =
  let id = state.next_var_id in
  state.next_var_id <- id + 1 ;
  id

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
  | Asr -> Ir.Shr
  | Land -> Ir.BitAnd
  | Lor -> Ir.BitOr
  | Lxor -> Ir.BitXor

(** Convert Sarek_ast.unop to Sarek_ir_ppx.unop *)
let ir_unop (op : unop) : Ir.unop =
  match op with Neg -> Ir.Neg | Not -> Ir.Not | Lnot -> Ir.BitNot

(** Convert memspace *)
let lower_memspace = function
  | Local -> Ir.Local
  | Shared -> Ir.Shared
  | Global -> Ir.Global

(** Create a var from typed var info *)
let make_var name id ty mutable_ : Ir.var =
  {
    var_name = name;
    var_id = id;
    var_type = elttype_of_typ ty;
    var_mutable = mutable_;
  }

(** Lower a declaration *)
let lower_decl ~mutable_ id name ty : Ir.decl =
  let v = make_var name id ty mutable_ in
  Ir.DLocal (v, None)

(** Convert a typed expression to IR expression *)
let rec lower_expr (state : state) (te : texpr) : Ir.expr =
  match te.te with
  | TEUnit -> Ir.EConst Ir.CUnit
  | TEBool b -> Ir.EConst (Ir.CBool b)
  | TEInt i -> Ir.EConst (Ir.CInt32 (Int32.of_int i))
  | TEInt32 i -> Ir.EConst (Ir.CInt32 i)
  | TEInt64 i -> Ir.EConst (Ir.CInt64 i)
  | TEFloat f -> Ir.EConst (Ir.CFloat32 f)
  | TEDouble f -> Ir.EConst (Ir.CFloat64 f)
  | TEVar (name, id) -> (
      match repr te.ty with
      | TFun _ ->
          (* Function reference - just use the name *)
          let v = make_var name id te.ty false in
          Ir.EVar v
      | _ ->
          let v = make_var name id te.ty false in
          Ir.EVar v)
  | TEVecGet (vec, idx) -> (
      match vec.te with
      | TEVar (name, _) -> Ir.EArrayRead (name, lower_expr state idx)
      | _ -> failwith "lower_expr: VecGet on non-variable")
  | TEArrGet (arr, idx) -> (
      match arr.te with
      | TEVar (name, _) -> Ir.EArrayRead (name, lower_expr state idx)
      | _ -> failwith "lower_expr: ArrGet on non-variable")
  | TEFieldGet (r, field, _) -> Ir.ERecordField (lower_expr state r, field)
  | TEBinop (op, a, b) ->
      Ir.EBinop (ir_binop op te.ty, lower_expr state a, lower_expr state b)
  | TEUnop (op, a) -> Ir.EUnop (ir_unop op, lower_expr state a)
  | TEApp (fn, args) -> (
      let args_ir = List.map (lower_expr state) args in
      match fn.te with
      | TEVar (name, _) when Hashtbl.mem state.fun_map name ->
          (* Module-level function call *)
          if Hashtbl.mem state.lowered_funs name then
            (* Already lowered - use cached *)
            Ir.EApp (Ir.EVar (make_var name 0 fn.ty false), args_ir)
          else if Hashtbl.mem state.lowering_stack name then
            (* Recursive call - emit by name *)
            Ir.EApp (Ir.EVar (make_var name 0 fn.ty false), args_ir)
          else
            (* First time - lower the function *)
            let _params, body = Hashtbl.find state.fun_map name in
            let ret_ty = repr body.ty in
            let ret_str = c_type_of_typ ret_ty in
            Hashtbl.add state.lowering_stack name () ;
            let fun_body_ir = lower_stmt state body in
            Hashtbl.remove state.lowering_stack name ;
            let fun_body_ir =
              match fun_body_ir with
              | Ir.SReturn _ -> fun_body_ir
              | _ -> Ir.SReturn (lower_expr state body)
            in
            Hashtbl.add state.lowered_funs name (fun_body_ir, ret_str) ;
            Ir.EApp (Ir.EVar (make_var name 0 fn.ty false), args_ir)
      | _ -> Ir.EApp (lower_expr state fn, args_ir))
  | TERecord (name, fields) ->
      Ir.ERecord (name, List.map (fun (n, e) -> (n, lower_expr state e)) fields)
  | TEConstr (ty_name, constr, arg) ->
      let args = match arg with None -> [] | Some e -> [lower_expr state e] in
      Ir.EVariant (ty_name, constr, args)
  | TETuple exprs -> Ir.ETuple (List.map (lower_expr state) exprs)
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
          Ir.EIntrinsic (path, name, List.map (lower_expr state) args)
      | Sarek_env.CorePrimitiveRef name ->
          Ir.EIntrinsic ([], name, List.map (lower_expr state) args))
  (* These need statement context *)
  | TEVecSet _ | TEArrSet _ | TEFieldSet _ | TEAssign _ | TELet _ | TELetRec _
  | TELetMut _ | TEIf _ | TEFor _ | TEWhile _ | TESeq _ | TEMatch _ | TEReturn _
  | TECreateArray _ | TENative _ | TEPragma _ | TELetShared _ | TESuperstep _
  | TEOpen _ ->
      failwith "lower_expr: expression requires statement context"

(** Convert a typed expression to IR statement *)
and lower_stmt (state : state) (te : texpr) : Ir.stmt =
  match te.te with
  | TEUnit -> Ir.SEmpty
  | TESeq [] -> Ir.SEmpty
  | TESeq [e] -> lower_stmt state e
  | TESeq es -> Ir.SSeq (List.map (lower_stmt state) es)
  | TEVecSet (vec, idx, value) -> (
      match vec.te with
      | TEVar (name, _id) ->
          Ir.SAssign
            (Ir.LArrayElem (name, lower_expr state idx), lower_expr state value)
      | _ -> failwith "lower_stmt: VecSet on non-variable")
  | TEArrSet (arr, idx, value) -> (
      match arr.te with
      | TEVar (name, _id) ->
          Ir.SAssign
            (Ir.LArrayElem (name, lower_expr state idx), lower_expr state value)
      | _ -> failwith "lower_stmt: ArrSet on non-variable")
  | TEFieldSet (r, field, _, value) ->
      let lv = lower_lvalue state r field in
      Ir.SAssign (lv, lower_expr state value)
  | TEAssign (name, id, value) ->
      let v = make_var name id value.ty true in
      Ir.SAssign (Ir.LVar v, lower_expr state value)
  | TELet (name, id, value, body) -> (
      match value.te with
      (* Special case: create_array - need proper array declaration *)
      | TECreateArray (size, elem_ty, mem) ->
          let _size_ir = lower_expr state size in
          let v = make_var name id (TArr (elem_ty, mem)) false in
          let body_ir = lower_stmt state body in
          (* Emit as let with array length expression *)
          Ir.SLet (v, Ir.EArrayLen (string_of_int (fresh_id state)), body_ir)
      (* Normal let binding *)
      | _ ->
          let v = make_var name id value.ty false in
          Ir.SLet (v, lower_expr state value, lower_stmt state body))
  | TELetMut (name, id, value, body) ->
      let v = make_var name id value.ty true in
      Ir.SLetMut (v, lower_expr state value, lower_stmt state body)
  | TELetRec (_name, _id, _params, _fn_body, cont) ->
      (* Inline functions - just emit continuation *)
      lower_stmt state cont
  | TEIf (cond, then_, else_opt) ->
      Ir.SIf
        ( lower_expr state cond,
          lower_stmt state then_,
          Option.map (lower_stmt state) else_opt )
  | TEFor (var, id, lo, hi, dir, body) ->
      let v = make_var var id (TPrim TInt32) true in
      let ir_dir = match dir with Upto -> Ir.Upto | Downto -> Ir.Downto in
      Ir.SFor
        ( v,
          lower_expr state lo,
          lower_expr state hi,
          ir_dir,
          lower_stmt state body )
  | TEWhile (cond, body) ->
      Ir.SWhile (lower_expr state cond, lower_stmt state body)
  | TEMatch (e, cases) ->
      let ir_cases =
        List.map
          (fun (pat, body) -> (lower_pattern pat, lower_stmt state body))
          cases
      in
      Ir.SMatch (lower_expr state e, ir_cases)
  | TEReturn e -> Ir.SReturn (lower_expr state e)
  | TEPragma (opts, body) -> Ir.SPragma (opts, lower_stmt state body)
  | TELetShared (name, id, elem_ty, size_opt, body) ->
      let size_ir =
        match size_opt with
        | Some size -> lower_expr state size
        | None -> Ir.EIntrinsic (["Sarek_stdlib"; "Gpu"], "block_dim_x", [])
      in
      let v = make_var name id (TArr (elem_ty, Sarek_types.Shared)) false in
      Ir.SLet (v, size_ir, lower_stmt state body)
  | TESuperstep (_name, _divergent, step_body, cont) ->
      Ir.SSeq [lower_stmt state step_body; Ir.SBarrier; lower_stmt state cont]
  | TEOpen (_path, body) -> lower_stmt state body
  | TECreateArray (_size, _elem_ty, _mem) ->
      (* Standalone array creation - just emit unit *)
      Ir.SExpr (Ir.EConst Ir.CUnit)
  | TENative _ -> Ir.SEmpty
  (* Pure expressions as statements *)
  | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEVecGet _ | TEArrGet _ | TEFieldGet _ | TEBinop _ | TEUnop _
  | TEApp _ | TERecord _ | TEConstr _ | TETuple _ | TEGlobalRef _
  | TEIntrinsicConst _ | TEIntrinsicFun _ ->
      Ir.SExpr (lower_expr state te)

and lower_lvalue (state : state) (r : texpr) (field : string) : Ir.lvalue =
  match r.te with
  | TEVar (name, id) ->
      let v = make_var name id r.ty false in
      Ir.LRecordField (Ir.LVar v, field)
  | TEFieldGet (base, inner_field, _) ->
      Ir.LRecordField (lower_lvalue state base inner_field, field)
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
let lower_kernel (kernel : tkernel) : Ir.kernel * string list =
  (* Build fun_map from module items *)
  let fun_map = Hashtbl.create 8 in
  List.iter
    (function
      | TMFun (name, _, params, body) ->
          Hashtbl.replace fun_map name (params, body)
      | _ -> ())
    kernel.tkern_module_items ;
  let state = create_state fun_map in

  (* Lower module-level constants *)
  let module_items_ir =
    List.fold_right
      (fun item acc ->
        match item with
        | TMConst (name, id, ty, expr) ->
            let v = make_var name id ty false in
            Ir.SSeq [Ir.SLet (v, lower_expr state expr, Ir.SEmpty); acc]
        | TMFun _ -> acc)
      kernel.tkern_module_items
      Ir.SEmpty
  in

  (* Generate type constructors *)
  let constructors =
    List.concat
      (List.map
         (function
           | TTypeRecord {tdecl_name; tdecl_fields; _} ->
               (* Strip mutability flag from fields *)
               let fields = List.map (fun (n, ty, _) -> (n, ty)) tdecl_fields in
               record_constructor_strings tdecl_name fields
           | TTypeVariant {tdecl_name; tdecl_constructors; _} ->
               variant_constructor_strings tdecl_name tdecl_constructors)
         kernel.tkern_type_decls)
  in

  (* Lower body *)
  let body_ir = lower_stmt state kernel.tkern_body in
  let full_body =
    match module_items_ir with
    | Ir.SEmpty -> body_ir
    | _ -> Ir.SSeq [module_items_ir; body_ir]
  in

  ( {
      Ir.kern_name = Option.value kernel.tkern_name ~default:"kernel";
      kern_params = List.map lower_param kernel.tkern_params;
      kern_locals = [];
      kern_body = full_body;
    },
    constructors )

(** Get the return value declaration for a kernel *)
let lower_return_value (kernel : tkernel) : Ir.decl option =
  match repr kernel.tkern_return_type with
  | TPrim TUnit -> None
  | ty ->
      let v = make_var "result" 0 ty true in
      Some (Ir.DLocal (v, None))
