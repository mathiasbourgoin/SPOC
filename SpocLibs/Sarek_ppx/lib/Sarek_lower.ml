(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module lowers the typed AST to Kirc_Ast.k_ext, the IR used by the
 * existing code generation backend.
 ******************************************************************************)

open Ppxlib
open Sarek_ast
open Sarek_types
open Sarek_typed_ast

let mangle_type_name name = String.map (function '.' -> '_' | c -> c) name

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

let record_constructor_strings name (fields : (string * typ * bool) list) =
  let name = mangle_type_name name in
  let struct_name = name ^ "_sarek" in
  let struct_fields =
    List.map
      (fun (fname, fty, _) -> "  " ^ c_type_of_typ fty ^ " " ^ fname ^ ";")
      fields
  in
  let struct_def =
    "struct " ^ struct_name ^ " {\n" ^ String.concat "\n" struct_fields ^ "\n};"
  in
  let params =
    String.concat
      ", "
      (List.map (fun (fname, fty, _) -> c_type_of_typ fty ^ " " ^ fname) fields)
  in
  let assigns =
    String.concat
      "\n"
      (List.map
         (fun (fname, _, _) -> "  res." ^ fname ^ " = " ^ fname ^ ";")
         fields)
  in
  let builder =
    "struct " ^ struct_name ^ " build_" ^ struct_name ^ "(" ^ params ^ ") {\n"
    ^ "  struct " ^ struct_name ^ " res;\n" ^ assigns ^ "\n  return res;\n}"
  in
  (* Emit struct definition first so OpenCL can see the type in builder
     signature. *)
  [struct_def; builder]

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
      (fun (cname, _carg) ->
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

let c_type_of_core_type ~loc (ct : core_type) =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident "float32"; _}, _) -> "float"
  | Ptyp_constr ({txt = Lident "float64"; _}, _) -> "double"
  | Ptyp_constr ({txt = Lident "float"; _}, _) -> "float"
  | Ptyp_constr ({txt = Lident "int32"; _}, _) -> "int"
  | Ptyp_constr ({txt = Lident "int"; _}, _) -> "int"
  | _ ->
      Location.raise_errorf
        ~loc
        "Unsupported type in Sarek top-level registration"

let typ_of_core_type ~loc (ct : core_type) =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident "float32"; _}, _) -> TReg "float32"
  | Ptyp_constr ({txt = Lident "float64"; _}, _) -> TReg "float64"
  | Ptyp_constr ({txt = Lident "float"; _}, _) -> TReg "float32"
  | Ptyp_constr ({txt = Lident "int32"; _}, _) -> TPrim TInt32
  | Ptyp_constr ({txt = Lident "int"; _}, _) -> TPrim TInt32
  | Ptyp_constr ({txt = Lident "int64"; _}, _) -> TReg "int64"
  | _ ->
      Location.raise_errorf
        ~loc
        "Unsupported type in Sarek top-level registration"

let constructor_strings_of_core_type_decl ~loc (tdecl : type_declaration) =
  match tdecl.ptype_kind with
  | Ptype_record labels ->
      let fields =
        List.map
          (fun ld ->
            ( ld.pld_name.txt,
              typ_of_core_type ~loc ld.pld_type,
              ld.pld_mutable = Mutable ))
          labels
      in
      let strs = record_constructor_strings tdecl.ptype_name.txt fields in
      Ast_builder.Default.elist
        ~loc
        (List.map (Ast_builder.Default.estring ~loc) strs)
  | Ptype_variant constrs ->
      let constrs =
        List.map
          (fun cd ->
            match cd.pcd_args with
            | Pcstr_tuple [] -> (cd.pcd_name.txt, None)
            | Pcstr_tuple [ct] ->
                let _ = typ_of_core_type ~loc ct in
                (cd.pcd_name.txt, None)
            | Pcstr_tuple _ | Pcstr_record _ ->
                Location.raise_errorf
                  ~loc
                  "Only zero or single-argument constructors supported")
          constrs
      in
      let strs = variant_constructor_strings tdecl.ptype_name.txt constrs in
      Ast_builder.Default.elist
        ~loc
        (List.map (Ast_builder.Default.estring ~loc) strs)
  | _ ->
      Location.raise_errorf ~loc "Only record/variant types can be registered"

(** Lowering state *)
type state = {
  mutable next_var_id : int;
  mutable declarations : Kirc_Ast.k_ext list;
  fun_map : (string, tparam list * texpr) Hashtbl.t;
}

let create_state fun_map = {next_var_id = 0; declarations = []; fun_map}

(** Generate a fresh variable ID *)
let fresh_id state =
  let id = state.next_var_id in
  state.next_var_id <- id + 1 ;
  id

(** Convert memspace to Kirc_Ast memspace *)
let lower_memspace = function
  | Local -> Kirc_Ast.LocalSpace
  | Shared -> Kirc_Ast.Shared
  | Global -> Kirc_Ast.Global

(** Convert prim_type to Kirc_Ast elttype *)
let lower_prim_elttype = function
  | TInt32 -> Kirc_Ast.EInt32
  | TUnit | TBool -> Kirc_Ast.EInt32 (* bool as int *)

(** Convert registered type to Kirc_Ast elttype *)
let lower_reg_elttype = function
  | "int64" -> Kirc_Ast.EInt64
  | "float32" -> Kirc_Ast.EFloat32
  | "float64" -> Kirc_Ast.EFloat64
  | _ -> Kirc_Ast.EInt32 (* Default *)

(** Get element type from a typ *)
let rec elttype_of_typ t =
  match repr t with
  | TPrim p -> lower_prim_elttype p
  | TReg r -> lower_reg_elttype r
  | TVec t -> elttype_of_typ t
  | TArr (t, _) -> elttype_of_typ t
  | _ -> Kirc_Ast.EInt32 (* Default *)

(** Lower a typed expression to Kirc_Ast *)
let rec lower_expr (state : state) (te : texpr) : Kirc_Ast.k_ext =
  match te.te with
  (* Literals *)
  | TEUnit -> Kirc_Ast.Unit
  | TEBool b -> Kirc_Ast.Int (if b then 1 else 0)
  | TEInt i -> Kirc_Ast.Int i
  | TEInt32 i -> Kirc_Ast.Int (Int32.to_int i)
  | TEInt64 i -> Kirc_Ast.Int (Int64.to_int i) (* Note: may lose precision *)
  | TEFloat f -> Kirc_Ast.Float f
  | TEDouble f -> Kirc_Ast.Double f
  (* Variables - use IntId for variable references in expressions *)
  | TEVar (name, id) -> (
      match repr te.ty with
      | TFun _ -> Kirc_Ast.IdName name
      | _ -> lower_ref id name te.ty)
  (* Vector access *)
  | TEVecGet (vec, idx) ->
      Kirc_Ast.IntVecAcc (lower_expr state vec, lower_expr state idx)
  (* Vector set *)
  | TEVecSet (vec, idx, value) ->
      Kirc_Ast.SetV
        ( Kirc_Ast.IntVecAcc (lower_expr state vec, lower_expr state idx),
          lower_expr state value )
  (* Array access *)
  | TEArrGet (arr, idx) ->
      Kirc_Ast.IntVecAcc (lower_expr state arr, lower_expr state idx)
  (* Array set *)
  | TEArrSet (arr, idx, value) ->
      Kirc_Ast.SetV
        ( Kirc_Ast.IntVecAcc (lower_expr state arr, lower_expr state idx),
          lower_expr state value )
  (* Field access *)
  | TEFieldGet (record, field, _idx) ->
      Kirc_Ast.RecGet (lower_expr state record, field)
  (* Field set *)
  | TEFieldSet (record, field, _idx, value) ->
      Kirc_Ast.RecSet
        ( Kirc_Ast.RecGet (lower_expr state record, field),
          lower_expr state value )
  (* Binary operations - dispatch based on type *)
  | TEBinop (op, e1, e2) -> lower_binop state op e1 e2 te.ty
  (* Unary operations *)
  | TEUnop (op, e) -> lower_unop state op e te.ty
  (* Function application *)
  | TEApp (fn, args) -> (
      let args_ir = Array.of_list (List.map (lower_expr state) args) in
      match fn.te with
      | TEVar (name, _) when Hashtbl.mem state.fun_map name ->
          let params, body = Hashtbl.find state.fun_map name in
          let ret_ty = repr body.ty in
          let ret_str =
            match ret_ty with
            | TPrim TInt32 -> "int"
            | TReg "int64" -> "long"
            | TReg "float32" -> "float"
            | TReg "float64" -> "double"
            | TPrim TUnit -> "void"
            | TRecord (n, _) -> "struct " ^ n ^ "_sarek"
            | _ -> "int"
          in
          let params_ir = lower_params params in
          let fun_body_ir = lower_expr state body in
          let fun_body_ir =
            match fun_body_ir with
            | Kirc_Ast.Return _ -> fun_body_ir
            | _ -> Kirc_Ast.Return fun_body_ir
          in
          let fun_ir = Kirc_Ast.Kern (Kirc_Ast.Params params_ir, fun_body_ir) in
          Kirc_Ast.App (Kirc_Ast.GlobalFun (fun_ir, ret_str, name), args_ir)
      | _ ->
          let fn_ir = lower_expr state fn in
          Kirc_Ast.App (fn_ir, args_ir))
  (* Mutable assignment *)
  | TEAssign (name, id, value) ->
      let value_ir = lower_expr state value in
      let ref_ir = lower_ref id name value.ty in
      Kirc_Ast.Set (ref_ir, value_ir)
  (* Let binding *)
  | TELet (name, _id, value, body) -> (
      match value.te with
      (* Special case: create_array - the Arr node IS the declaration *)
      | TECreateArray (size, elem_ty, mem) ->
          let size_ir = lower_expr state size in
          let elt =
            match repr elem_ty with
            | TPrim p -> lower_prim_elttype p
            | TReg r -> lower_reg_elttype r
            | _ -> Kirc_Ast.EInt32
          in
          let arr_ir = Kirc_Ast.Arr (name, size_ir, elt, lower_memspace mem) in
          let body_ir = lower_expr state body in
          Kirc_Ast.Local (arr_ir, body_ir)
      (* Normal let binding *)
      | _ ->
          let value_ir = lower_expr state value in
          let decl_ir = lower_decl ~mutable_:false _id name value.ty in
          let ref_ir = lower_ref _id name value.ty in
          let body_ir = lower_expr state body in
          Kirc_Ast.Seq
            ( Kirc_Ast.Decl decl_ir,
              Kirc_Ast.Seq (Kirc_Ast.Set (ref_ir, value_ir), body_ir) ))
  (* Mutable let binding - same as regular let in IR *)
  | TELetMut (name, id, value, body) ->
      let value_ir = lower_expr state value in
      let decl_ir = lower_decl ~mutable_:true id name value.ty in
      let ref_ir = lower_ref id name value.ty in
      let body_ir = lower_expr state body in
      Kirc_Ast.Seq
        ( Kirc_Ast.Decl decl_ir,
          Kirc_Ast.Seq (Kirc_Ast.Set (ref_ir, value_ir), body_ir) )
  (* If-then-else *)
  | TEIf (cond, then_e, else_opt) -> (
      let cond_ir = lower_expr state cond in
      let then_ir = lower_expr state then_e in
      match else_opt with
      | None -> Kirc_Ast.If (cond_ir, then_ir)
      | Some else_e ->
          let else_ir = lower_expr state else_e in
          Kirc_Ast.Ife (cond_ir, then_ir, else_ir))
  (* For loop *)
  | TEFor (var, id, lo, hi, dir, body) ->
      let lo_ir = lower_expr state lo in
      let hi_ir = lower_expr state hi in
      let var_ir = lower_ref id var te.ty in
      let body_ir = lower_expr state body in
      (* DoLoop takes: var, start, end, body *)
      (* For downto, we need to handle differently - for now assume upto *)
      let _ = dir in
      Kirc_Ast.DoLoop (var_ir, lo_ir, hi_ir, body_ir)
  (* While loop *)
  | TEWhile (cond, body) ->
      let cond_ir = lower_expr state cond in
      let body_ir = lower_expr state body in
      Kirc_Ast.While (cond_ir, body_ir)
  (* Sequence *)
  | TESeq exprs -> lower_seq state exprs
  (* Match *)
  | TEMatch (scrutinee, cases) -> lower_match state scrutinee cases
  (* Record construction *)
  | TERecord (type_name, fields) ->
      let field_irs = List.map (fun (_, e) -> lower_expr state e) fields in
      Kirc_Ast.Record (mangle_type_name type_name ^ "_sarek", field_irs)
  (* Constructor application *)
  | TEConstr (type_name, constr_name, arg_opt) ->
      let args =
        match arg_opt with None -> [] | Some arg -> [lower_expr state arg]
      in
      Kirc_Ast.Constr (mangle_type_name type_name, constr_name, args)
  (* Tuple - represented as anonymous record *)
  | TETuple exprs ->
      let exprs_ir = List.map (lower_expr state) exprs in
      Kirc_Ast.Record ("tuple", exprs_ir)
  (* Return *)
  | TEReturn e -> Kirc_Ast.Return (lower_expr state e)
  (* Create array *)
  | TECreateArray (size, elem_ty, mem) ->
      let size_ir = lower_expr state size in
      let elt =
        match repr elem_ty with
        | TPrim p -> lower_prim_elttype p
        | TReg r -> lower_reg_elttype r
        | _ -> Kirc_Ast.EInt32
      in
      let name = "local_array_" ^ string_of_int (fresh_id state) in
      Kirc_Ast.Arr (name, size_ir, elt, lower_memspace mem)
  (* Global reference - use *Var variants to preserve name for quoting *)
  | TEGlobalRef (name, ty) -> (
      match repr ty with
      | TPrim TInt32 -> Kirc_Ast.GIntVar name
      | TReg "int64" -> Kirc_Ast.GIntVar name
      | TReg "float32" -> Kirc_Ast.GFloatVar name
      | TReg "float64" -> Kirc_Ast.GFloat64Var name
      | _ -> Kirc_Ast.GIntVar name)
  (* Native code - use NativeVar to preserve string for quoting *)
  | TENative s -> Kirc_Ast.NativeVar s
  (* Native code - function expression, store for quoting *)
  | TENativeFun func_expr -> Kirc_Ast.NativeFunExpr func_expr
  (* Pragma *)
  | TEPragma (opts, body) -> Kirc_Ast.Pragma (opts, lower_expr state body)
  (* BSP constructs *)
  | TELetShared (name, _id, elem_ty, size_opt, body) ->
      (* Lower the size: use block_dim_x if not specified *)
      let size_ir =
        match size_opt with
        | Some size -> lower_expr state size
        | None -> Kirc_Ast.IntrinsicRef (["Gpu"], "block_dim_x")
      in
      let elt = elttype_of_typ elem_ty in
      let arr_ir = Kirc_Ast.Arr (name, size_ir, elt, Kirc_Ast.Shared) in
      let body_ir = lower_expr state body in
      Kirc_Ast.Local (arr_ir, body_ir)
  | TESuperstep (_name, _divergent, step_body, cont) ->
      (* Lower body, then emit barrier, then lower continuation *)
      let body_ir = lower_expr state step_body in
      let barrier_ir = Kirc_Ast.IntrinsicRef (["Gpu"], "block_barrier") in
      let barrier_call = Kirc_Ast.App (barrier_ir, [|Kirc_Ast.Unit|]) in
      let cont_ir = lower_expr state cont in
      Kirc_Ast.Seq (body_ir, Kirc_Ast.Seq (barrier_call, cont_ir))
  (* Intrinsic constant - emit IntrinsicRef, device code resolved at JIT *)
  | TEIntrinsicConst ref -> (
      match ref with
      | Sarek_env.IntrinsicRef (path, name) -> Kirc_Ast.IntrinsicRef (path, name)
      | Sarek_env.CorePrimitiveRef name ->
          (* Core primitives use Gpu module path for registry lookup *)
          Kirc_Ast.IntrinsicRef (["Gpu"], name))
  (* Intrinsic function call - emit IntrinsicRef, device code resolved at JIT *)
  | TEIntrinsicFun (ref, _convergence, args) ->
      (* Filter out Unit arguments - they're just () for function application syntax *)
      let non_unit_args =
        List.filter
          (fun arg -> match arg.te with TEUnit -> false | _ -> true)
          args
      in
      let args_ir = Array.of_list (List.map (lower_expr state) non_unit_args) in
      let path, name =
        match ref with
        | Sarek_env.IntrinsicRef (path, name) -> (path, name)
        | Sarek_env.CorePrimitiveRef name ->
            (* Core primitives use Gpu module path for registry lookup *)
            (["Gpu"], name)
      in
      if Array.length args_ir = 0 then Kirc_Ast.IntrinsicRef (path, name)
      else Kirc_Ast.App (Kirc_Ast.IntrinsicRef (path, name), args_ir)

(** Lower a declaration for a local/kernel variable. *)
and lower_decl ~mutable_ id name ty =
  match repr ty with
  | TPrim TInt32 -> Kirc_Ast.IntVar (id, name, mutable_)
  | TReg "int64" -> Kirc_Ast.IntVar (id, name, mutable_)
  | TReg "float32" -> Kirc_Ast.FloatVar (id, name, mutable_)
  | TReg "float64" -> Kirc_Ast.DoubleVar (id, name, mutable_)
  | TPrim TBool -> Kirc_Ast.BoolVar (id, name, mutable_)
  | TPrim TUnit -> Kirc_Ast.UnitVar (id, name, mutable_)
  | TVec _ -> Kirc_Ast.VecVar (Kirc_Ast.Empty, id, name)
  | TRecord (type_name, _) ->
      Kirc_Ast.Custom (mangle_type_name type_name, id, name)
  | TVariant (type_name, _) ->
      Kirc_Ast.Custom (mangle_type_name type_name, id, name)
  | _ -> Kirc_Ast.IntVar (id, name, mutable_)

(** Lower a reference to a previously-declared variable. *)
and lower_ref id name _ty = Kirc_Ast.IntId (name, id)

(** Lower binary operation based on operand types *)
and lower_binop state op e1 e2 _result_ty =
  let ir1 = lower_expr state e1 in
  let ir2 = lower_expr state e2 in
  let is_float =
    match repr e1.ty with TReg ("float32" | "float64") -> true | _ -> false
  in
  match op with
  | Add ->
      if is_float then Kirc_Ast.Plusf (ir1, ir2) else Kirc_Ast.Plus (ir1, ir2)
  | Sub -> if is_float then Kirc_Ast.Minf (ir1, ir2) else Kirc_Ast.Min (ir1, ir2)
  | Mul -> if is_float then Kirc_Ast.Mulf (ir1, ir2) else Kirc_Ast.Mul (ir1, ir2)
  | Div -> if is_float then Kirc_Ast.Divf (ir1, ir2) else Kirc_Ast.Div (ir1, ir2)
  | Mod -> Kirc_Ast.Mod (ir1, ir2)
  | Eq -> Kirc_Ast.EqBool (ir1, ir2)
  | Ne -> Kirc_Ast.Not (Kirc_Ast.EqBool (ir1, ir2))
  | Lt -> Kirc_Ast.LtBool (ir1, ir2)
  | Le -> Kirc_Ast.LtEBool (ir1, ir2)
  | Gt -> Kirc_Ast.GtBool (ir1, ir2)
  | Ge -> Kirc_Ast.GtEBool (ir1, ir2)
  | And -> Kirc_Ast.And (ir1, ir2)
  | Or -> Kirc_Ast.Or (ir1, ir2)
  | Land -> Kirc_Ast.And (ir1, ir2) (* Bitwise and for ints *)
  | Lor -> Kirc_Ast.Or (ir1, ir2)
  | Lxor ->
      Kirc_Ast.App (Kirc_Ast.Intrinsics ("spoc_xor", "spoc_xor"), [|ir1; ir2|])
  | Lsl | Lsr | Asr ->
      (* Shift operations - need to implement *)
      ir1
(* Placeholder *)

(** Lower unary operation *)
and lower_unop state op e _result_ty =
  let ir = lower_expr state e in
  let is_float =
    match repr e.ty with TReg ("float32" | "float64") -> true | _ -> false
  in
  match op with
  | Neg ->
      if is_float then Kirc_Ast.Minf (Kirc_Ast.Float 0.0, ir)
      else Kirc_Ast.Min (Kirc_Ast.Int 0, ir)
  | Not -> Kirc_Ast.Not ir
  | Lnot -> Kirc_Ast.Not ir (* Bitwise not as logical for now *)

(** Lower a sequence of expressions *)
and lower_seq state exprs =
  match exprs with
  | [] -> Kirc_Ast.Unit
  | [e] -> lower_expr state e
  | e :: rest ->
      let ir = lower_expr state e in
      Kirc_Ast.Seq (ir, lower_seq state rest)

(** Lower a match expression *)
and lower_match state scrutinee cases =
  let scr_ir = lower_expr state scrutinee in
  let type_name =
    match repr scrutinee.ty with
    | TVariant (n, _) -> mangle_type_name n
    | _ -> "match_type"
  in
  let cases_ir =
    Array.of_list
      (List.mapi
         (fun i (pat, body) ->
           let body_ir = lower_expr state body in
           let case_info =
             match pat.tpat with
             | TPConstr (_, name, arg_opt) ->
                 let arg_info =
                   match arg_opt with
                   | None -> None
                   | Some ap -> (
                       match ap.tpat with
                       | TPVar (vname, vid) ->
                           let arg_typ = c_type_of_typ ap.tpat_ty in
                           Some (arg_typ, name, vid, vname)
                       | _ -> None)
                 in
                 (i, arg_info, body_ir)
             | TPVar (name, id) ->
                 (i, Some (type_name, "var", id, name), body_ir)
             | TPAny -> (i, None, body_ir)
             | TPTuple _ -> (i, None, body_ir)
             (* TODO: handle tuple patterns *)
           in
           case_info)
         cases)
  in
  Kirc_Ast.Match (type_name, scr_ir, cases_ir)

(** Lower a kernel parameter to IR *)
and lower_param (p : tparam) : Kirc_Ast.k_ext =
  match repr p.tparam_type with
  | TVec elem_ty ->
      let elem_ir =
        match repr elem_ty with
        | TPrim TInt32 -> Kirc_Ast.Int 0
        | TReg "int64" -> Kirc_Ast.Int 0
        | TReg "float32" -> Kirc_Ast.Float 0.0
        | TReg "float64" -> Kirc_Ast.Double 0.0
        | TRecord (type_name, _) | TVariant (type_name, _) ->
            Kirc_Ast.Custom (mangle_type_name type_name, 0, p.tparam_name)
        | _ -> Kirc_Ast.Int 0
      in
      Kirc_Ast.VecVar (elem_ir, p.tparam_id, p.tparam_name)
  | TPrim TInt32 -> Kirc_Ast.IntVar (p.tparam_id, p.tparam_name, false)
  | TReg "int64" -> Kirc_Ast.IntVar (p.tparam_id, p.tparam_name, false)
  | TReg "float32" -> Kirc_Ast.FloatVar (p.tparam_id, p.tparam_name, false)
  | TReg "float64" -> Kirc_Ast.DoubleVar (p.tparam_id, p.tparam_name, false)
  | TPrim TBool -> Kirc_Ast.BoolVar (p.tparam_id, p.tparam_name, false)
  | TPrim TUnit -> Kirc_Ast.UnitVar (p.tparam_id, p.tparam_name, false)
  | TRecord (type_name, _) | TVariant (type_name, _) ->
      Kirc_Ast.Custom (mangle_type_name type_name, p.tparam_id, p.tparam_name)
  | _ -> Kirc_Ast.IntVar (p.tparam_id, p.tparam_name, false)

(** Lower kernel parameters to IR params *)
and lower_params (params : tparam list) : Kirc_Ast.k_ext =
  (* Build right-associative Concat: Concat(p1, Concat(p2, Concat(p3, Empty))) *)
  List.fold_right
    (fun p acc -> Kirc_Ast.Concat (lower_param p, acc))
    params
    Kirc_Ast.Empty

(** Lower a complete kernel *)
let lower_kernel (kernel : tkernel) : Kirc_Ast.k_ext * string list =
  let fun_map = Hashtbl.create 8 in
  List.iter
    (function
      | TMFun (name, params, body) -> Hashtbl.replace fun_map name (params, body)
      | _ -> ())
    kernel.tkern_module_items ;
  let state = create_state fun_map in
  let params_ir = Kirc_Ast.Params (lower_params kernel.tkern_params) in
  let module_items_ir =
    List.fold_right
      (fun item acc ->
        match item with
        | TMConst (name, id, ty, expr) ->
            let decl = lower_decl ~mutable_:false id name ty in
            let setv =
              Kirc_Ast.Set (Kirc_Ast.IntId (name, id), lower_expr state expr)
            in
            Kirc_Ast.Seq (Kirc_Ast.Decl decl, Kirc_Ast.Seq (setv, acc))
        | TMFun (_name, _params, _body) -> acc)
      kernel.tkern_module_items
      Kirc_Ast.Empty
  in
  let constructors =
    List.concat
      (List.map
         (function
           | TTypeRecord {tdecl_name; tdecl_fields; _} ->
               record_constructor_strings tdecl_name tdecl_fields
           | TTypeVariant {tdecl_name; tdecl_constructors; _} ->
               variant_constructor_strings tdecl_name tdecl_constructors)
         kernel.tkern_type_decls)
  in
  let body_ir = lower_expr state kernel.tkern_body in
  ( Kirc_Ast.Kern (params_ir, Kirc_Ast.Seq (module_items_ir, body_ir)),
    constructors )

(** Get the return value IR from a kernel *)
let lower_return_value (kernel : tkernel) : Kirc_Ast.k_ext =
  match repr kernel.tkern_return_type with
  | TPrim TUnit -> Kirc_Ast.Unit
  | TPrim TInt32 -> Kirc_Ast.IntVar (0, "result", true)
  | TReg "int64" -> Kirc_Ast.IntVar (0, "result", true)
  | TReg "float32" -> Kirc_Ast.FloatVar (0, "result", true)
  | TReg "float64" -> Kirc_Ast.DoubleVar (0, "result", true)
  | TVec elem_ty ->
      let elem =
        match repr elem_ty with
        | TPrim TInt32 -> Kirc_Ast.Int 0
        | TReg "float32" -> Kirc_Ast.Float 0.0
        | TReg "float64" -> Kirc_Ast.Double 0.0
        | _ -> Kirc_Ast.Int 0
      in
      Kirc_Ast.VecVar (elem, 0, "result")
  | _ -> Kirc_Ast.Unit
