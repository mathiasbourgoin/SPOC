(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module quotes Kirc_Ast values back to OCaml AST, so the IR can be
 * embedded in the generated code.
 ******************************************************************************)

open Ppxlib
open Sarek_typed_ast
open Sarek_types

(** Helper to create an identifier expression *)
let evar ~loc name =
  Ast_builder.Default.pexp_ident ~loc {txt = Lident name; loc}

let evar_qualified ~loc path name =
  let lid =
    List.fold_left
      (fun acc m -> Ldot (acc, m))
      (Lident (List.hd path))
      (List.tl path @ [name])
  in
  Ast_builder.Default.pexp_ident ~loc {txt = lid; loc}

(** Quote an int as OCaml expression *)
let quote_int ~loc i = Ast_builder.Default.eint ~loc i

(** Quote an int32 as OCaml expression *)
let quote_int32 ~loc i =
  [%expr [%e Ast_builder.Default.eint ~loc (Int32.to_int i)]]

(** Quote a float as OCaml expression *)
let quote_float ~loc f = Ast_builder.Default.efloat ~loc (string_of_float f)

(** Quote a string as OCaml expression *)
let quote_string ~loc s = Ast_builder.Default.estring ~loc s

(** Quote a bool as OCaml expression *)
let quote_bool ~loc b = if b then [%expr true] else [%expr false]

(** Quote an elttype *)
let quote_elttype ~loc (e : Kirc_Ast.elttype) : expression =
  match e with
  | Kirc_Ast.EInt32 -> [%expr Sarek.Kirc_Ast.EInt32]
  | Kirc_Ast.EInt64 -> [%expr Sarek.Kirc_Ast.EInt64]
  | Kirc_Ast.EFloat32 -> [%expr Sarek.Kirc_Ast.EFloat32]
  | Kirc_Ast.EFloat64 -> [%expr Sarek.Kirc_Ast.EFloat64]

(** Quote a memspace *)
let quote_memspace ~loc (m : Kirc_Ast.memspace) : expression =
  match m with
  | Kirc_Ast.LocalSpace -> [%expr Sarek.Kirc_Ast.LocalSpace]
  | Kirc_Ast.Global -> [%expr Sarek.Kirc_Ast.Global]
  | Kirc_Ast.Shared -> [%expr Sarek.Kirc_Ast.Shared]

(** Quote a list *)
let quote_list ~loc quote_elem elems =
  List.fold_right
    (fun elem acc -> [%expr [%e quote_elem ~loc elem] :: [%e acc]])
    elems
    [%expr []]

(** Quote an array *)
let quote_array ~loc quote_elem elems =
  [%expr Array.of_list [%e quote_list ~loc quote_elem (Array.to_list elems)]]

(** Quote an option *)
let quote_option ~loc quote_elem = function
  | None -> [%expr None]
  | Some x -> [%expr Some [%e quote_elem ~loc x]]

(** Quote a case *)
let rec quote_case ~loc ((i, opt, body) : Kirc_Ast.case) : expression =
  let opt_expr =
    match opt with
    | None -> [%expr None]
    | Some (s1, s2, id, s3) ->
        [%expr
          Some
            ( [%e quote_string ~loc s1],
              [%e quote_string ~loc s2],
              [%e quote_int ~loc id],
              [%e quote_string ~loc s3] )]
  in
  [%expr [%e quote_int ~loc i], [%e opt_expr], [%e quote_k_ext ~loc body]]

(** Quote a Kirc_Ast.k_ext value to OCaml expression *)
and quote_k_ext ~loc (k : Kirc_Ast.k_ext) : expression =
  match k with
  | Kirc_Ast.Kern (params, body) ->
      [%expr
        Sarek.Kirc_Ast.Kern
          ([%e quote_k_ext ~loc params], [%e quote_k_ext ~loc body])]
  | Kirc_Ast.Block b -> [%expr Sarek.Kirc_Ast.Block [%e quote_k_ext ~loc b]]
  | Kirc_Ast.Params p -> [%expr Sarek.Kirc_Ast.Params [%e quote_k_ext ~loc p]]
  | Kirc_Ast.Plus (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Plus ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Plusf (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Plusf ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Min (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Min ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Minf (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Minf ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Mul (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Mul ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Mulf (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Mulf ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Div (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Div ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Divf (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Divf ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Mod (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Mod ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Id s -> [%expr Sarek.Kirc_Ast.Id [%e quote_string ~loc s]]
  | Kirc_Ast.IdName s -> [%expr Sarek.Kirc_Ast.IdName [%e quote_string ~loc s]]
  | Kirc_Ast.GlobalFun (body, ret_type, name) ->
      [%expr
        Sarek.Kirc_Ast.GlobalFun
          ( [%e quote_k_ext ~loc body],
            [%e quote_string ~loc ret_type],
            [%e quote_string ~loc name] )]
  | Kirc_Ast.IntVar (id, name, is_mut) ->
      [%expr
        Sarek.Kirc_Ast.IntVar
          ( [%e quote_int ~loc id],
            [%e quote_string ~loc name],
            [%e quote_bool ~loc is_mut] )]
  | Kirc_Ast.FloatVar (id, name, is_mut) ->
      [%expr
        Sarek.Kirc_Ast.FloatVar
          ( [%e quote_int ~loc id],
            [%e quote_string ~loc name],
            [%e quote_bool ~loc is_mut] )]
  | Kirc_Ast.UnitVar (id, name, is_mut) ->
      [%expr
        Sarek.Kirc_Ast.UnitVar
          ( [%e quote_int ~loc id],
            [%e quote_string ~loc name],
            [%e quote_bool ~loc is_mut] )]
  | Kirc_Ast.CastDoubleVar (id, name) ->
      [%expr
        Sarek.Kirc_Ast.CastDoubleVar
          ([%e quote_int ~loc id], [%e quote_string ~loc name])]
  | Kirc_Ast.DoubleVar (id, name, is_mut) ->
      [%expr
        Sarek.Kirc_Ast.DoubleVar
          ( [%e quote_int ~loc id],
            [%e quote_string ~loc name],
            [%e quote_bool ~loc is_mut] )]
  | Kirc_Ast.BoolVar (id, name, is_mut) ->
      [%expr
        Sarek.Kirc_Ast.BoolVar
          ( [%e quote_int ~loc id],
            [%e quote_string ~loc name],
            [%e quote_bool ~loc is_mut] )]
  | Kirc_Ast.VecVar (elem, id, name) ->
      [%expr
        Sarek.Kirc_Ast.VecVar
          ( [%e quote_k_ext ~loc elem],
            [%e quote_int ~loc id],
            [%e quote_string ~loc name] )]
  | Kirc_Ast.Arr (name, size, elt, mem) ->
      [%expr
        Sarek.Kirc_Ast.Arr
          ( [%e quote_string ~loc name],
            [%e quote_k_ext ~loc size],
            [%e quote_elttype ~loc elt],
            [%e quote_memspace ~loc mem] )]
  | Kirc_Ast.Concat (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Concat ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Constr (type_name, constr_name, args) ->
      [%expr
        Sarek.Kirc_Ast.Constr
          ( [%e quote_string ~loc type_name],
            [%e quote_string ~loc constr_name],
            [%e quote_list ~loc quote_k_ext args] )]
  | Kirc_Ast.Record (type_name, fields) ->
      [%expr
        Sarek.Kirc_Ast.Record
          ( [%e quote_string ~loc type_name],
            [%e quote_list ~loc quote_k_ext fields] )]
  | Kirc_Ast.RecGet (record, field) ->
      [%expr
        Sarek.Kirc_Ast.RecGet
          ([%e quote_k_ext ~loc record], [%e quote_string ~loc field])]
  | Kirc_Ast.RecSet (record, value) ->
      [%expr
        Sarek.Kirc_Ast.RecSet
          ([%e quote_k_ext ~loc record], [%e quote_k_ext ~loc value])]
  | Kirc_Ast.Empty -> [%expr Sarek.Kirc_Ast.Empty]
  | Kirc_Ast.Seq (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Seq ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Return k -> [%expr Sarek.Kirc_Ast.Return [%e quote_k_ext ~loc k]]
  | Kirc_Ast.Set (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Set ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Decl k -> [%expr Sarek.Kirc_Ast.Decl [%e quote_k_ext ~loc k]]
  | Kirc_Ast.SetV (a, b) ->
      [%expr
        Sarek.Kirc_Ast.SetV ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.SetLocalVar (a, b, c) ->
      [%expr
        Sarek.Kirc_Ast.SetLocalVar
          ( [%e quote_k_ext ~loc a],
            [%e quote_k_ext ~loc b],
            [%e quote_k_ext ~loc c] )]
  | Kirc_Ast.Intrinsics (cuda, opencl) ->
      [%expr
        Sarek.Kirc_Ast.Intrinsics
          ([%e quote_string ~loc cuda], [%e quote_string ~loc opencl])]
  | Kirc_Ast.IntId (name, id) ->
      [%expr
        Sarek.Kirc_Ast.IntId
          ([%e quote_string ~loc name], [%e quote_int ~loc id])]
  | Kirc_Ast.Int i -> [%expr Sarek.Kirc_Ast.Int [%e quote_int ~loc i]]
  | Kirc_Ast.Float f -> [%expr Sarek.Kirc_Ast.Float [%e quote_float ~loc f]]
  | Kirc_Ast.Double f -> [%expr Sarek.Kirc_Ast.Double [%e quote_float ~loc f]]
  | Kirc_Ast.Custom (type_name, id, name) ->
      [%expr
        Sarek.Kirc_Ast.Custom
          ( [%e quote_string ~loc type_name],
            [%e quote_int ~loc id],
            [%e quote_string ~loc name] )]
  | Kirc_Ast.CustomVar (type_name, constr_name, var_name) ->
      [%expr
        Sarek.Kirc_Ast.CustomVar
          ( [%e quote_string ~loc type_name],
            [%e quote_string ~loc constr_name],
            [%e quote_string ~loc var_name] )]
  | Kirc_Ast.IntVecAcc (vec, idx) ->
      [%expr
        Sarek.Kirc_Ast.IntVecAcc
          ([%e quote_k_ext ~loc vec], [%e quote_k_ext ~loc idx])]
  | Kirc_Ast.Local (decl, body) ->
      [%expr
        Sarek.Kirc_Ast.Local
          ([%e quote_k_ext ~loc decl], [%e quote_k_ext ~loc body])]
  | Kirc_Ast.Acc (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Acc ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Ife (cond, then_e, else_e) ->
      [%expr
        Sarek.Kirc_Ast.Ife
          ( [%e quote_k_ext ~loc cond],
            [%e quote_k_ext ~loc then_e],
            [%e quote_k_ext ~loc else_e] )]
  | Kirc_Ast.If (cond, then_e) ->
      [%expr
        Sarek.Kirc_Ast.If
          ([%e quote_k_ext ~loc cond], [%e quote_k_ext ~loc then_e])]
  | Kirc_Ast.Match (type_name, scrutinee, cases) ->
      [%expr
        Sarek.Kirc_Ast.Match
          ( [%e quote_string ~loc type_name],
            [%e quote_k_ext ~loc scrutinee],
            [%e quote_array ~loc quote_case cases] )]
  | Kirc_Ast.Or (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Or ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.And (a, b) ->
      [%expr
        Sarek.Kirc_Ast.And ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Not k -> [%expr Sarek.Kirc_Ast.Not [%e quote_k_ext ~loc k]]
  | Kirc_Ast.EqCustom (type_name, a, b) ->
      [%expr
        Sarek.Kirc_Ast.EqCustom
          ( [%e quote_string ~loc type_name],
            [%e quote_k_ext ~loc a],
            [%e quote_k_ext ~loc b] )]
  | Kirc_Ast.EqBool (a, b) ->
      [%expr
        Sarek.Kirc_Ast.EqBool ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.LtBool (a, b) ->
      [%expr
        Sarek.Kirc_Ast.LtBool ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.GtBool (a, b) ->
      [%expr
        Sarek.Kirc_Ast.GtBool ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.LtEBool (a, b) ->
      [%expr
        Sarek.Kirc_Ast.LtEBool ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.GtEBool (a, b) ->
      [%expr
        Sarek.Kirc_Ast.GtEBool ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.DoLoop (var, lo, hi, body) ->
      [%expr
        Sarek.Kirc_Ast.DoLoop
          ( [%e quote_k_ext ~loc var],
            [%e quote_k_ext ~loc lo],
            [%e quote_k_ext ~loc hi],
            [%e quote_k_ext ~loc body] )]
  | Kirc_Ast.While (cond, body) ->
      [%expr
        Sarek.Kirc_Ast.While
          ([%e quote_k_ext ~loc cond], [%e quote_k_ext ~loc body])]
  | Kirc_Ast.App (fn, args) ->
      [%expr
        Sarek.Kirc_Ast.App
          ([%e quote_k_ext ~loc fn], [%e quote_array ~loc quote_k_ext args])]
  | Kirc_Ast.GInt _ ->
      (* This shouldn't be reached in PPX - we use GIntVar instead *)
      [%expr Sarek.Kirc_Ast.GInt (fun () -> 0l)]
  | Kirc_Ast.GFloat _ -> [%expr Sarek.Kirc_Ast.GFloat (fun () -> 0.0)]
  | Kirc_Ast.GFloat64 _ -> [%expr Sarek.Kirc_Ast.GFloat64 (fun () -> 0.0)]
  (* GIntVar, GFloatVar, GFloat64Var - generate closures that capture the variable *)
  (* These variables are refs, so we dereference them with ! *)
  | Kirc_Ast.GIntVar name ->
      let var_expr = evar ~loc name in
      [%expr Sarek.Kirc_Ast.GInt (fun () -> ![%e var_expr])]
  | Kirc_Ast.GFloatVar name ->
      let var_expr = evar ~loc name in
      [%expr Sarek.Kirc_Ast.GFloat (fun () -> ![%e var_expr])]
  | Kirc_Ast.GFloat64Var name ->
      let var_expr = evar ~loc name in
      [%expr Sarek.Kirc_Ast.GFloat64 (fun () -> ![%e var_expr])]
  | Kirc_Ast.Native _ ->
      (* This shouldn't be reached in PPX - we use NativeVar instead *)
      [%expr Sarek.Kirc_Ast.Native (fun _dev -> "")]
  | Kirc_Ast.NativeVar code ->
      (* Native code is a string literal, not a variable reference *)
      [%expr Sarek.Kirc_Ast.Native (fun _dev -> [%e quote_string ~loc code])]
  | Kirc_Ast.NativeFunExpr func_expr ->
      (* Native function expression - pass it through directly to Native *)
      [%expr Sarek.Kirc_Ast.Native [%e func_expr]]
  | Kirc_Ast.Pragma (opts, body) ->
      [%expr
        Sarek.Kirc_Ast.Pragma
          ([%e quote_list ~loc quote_string opts], [%e quote_k_ext ~loc body])]
  | Kirc_Ast.Map (f, a, b) ->
      [%expr
        Sarek.Kirc_Ast.Map
          ( [%e quote_k_ext ~loc f],
            [%e quote_k_ext ~loc a],
            [%e quote_k_ext ~loc b] )]
  | Kirc_Ast.IntrinsicRef (path, name) ->
      let path_expr =
        Ast_builder.Default.elist
          ~loc
          (List.map (Ast_builder.Default.estring ~loc) path)
      in
      let name_expr = Ast_builder.Default.estring ~loc name in
      [%expr Sarek.Kirc_Ast.IntrinsicRef ([%e path_expr], [%e name_expr])]
  | Kirc_Ast.Unit -> [%expr Sarek.Kirc_Ast.Unit]

(** Build argument handling for spoc_kernel methods *)
let kernel_ctor_lid ~loc name =
  {txt = Ldot (Ldot (Lident "Spoc", "Kernel"), name); loc}

let kernel_ctor_expr ~loc name arg =
  Ast_builder.Default.pexp_construct ~loc (kernel_ctor_lid ~loc name) (Some arg)

let kernel_ctor_pat ~loc name arg =
  Ast_builder.Default.ppat_construct ~loc (kernel_ctor_lid ~loc name) (Some arg)

let core_type_of_typ ~loc (t : typ) : core_type option =
  match repr t with
  | TPrim TUnit -> Some [%type: unit]
  | TPrim (TBool | TInt32) -> Some [%type: int]
  | TReg "int64" -> Some [%type: int]
  | TReg ("float32" | "float64") -> Some [%type: float]
  | TVec elem -> (
      match repr elem with
      | TPrim TInt32 ->
          Some [%type: (int32, Bigarray.int32_elt) Spoc.Vector.vector]
      | TReg "int64" ->
          Some [%type: (int64, Bigarray.int64_elt) Spoc.Vector.vector]
      | TReg "float32" ->
          Some [%type: (float, Bigarray.float32_elt) Spoc.Vector.vector]
      | TReg "float64" ->
          Some [%type: (float, Bigarray.float64_elt) Spoc.Vector.vector]
      | TPrim TBool -> Some [%type: (bool, bool) Spoc.Vector.vector]
      | TRecord _ | TVariant _ ->
          (* Don't add type constraint for custom vectors - let OCaml infer *)
          None
      | _ -> None)
  | TRecord _ | TVariant _ -> None
  | TArr _ | TFun _ | TTuple _ | TVar _ | TReg _ -> None

let kernel_ctor_name (t : typ) : string =
  match repr t with
  | TPrim (TBool | TInt32) -> "Int32"
  | TReg "int64" -> "Int64"
  | TReg "float32" -> "Float32"
  | TReg "float64" -> "Float64"
  | TVec elem -> (
      match repr elem with
      | TPrim (TBool | TInt32) -> "VInt32"
      | TReg "int64" -> "VInt64"
      | TReg "float32" -> "VFloat32"
      | TReg "float64" -> "VFloat64"
      | TRecord _ | TVariant _ -> "VCustom"
      | _ -> "Vector")
  | TRecord _ | TVariant _ -> "Custom"
  | _ -> "Vector"

let kernel_arg_expr ~loc ty var =
  match repr ty with
  | TVec _ ->
      let relaxed = [%expr Spoc.Kernel.relax_vector [%e var]] in
      kernel_ctor_expr ~loc (kernel_ctor_name ty) relaxed
  | _ -> kernel_ctor_expr ~loc (kernel_ctor_name ty) var

let kernel_arg_pat ~loc ty var = kernel_ctor_pat ~loc (kernel_ctor_name ty) var

let build_kernel_args ~loc (params : tparam list) =
  let names = List.mapi (fun idx _ -> Printf.sprintf "spoc_arg%d" idx) params in
  let vars = List.map (fun name -> (name, evar ~loc name)) names in
  let args_pat =
    let pats =
      List.map
        (fun (name, _) -> Ast_builder.Default.ppat_var ~loc {txt = name; loc})
        vars
    in
    match pats with
    | [] -> [%pat? ()]
    | [p] -> p
    | _ -> Ast_builder.Default.ppat_tuple ~loc pats
  in
  let args_array_expr =
    let exprs =
      List.map2
        (fun p (_, v) -> kernel_arg_expr ~loc p.tparam_type v)
        params
        vars
    in
    Ast_builder.Default.pexp_array ~loc exprs
  in
  let list_to_args_pat =
    let pats =
      List.map2
        (fun p (name, _) ->
          kernel_arg_pat
            ~loc
            p.tparam_type
            (Ast_builder.Default.ppat_var ~loc {txt = name; loc}))
        params
        vars
    in
    Ast_builder.Default.ppat_array ~loc pats
  in
  let list_to_args_expr =
    let first_vec = ref false in
    let exprs =
      List.map2
        (fun p (_, v) ->
          let base =
            match repr p.tparam_type with
            | TVec _ ->
                if not !first_vec then (
                  first_vec := true ;
                  v)
                else [%expr Spoc.Kernel.relax_vector [%e v]]
            | _ -> v
          in
          match core_type_of_typ ~loc p.tparam_type with
          | Some ty -> Ast_builder.Default.pexp_constraint ~loc base ty
          | None -> base)
        params
        vars
    in
    match exprs with
    | [] -> [%expr ()]
    | [e] -> e
    | _ -> Ast_builder.Default.pexp_tuple ~loc exprs
  in
  (args_pat, args_array_expr, list_to_args_pat, list_to_args_expr)

(******************************************************************************
 * Intrinsic Reference Collection
 *
 * Collect all intrinsic function references from a kernel and generate
 * a dummy expression that mentions each one. This ensures that if an
 * intrinsic function is missing from the stdlib, compilation will fail.
 ******************************************************************************)

module IntrinsicRefSet = Set.Make (struct
  type t = Sarek_env.intrinsic_ref

  let compare = compare
end)

(** Generate an OCaml expression for an intrinsic reference.

    Intrinsics are now defined in stdlib modules (Float32, Float64, Int32, etc.)
    via %sarek_intrinsic. We reference the function to ensure it exists at
    compile time. The module path enables extensibility: user libraries can
    define their own intrinsics and the PPX will reference them correctly.

    Examples:
    - IntrinsicRef (["Float32"], "sin") -> Float32.sin
    - IntrinsicRef (["Sarek"; "Sarek_prim"], "block_barrier") ->
      Sarek.Sarek_prim.block_barrier *)
let expr_of_intrinsic_ref ~loc (ref : Sarek_env.intrinsic_ref) : expression =
  match ref with
  | Sarek_env.IntrinsicRef (module_path, name) ->
      (* Build longident from module path: ["A"; "B"] + "f" -> A.B.f *)
      let lid =
        List.fold_left
          (fun acc m -> Ldot (acc, m))
          (Lident (List.hd module_path))
          (List.tl module_path @ [name])
      in
      Ast_builder.Default.pexp_ident ~loc {txt = lid; loc}

(** Collect all intrinsic function refs from a typed expression *)
let rec collect_intrinsic_refs (te : texpr) : IntrinsicRefSet.t =
  match te.te with
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEIntrinsicConst _ | TENative _ | TENativeFun _ | TEGlobalRef _ ->
      IntrinsicRefSet.empty
  | TEVecGet (v, i) | TEVecSet (v, i, _) | TEArrGet (v, i) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs v)
        (collect_intrinsic_refs i)
  | TEArrSet (a, i, x) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs a)
        (IntrinsicRefSet.union
           (collect_intrinsic_refs i)
           (collect_intrinsic_refs x))
  | TEFieldGet (r, _, _) -> collect_intrinsic_refs r
  | TEFieldSet (r, _, _, x) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs r)
        (collect_intrinsic_refs x)
  | TEBinop (_, a, b) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs a)
        (collect_intrinsic_refs b)
  | TEUnop (_, a) -> collect_intrinsic_refs a
  | TEApp (f, args) ->
      List.fold_left
        (fun acc arg -> IntrinsicRefSet.union acc (collect_intrinsic_refs arg))
        (collect_intrinsic_refs f)
        args
  | TEAssign (_, _, value) -> collect_intrinsic_refs value
  | TELet (_, _, value, body) | TELetMut (_, _, value, body) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs value)
        (collect_intrinsic_refs body)
  | TEIf (cond, then_e, else_e) -> (
      let base =
        IntrinsicRefSet.union
          (collect_intrinsic_refs cond)
          (collect_intrinsic_refs then_e)
      in
      match else_e with
      | Some e -> IntrinsicRefSet.union base (collect_intrinsic_refs e)
      | None -> base)
  | TEFor (_, _, lo, hi, _, body) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs lo)
        (IntrinsicRefSet.union
           (collect_intrinsic_refs hi)
           (collect_intrinsic_refs body))
  | TEWhile (cond, body) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs cond)
        (collect_intrinsic_refs body)
  | TESeq exprs ->
      List.fold_left
        (fun acc e -> IntrinsicRefSet.union acc (collect_intrinsic_refs e))
        IntrinsicRefSet.empty
        exprs
  | TEMatch (scrutinee, cases) ->
      List.fold_left
        (fun acc (_, body) ->
          IntrinsicRefSet.union acc (collect_intrinsic_refs body))
        (collect_intrinsic_refs scrutinee)
        cases
  | TERecord (_, fields) ->
      List.fold_left
        (fun acc (_, e) -> IntrinsicRefSet.union acc (collect_intrinsic_refs e))
        IntrinsicRefSet.empty
        fields
  | TEConstr (_, _, arg) -> (
      match arg with
      | Some e -> collect_intrinsic_refs e
      | None -> IntrinsicRefSet.empty)
  | TETuple exprs ->
      List.fold_left
        (fun acc e -> IntrinsicRefSet.union acc (collect_intrinsic_refs e))
        IntrinsicRefSet.empty
        exprs
  | TEReturn e | TEPragma (_, e) -> collect_intrinsic_refs e
  | TECreateArray (size, _, _) -> collect_intrinsic_refs size
  | TEIntrinsicFun (_, _, ocaml_ref, args) ->
      let base = IntrinsicRefSet.singleton ocaml_ref in
      List.fold_left
        (fun acc arg -> IntrinsicRefSet.union acc (collect_intrinsic_refs arg))
        base
        args

(** Collect intrinsic refs from module items *)
let collect_from_module_items (items : tmodule_item list) : IntrinsicRefSet.t =
  List.fold_left
    (fun acc item ->
      match item with
      | TMConst (_, _, _, e) ->
          IntrinsicRefSet.union acc (collect_intrinsic_refs e)
      | TMFun (_, _, e) -> IntrinsicRefSet.union acc (collect_intrinsic_refs e))
    IntrinsicRefSet.empty
    items

(** Generate a dummy expression that references all intrinsic functions. This
    ensures compile-time checking that all intrinsics exist in their stdlib
    modules. *)
let generate_intrinsic_check ~loc (kernel : tkernel) : expression =
  let refs_from_body = collect_intrinsic_refs kernel.tkern_body in
  let refs_from_items = collect_from_module_items kernel.tkern_module_items in
  let all_refs = IntrinsicRefSet.union refs_from_body refs_from_items in
  let ref_list = IntrinsicRefSet.elements all_refs in
  let fn_exprs = List.map (expr_of_intrinsic_ref ~loc) ref_list in
  match fn_exprs with
  | [] -> [%expr ()]
  | exprs ->
      (* Generate: let _ = (fn1, fn2, fn3, ...) in () *)
      let tuple_expr =
        match exprs with
        | [e] -> e
        | es -> Ast_builder.Default.pexp_tuple ~loc es
      in
      [%expr
        let _ = [%e tuple_expr] in
        ()]

(** Quote a kernel to create a sarek_kernel expression *)
let quote_kernel ~loc (kernel : tkernel) (ir : Kirc_Ast.k_ext)
    (constructors : string list) (ret_val : Kirc_Ast.k_ext) : expression =
  let args_pat, args_array_expr, list_to_args_pat, list_to_args_expr =
    build_kernel_args ~loc kernel.tkern_params
  in
  [%expr
    let open Spoc in
    let () =
      List.iter
        Sarek.Kirc.register_constructor_string
        [%e
          Ast_builder.Default.elist
            ~loc
            (List.map (Ast_builder.Default.estring ~loc) constructors)]
    in
    let module M = struct
      let exec_fun [%p args_pat] = Spoc.Kernel.exec [%e args_array_expr]

      class ['a, 'b] sarek_kern =
        object
          inherit ['a, 'b] Spoc.Kernel.spoc_kernel "kirc_kernel" "spoc_dummy"

          method exec = exec_fun

          method args_to_list = fun [%p args_pat] -> [%e args_array_expr]

          method list_to_args =
            function
            | [%p list_to_args_pat] -> [%e list_to_args_expr]
            | _ -> failwith "spoc_kernel_extension error"
        end
    end in
    let open Sarek.Kirc in
    let body_ir = [%e quote_k_ext ~loc ir] in
    let ret_ir = [%e quote_k_ext ~loc ret_val] in
    let _intrinsic_check = [%e generate_intrinsic_check ~loc kernel] in
    let kirc_kernel =
      {
        Sarek.Kirc.ml_kern = (fun () -> ());
        Sarek.Kirc.body = body_ir;
        Sarek.Kirc.ret_val = (ret_ir, Spoc.Vector.int32);
        Sarek.Kirc.extensions = [||];
      }
    in
    (new M.sarek_kern, kirc_kernel)]
