(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module generates native OCaml code from Sarek's typed AST. The generated
 * code runs on CPU via Sarek_cpu_runtime for fast execution without GPU.
 *
 * Unlike Sarek_interp which walks the AST at runtime, this generates code
 * at compile time that runs at full native speed.
 ******************************************************************************)

open Ppxlib
open Sarek_typed_ast
open Sarek_types

(** Convert Sarek_ast.loc to Ppxlib.location *)
let ppxlib_loc_of_sarek (l : Sarek_ast.loc) : location =
  let pos_start =
    {
      Lexing.pos_fname = l.loc_file;
      pos_lnum = l.loc_line;
      pos_bol = 0;
      pos_cnum = l.loc_col;
    }
  in
  let pos_end =
    {
      Lexing.pos_fname = l.loc_file;
      pos_lnum = l.loc_end_line;
      pos_bol = 0;
      pos_cnum = l.loc_end_col;
    }
  in
  {loc_start = pos_start; loc_end = pos_end; loc_ghost = false}

(** Helper to create an identifier expression *)
let evar ~loc name =
  Ast_builder.Default.pexp_ident ~loc {txt = Lident name; loc}

(** Helper to create a qualified identifier expression *)
let evar_qualified ~loc path name =
  let lid =
    List.fold_left
      (fun acc m -> Ldot (acc, m))
      (Lident (List.hd path))
      (List.tl path @ [name])
  in
  Ast_builder.Default.pexp_ident ~loc {txt = lid; loc}

(** Create a unique name for a variable by id *)
let var_name id = Printf.sprintf "__v%d" id

(** Create a unique name for a mutable variable by id *)
let mut_var_name id = Printf.sprintf "__m%d" id

(** Thread state variable name - bound in kernel wrapper *)
let state_var = "__state"

(** Shared memory variable name - bound in block wrapper *)
let shared_var = "__shared"

(** {1 Type Mapping} *)

(** Generate a core type from a Sarek type (for type annotations).

    For scalar types (int32, float, etc), we generate explicit types to help
    OCaml infer the correct numeric type.

    For record types, we generate the type name to help OCaml resolve field
    accesses. The type name may be qualified (e.g., "Module.point").

    For vectors and other complex types, we use wildcards. *)
let rec core_type_of_typ ~loc typ : Ppxlib.core_type =
  match repr typ with
  (* Primitive types - only TUnit, TBool, TInt32 exist as primitives *)
  | TPrim TUnit -> [%type: unit]
  | TPrim TBool -> [%type: bool]
  | TPrim TInt32 -> [%type: int32]
  (* Registered types - numeric types are registered by stdlib *)
  | TReg "float32" -> [%type: float]
  | TReg "float64" -> [%type: float]
  | TReg "int32" -> [%type: int32]
  | TReg "int64" -> [%type: int64]
  (* Vector/array types - use wildcard to avoid scope issues *)
  | TVec _ -> [%type: _]
  | TArr _ -> [%type: _]
  (* Record types - for qualified types (Module.type), generate the type name
     to help resolve field accesses. For inline types (no module prefix),
     use wildcard to avoid scope issues. *)
  | TRecord (name, _fields) -> (
      match String.split_on_char '.' name with
      | [_simple_name] ->
          (* Inline type - use wildcard to avoid "type escapes scope" errors *)
          [%type: _]
      | parts ->
          (* Qualified type like "Module.type" - generate the type path *)
          let rec build_lid = function
            | [] -> failwith "empty type name"
            | [x] -> Lident x
            | x :: rest -> Ldot (build_lid rest, x)
          in
          let lid = build_lid (List.rev parts) in
          Ast_builder.Default.ptyp_constr ~loc {txt = lid; loc} [])
  (* Variant types - same as records *)
  | TVariant (name, _constrs) -> (
      match String.split_on_char '.' name with
      | [_simple_name] ->
          (* Inline type - use wildcard *)
          [%type: _]
      | parts ->
          let rec build_lid = function
            | [] -> failwith "empty type name"
            | [x] -> Lident x
            | x :: rest -> Ldot (build_lid rest, x)
          in
          let lid = build_lid (List.rev parts) in
          Ast_builder.Default.ptyp_constr ~loc {txt = lid; loc} [])
  | TTuple tys ->
      Ast_builder.Default.ptyp_tuple ~loc (List.map (core_type_of_typ ~loc) tys)
  | TFun _ | TVar _ | TReg _ -> [%type: _]

(** Map Sarek types to OCaml types for bigarray access *)
let bigarray_kind_of_typ ~loc typ =
  match repr typ with
  | TReg "float32" | TPrim (TBool | TInt32) -> [%expr Bigarray.float32]
  | TReg "float64" -> [%expr Bigarray.float64]
  | TReg "int64" -> [%expr Bigarray.int64]
  | _ -> [%expr Bigarray.float32]
(* Default *)

(** {1 Intrinsic Mapping}

    Map Sarek intrinsics to their OCaml equivalents. For cpu_kern, we call the
    OCaml implementations directly rather than generating GPU code. *)

let gen_intrinsic_const ~loc (ref : Sarek_env.intrinsic_ref) : expression =
  match ref with
  | Sarek_env.CorePrimitiveRef name -> (
      let state = evar ~loc state_var in
      match name with
      | "thread_idx_x" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.thread_idx_x]
      | "thread_idx_y" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.thread_idx_y]
      | "thread_idx_z" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.thread_idx_z]
      | "block_idx_x" -> [%expr [%e state].Sarek.Sarek_cpu_runtime.block_idx_x]
      | "block_idx_y" -> [%expr [%e state].Sarek.Sarek_cpu_runtime.block_idx_y]
      | "block_idx_z" -> [%expr [%e state].Sarek.Sarek_cpu_runtime.block_idx_z]
      | "block_dim_x" -> [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_x]
      | "block_dim_y" -> [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_y]
      | "block_dim_z" -> [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_z]
      | "grid_dim_x" -> [%expr [%e state].Sarek.Sarek_cpu_runtime.grid_dim_x]
      | "grid_dim_y" -> [%expr [%e state].Sarek.Sarek_cpu_runtime.grid_dim_y]
      | "grid_dim_z" -> [%expr [%e state].Sarek.Sarek_cpu_runtime.grid_dim_z]
      | _ ->
          (* Unknown - try to generate call to Gpu module *)
          evar_qualified ~loc ["Gpu"] name)
  | Sarek_env.IntrinsicRef (path, name) -> (
      (* Check if this is a Gpu module constant that maps to thread state *)
      let state = evar ~loc state_var in
      match (path, name) with
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "thread_idx_x" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.thread_idx_x]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "thread_idx_y" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.thread_idx_y]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "thread_idx_z" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.thread_idx_z]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "block_idx_x" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.block_idx_x]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "block_idx_y" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.block_idx_y]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "block_idx_z" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.block_idx_z]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "block_dim_x" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_x]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "block_dim_y" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_y]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "block_dim_z" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_z]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "grid_dim_x" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.grid_dim_x]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "grid_dim_y" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.grid_dim_y]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "grid_dim_z" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.grid_dim_z]
      | _ ->
          (* For other intrinsic constants from stdlib modules, look up via module path *)
          evar_qualified ~loc path name)

let gen_intrinsic_fun ~loc (ref : Sarek_env.intrinsic_ref)
    (args : expression list) : expression =
  match ref with
  | Sarek_env.CorePrimitiveRef name -> (
      let state = evar ~loc state_var in
      match name with
      | "block_barrier" | "warp_barrier" ->
          (* Call the barrier function from thread state *)
          [%expr [%e state].Sarek.Sarek_cpu_runtime.barrier ()]
      | "global_idx" -> [%expr Sarek.Sarek_cpu_runtime.global_idx_x [%e state]]
      | "global_size" ->
          [%expr Sarek.Sarek_cpu_runtime.global_size_x [%e state]]
      | _ ->
          (* Try Gpu module *)
          let fn = evar_qualified ~loc ["Gpu"] name in
          Ast_builder.Default.pexp_apply
            ~loc
            fn
            (List.map (fun a -> (Nolabel, a)) args))
  | Sarek_env.IntrinsicRef (path, name) -> (
      (* Check if this is a Gpu module function that maps to thread state *)
      let state = evar ~loc state_var in
      match (path, name) with
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "block_barrier" ->
          [%expr [%e state].Sarek.Sarek_cpu_runtime.barrier ()]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "global_idx" ->
          [%expr Sarek.Sarek_cpu_runtime.global_idx_x [%e state]]
      | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "global_size" ->
          [%expr Sarek.Sarek_cpu_runtime.global_size_x [%e state]]
      | _ -> (
          (* Call the OCaml implementation from the stdlib module *)
          let fn = evar_qualified ~loc path name in
          match args with
          | [] -> fn
          | _ ->
              Ast_builder.Default.pexp_apply
                ~loc
                fn
                (List.map (fun a -> (Nolabel, a)) args)))

(** {1 Expression Generation} *)

(** Mutable variable tracking. We track which variable IDs are mutable so that
    TEVar can dereference them. *)
module IntSet = Set.Make (Int)

(** Generate OCaml expression from typed Sarek expression.
    @param mut_vars Set of variable IDs that are mutable (need dereferencing) *)
let rec gen_expr_impl ~loc:_ ~mut_vars (te : texpr) : expression =
  let loc = ppxlib_loc_of_sarek te.te_loc in
  (* Helper to recursively generate, passing mut_vars *)
  let gen_expr ~loc e = gen_expr_impl ~loc ~mut_vars e in
  match te.te with
  (* Literals *)
  | TEUnit -> [%expr ()]
  | TEBool b -> if b then [%expr true] else [%expr false]
  | TEInt n -> (
      (* Check the type annotation to generate the correct literal type.
         In GPU kernels, integer literals compared with int32 should be int32. *)
      match repr te.ty with
      | TReg "int32" | TPrim TInt32 ->
          [%expr Int32.of_int [%e Ast_builder.Default.eint ~loc n]]
      | TReg "int64" ->
          [%expr Int64.of_int [%e Ast_builder.Default.eint ~loc n]]
      | _ ->
          (* Default to plain int *)
          Ast_builder.Default.eint ~loc n)
  | TEInt32 n ->
      (* Generate int32 literal using Int32.of_int *)
      [%expr Int32.of_int [%e Ast_builder.Default.eint ~loc (Int32.to_int n)]]
  | TEInt64 n ->
      (* Generate int64 literal using Int64.of_int *)
      [%expr Int64.of_int [%e Ast_builder.Default.eint ~loc (Int64.to_int n)]]
  | TEFloat f | TEDouble f ->
      Ast_builder.Default.efloat ~loc (string_of_float f)
  (* Variables *)
  | TEVar (name, id) ->
      (* Use the original variable name. This works for both:
         - Local let-bound variables (like "tid")
         - Module-level functions/constants (like "add_scale")
         - Mutable variables (need dereferencing with !)
         OCaml handles shadowing correctly, so using original names is safe. *)
      let var_e = evar ~loc name in
      if IntSet.mem id mut_vars then
        (* Mutable variable - dereference the ref *)
        [%expr ![%e var_e]]
      else var_e
  (* Vector/array access - convert int32 index to int for Bigarray *)
  | TEVecGet (vec, idx) ->
      let vec_e = gen_expr ~loc vec in
      let idx_e = gen_expr ~loc idx in
      [%expr Bigarray.Array1.get [%e vec_e] (Int32.to_int [%e idx_e])]
  | TEVecSet (vec, idx, value) ->
      let vec_e = gen_expr ~loc vec in
      let idx_e = gen_expr ~loc idx in
      let val_e = gen_expr ~loc value in
      [%expr
        Bigarray.Array1.set [%e vec_e] (Int32.to_int [%e idx_e]) [%e val_e]]
  | TEArrGet (arr, idx) ->
      let arr_e = gen_expr ~loc arr in
      let idx_e = gen_expr ~loc idx in
      [%expr Bigarray.Array1.get [%e arr_e] (Int32.to_int [%e idx_e])]
  | TEArrSet (arr, idx, value) ->
      let arr_e = gen_expr ~loc arr in
      let idx_e = gen_expr ~loc idx in
      let val_e = gen_expr ~loc value in
      [%expr
        Bigarray.Array1.set [%e arr_e] (Int32.to_int [%e idx_e]) [%e val_e]]
  (* Record field access - qualify field with module path from record type.
     For Geometry_lib.point, we need p.Geometry_lib.x, not p.x.
     For inline types (no module prefix), use unqualified name - the types
     are in scope via TEOpen from the original `let module Types = ... in`. *)
  | TEFieldGet (record, field_name, _field_idx) ->
      let rec_e = gen_expr ~loc record in
      let field_lid =
        match repr record.ty with
        | TRecord (type_name, _) -> (
            (* type_name may be "Module.type" or just "type" *)
            match String.rindex_opt type_name '.' with
            | Some idx ->
                (* Extract module path: "Geometry_lib.point" -> "Geometry_lib" *)
                let module_path = String.sub type_name 0 idx in
                (* Parse module path which may be nested: "A.B.C" *)
                let parts = String.split_on_char '.' module_path in
                (* Build Ldot chain: A.B.C.field_name *)
                let rec build_lid = function
                  | [] -> Lident field_name
                  | [m] -> Ldot (Lident m, field_name)
                  | m :: rest -> Ldot (build_lid rest, m)
                in
                {txt = build_lid (List.rev parts); loc}
            | None ->
                (* Inline type - use unqualified name, types are in scope *)
                {txt = Lident field_name; loc})
        | _ ->
            (* Not a record type - shouldn't happen, but use unqualified name *)
            {txt = Lident field_name; loc}
      in
      Ast_builder.Default.pexp_field ~loc rec_e field_lid
  | TEFieldSet (record, field_name, _field_idx, value) ->
      let rec_e = gen_expr ~loc record in
      let val_e = gen_expr ~loc value in
      let field_lid =
        match repr record.ty with
        | TRecord (type_name, _) -> (
            match String.rindex_opt type_name '.' with
            | Some idx ->
                let module_path = String.sub type_name 0 idx in
                let parts = String.split_on_char '.' module_path in
                let rec build_lid = function
                  | [] -> Lident field_name
                  | [m] -> Ldot (Lident m, field_name)
                  | m :: rest -> Ldot (build_lid rest, m)
                in
                {txt = build_lid (List.rev parts); loc}
            | None ->
                (* Inline type - use unqualified name *)
                {txt = Lident field_name; loc})
        | _ -> {txt = Lident field_name; loc}
      in
      Ast_builder.Default.pexp_setfield ~loc rec_e field_lid val_e
  (* Binary operations *)
  | TEBinop (op, a, b) ->
      let a_e = gen_expr ~loc a in
      let b_e = gen_expr ~loc b in
      gen_binop ~loc op a_e b_e a.ty
  (* Unary operations *)
  | TEUnop (op, a) ->
      let a_e = gen_expr ~loc a in
      gen_unop ~loc op a_e a.ty
  (* Function application *)
  | TEApp (fn, args) ->
      let fn_e = gen_expr ~loc fn in
      let args_e = List.map (gen_expr ~loc) args in
      Ast_builder.Default.pexp_apply
        ~loc
        fn_e
        (List.map (fun a -> (Nolabel, a)) args_e)
  (* Assignment to mutable variable *)
  | TEAssign (name, _id, value) ->
      let val_e = gen_expr ~loc value in
      (* Mutable variables are stored as refs with the original name *)
      let var_e = evar ~loc name in
      [%expr [%e var_e] := [%e val_e]]
  (* Let binding *)
  | TELet (name, _id, value, body) ->
      let val_e = gen_expr ~loc value in
      let body_e = gen_expr ~loc body in
      let pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
      [%expr
        let [%p pat] = [%e val_e] in
        [%e body_e]]
  (* Mutable let binding - use a ref cell with same name.
     TEVar will dereference the ref when reading.
     TEAssign will update the ref.
     This isn't ideal for performance but works semantically. *)
  | TELetMut (name, id, value, body) ->
      let val_e = gen_expr ~loc value in
      (* Add this variable to the mutable set for the body *)
      let mut_vars' = IntSet.add id mut_vars in
      let body_e = gen_expr_impl ~loc ~mut_vars:mut_vars' body in
      let pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
      (* Create: let x = ref val in body
         Note: TEVar for mutable vars will dereference with !.
         TEAssign uses := which works on refs. *)
      [%expr
        let [%p pat] = ref [%e val_e] in
        [%e body_e]]
  (* Conditionals *)
  | TEIf (cond, then_e, else_e) ->
      let cond_e = gen_expr ~loc cond in
      let then_e' = gen_expr ~loc then_e in
      let else_e' =
        match else_e with Some e -> gen_expr ~loc e | None -> [%expr ()]
      in
      [%expr if [%e cond_e] then [%e then_e'] else [%e else_e']]
  (* For loop - OCaml for loops use int, but kernel expects int32.
     We use a temporary int variable for the loop, then shadow it with int32 in body. *)
  | TEFor (var_name_str, _var_id, lo, hi, dir, body) -> (
      let lo_e = gen_expr ~loc lo in
      let hi_e = gen_expr ~loc hi in
      let body_e = gen_expr ~loc body in
      (* Use a temporary name for the int loop variable *)
      let int_var_name = var_name_str ^ "__int" in
      let int_var_pat =
        Ast_builder.Default.ppat_var ~loc {txt = int_var_name; loc}
      in
      let int_var_e = evar ~loc int_var_name in
      (* The int32 variable that shadows the int one in the body, using original name *)
      let int32_var_pat =
        Ast_builder.Default.ppat_var ~loc {txt = var_name_str; loc}
      in
      (* Wrap body with int32 conversion *)
      let wrapped_body =
        [%expr
          let [%p int32_var_pat] = Int32.of_int [%e int_var_e] in
          [%e body_e]]
      in
      match dir with
      | Sarek_ast.Upto ->
          [%expr
            for
              [%p int_var_pat] = Int32.to_int [%e lo_e]
              to Int32.to_int [%e hi_e] - 1
            do
              [%e wrapped_body]
            done]
      | Sarek_ast.Downto ->
          [%expr
            for
              [%p int_var_pat] = Int32.to_int [%e hi_e] - 1
              downto Int32.to_int [%e lo_e]
            do
              [%e wrapped_body]
            done])
  (* While loop *)
  | TEWhile (cond, body) ->
      let cond_e = gen_expr ~loc cond in
      let body_e = gen_expr ~loc body in
      [%expr
        while [%e cond_e] do
          [%e body_e]
        done]
  (* Sequence *)
  | TESeq exprs -> (
      let exprs_e = List.map (gen_expr ~loc) exprs in
      match exprs_e with
      | [] -> [%expr ()]
      | [e] -> e
      | es ->
          List.fold_right
            (fun e acc ->
              [%expr
                [%e e] ;
                [%e acc]])
            (List.rev (List.tl (List.rev es)))
            (List.hd (List.rev es)))
  (* Match *)
  | TEMatch (scrutinee, cases) ->
      let scrut_e = gen_expr ~loc scrutinee in
      let cases_e =
        List.map
          (fun (pat, body) ->
            let pat_e = gen_pattern ~loc pat in
            let body_e = gen_expr ~loc body in
            Ast_builder.Default.case ~lhs:pat_e ~guard:None ~rhs:body_e)
          cases
      in
      Ast_builder.Default.pexp_match ~loc scrut_e cases_e
  (* Record construction - qualify field names with module path from type_name.
     For inline types, use unqualified name - types are in scope via TEOpen. *)
  | TERecord (type_name, fields) ->
      let field_lid name =
        match String.rindex_opt type_name '.' with
        | Some idx ->
            let module_path = String.sub type_name 0 idx in
            let parts = String.split_on_char '.' module_path in
            let rec build_lid = function
              | [] -> Lident name
              | [m] -> Ldot (Lident m, name)
              | m :: rest -> Ldot (build_lid rest, m)
            in
            {txt = build_lid (List.rev parts); loc}
        | None ->
            (* Inline type - use unqualified name *)
            {txt = Lident name; loc}
      in
      let fields_e =
        List.map
          (fun (name, expr) -> (field_lid name, gen_expr ~loc expr))
          fields
      in
      Ast_builder.Default.pexp_record ~loc fields_e None
  (* Variant construction - qualify constructor with module path.
     For inline types, use unqualified name - types are in scope via TEOpen. *)
  | TEConstr (type_name, constr_name, arg) ->
      let arg_e = Option.map (gen_expr ~loc) arg in
      let constr_lid =
        match String.rindex_opt type_name '.' with
        | Some idx ->
            let module_path = String.sub type_name 0 idx in
            let parts = String.split_on_char '.' module_path in
            let rec build_lid = function
              | [] -> Lident constr_name
              | [m] -> Ldot (Lident m, constr_name)
              | m :: rest -> Ldot (build_lid rest, m)
            in
            build_lid (List.rev parts)
        | None ->
            (* Inline type - use unqualified name *)
            Lident constr_name
      in
      Ast_builder.Default.pexp_construct ~loc {txt = constr_lid; loc} arg_e
  (* Tuple *)
  | TETuple exprs ->
      let exprs_e = List.map (gen_expr ~loc) exprs in
      Ast_builder.Default.pexp_tuple ~loc exprs_e
  (* Return - just evaluate the expression *)
  | TEReturn e -> gen_expr ~loc e
  (* Create local array - allocate on stack/heap *)
  | TECreateArray (size, elem_ty, _memspace) ->
      let size_e = gen_expr ~loc size in
      let kind_e = bigarray_kind_of_typ ~loc elem_ty in
      [%expr Bigarray.Array1.create [%e kind_e] Bigarray.c_layout [%e size_e]]
  (* Global ref - reference to external value *)
  | TEGlobalRef (name, _typ) ->
      (* Dereference the ref *)
      let var_e = evar ~loc name in
      [%expr ![%e var_e]]
  (* Native code - wrap as OCaml string (will fail at runtime if executed) *)
  | TENative code ->
      [%expr
        failwith
          [%e
            Ast_builder.Default.estring
              ~loc
              ("Native code not supported on CPU: " ^ code)]]
  (* Native function - call with interpreter device to get code string *)
  | TENativeFun _fn_expr ->
      [%expr failwith "Native functions not supported on CPU backend"]
  (* Pragma - just evaluate body (pragmas are hints for GPU) *)
  | TEPragma (_opts, body) -> gen_expr ~loc body
  (* Intrinsic constant - thread indices, etc. *)
  | TEIntrinsicConst ref -> gen_intrinsic_const ~loc ref
  (* Intrinsic function - math functions, barriers, etc. *)
  | TEIntrinsicFun (ref, _convergence, args) ->
      let args_e = List.map (gen_expr ~loc) args in
      gen_intrinsic_fun ~loc ref args_e
  (* BSP let%shared - allocate shared memory *)
  | TELetShared (name, _id, elem_ty, size_opt, body) ->
      (* Size needs to be int, but expressions may be int32 (like block_dim_x).
         Wrap in Int32.to_int for conversion. *)
      let size_e =
        match size_opt with
        | Some s ->
            let s_e = gen_expr ~loc s in
            [%expr Int32.to_int [%e s_e]]
        | None ->
            (* Default to block_dim_x - convert from int32 to int *)
            let state = evar ~loc state_var in
            [%expr Int32.to_int [%e state].Sarek.Sarek_cpu_runtime.block_dim_x]
      in
      let alloc_fn =
        match repr elem_ty with
        | TReg "float32" -> [%expr Sarek.Sarek_cpu_runtime.alloc_shared_float32]
        | TPrim TInt32 -> [%expr Sarek.Sarek_cpu_runtime.alloc_shared_int32]
        | _ -> [%expr Sarek.Sarek_cpu_runtime.alloc_shared_float32]
      in
      let shared = evar ~loc shared_var in
      let body_e = gen_expr ~loc body in
      let pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
      [%expr
        let [%p pat] =
          [%e alloc_fn]
            [%e shared]
            [%e Ast_builder.Default.estring ~loc name]
            [%e size_e]
        in
        [%e body_e]]
  (* BSP let%superstep - synchronized block + barrier *)
  | TESuperstep (_name, _divergent, step_body, cont) ->
      let body_e = gen_expr ~loc step_body in
      let cont_e = gen_expr ~loc cont in
      let state = evar ~loc state_var in
      [%expr
        [%e body_e] ;
        [%e state].Sarek.Sarek_cpu_runtime.barrier () ;
        [%e cont_e]]
  (* Module open - generate let open M.N in body *)
  | TEOpen (path, body) ->
      let body_e = gen_expr ~loc body in
      (* Build the module path longident: M.N.O *)
      let mod_lid =
        match path with
        | [] -> failwith "empty module path in TEOpen"
        | [m] -> Lident m
        | m :: rest ->
            List.fold_left (fun acc p -> Ldot (acc, p)) (Lident m) rest
      in
      Ast_builder.Default.pexp_open
        ~loc
        (Ast_builder.Default.open_infos
           ~loc
           ~override:Fresh
           ~expr:(Ast_builder.Default.pmod_ident ~loc {txt = mod_lid; loc}))
        body_e

(** Generate pattern from typed pattern *)
and gen_pattern ~loc:_ (tpat : tpattern) : pattern =
  let loc = ppxlib_loc_of_sarek tpat.tpat_loc in
  match tpat.tpat with
  | TPAny -> Ast_builder.Default.ppat_any ~loc
  | TPVar (name, _id) ->
      (* Use original name for pattern variables *)
      Ast_builder.Default.ppat_var ~loc {txt = name; loc}
  | TPConstr (type_name, constr_name, arg) ->
      let arg_p = Option.map (gen_pattern ~loc) arg in
      (* Qualify constructor with module path from type_name if present.
         For "Geometry_lib.shape", we need Geometry_lib.Circle, not Circle.
         For inline types, use unqualified name - types are in scope via TEOpen. *)
      let constr_lid =
        match String.rindex_opt type_name '.' with
        | Some idx ->
            let module_path = String.sub type_name 0 idx in
            let parts = String.split_on_char '.' module_path in
            let rec build_lid = function
              | [] -> Lident constr_name
              | [m] -> Ldot (Lident m, constr_name)
              | m :: rest -> Ldot (build_lid rest, m)
            in
            build_lid (List.rev parts)
        | None ->
            (* Inline type - use unqualified name *)
            Lident constr_name
      in
      Ast_builder.Default.ppat_construct ~loc {txt = constr_lid; loc} arg_p
  | TPTuple pats ->
      let pats_p = List.map (gen_pattern ~loc) pats in
      Ast_builder.Default.ppat_tuple ~loc pats_p

(** Generate binary operation *)
and gen_binop ~loc op a b ty : expression =
  let ty_repr = repr ty in
  match op with
  | Sarek_ast.Add -> (
      match ty_repr with
      | TReg "float32" | TReg "float64" -> [%expr [%e a] +. [%e b]]
      | TReg "int32" | TPrim TInt32 -> [%expr Int32.add [%e a] [%e b]]
      | TReg "int64" -> [%expr Int64.add [%e a] [%e b]]
      | _ -> [%expr [%e a] + [%e b]])
  | Sarek_ast.Sub -> (
      match ty_repr with
      | TReg "float32" | TReg "float64" -> [%expr [%e a] -. [%e b]]
      | TReg "int32" | TPrim TInt32 -> [%expr Int32.sub [%e a] [%e b]]
      | TReg "int64" -> [%expr Int64.sub [%e a] [%e b]]
      | _ -> [%expr [%e a] - [%e b]])
  | Sarek_ast.Mul -> (
      match ty_repr with
      | TReg "float32" | TReg "float64" -> [%expr [%e a] *. [%e b]]
      | TReg "int32" | TPrim TInt32 -> [%expr Int32.mul [%e a] [%e b]]
      | TReg "int64" -> [%expr Int64.mul [%e a] [%e b]]
      | _ -> [%expr [%e a] * [%e b]])
  | Sarek_ast.Div -> (
      match ty_repr with
      | TReg "float32" | TReg "float64" -> [%expr [%e a] /. [%e b]]
      | TReg "int32" | TPrim TInt32 -> [%expr Int32.div [%e a] [%e b]]
      | TReg "int64" -> [%expr Int64.div [%e a] [%e b]]
      | _ -> [%expr [%e a] / [%e b]])
  | Sarek_ast.Mod -> (
      match ty_repr with
      | TReg "int32" | TPrim TInt32 -> [%expr Int32.rem [%e a] [%e b]]
      | TReg "int64" -> [%expr Int64.rem [%e a] [%e b]]
      | _ -> [%expr [%e a] mod [%e b]])
  | Sarek_ast.And -> [%expr [%e a] && [%e b]]
  | Sarek_ast.Or -> [%expr [%e a] || [%e b]]
  (* Comparison operators work polymorphically in OCaml *)
  | Sarek_ast.Eq -> [%expr [%e a] = [%e b]]
  | Sarek_ast.Ne -> [%expr [%e a] <> [%e b]]
  | Sarek_ast.Lt -> [%expr [%e a] < [%e b]]
  | Sarek_ast.Gt -> [%expr [%e a] > [%e b]]
  | Sarek_ast.Le -> [%expr [%e a] <= [%e b]]
  | Sarek_ast.Ge -> [%expr [%e a] >= [%e b]]
  (* Bitwise operations *)
  | Sarek_ast.Land -> (
      match ty_repr with
      | TReg "int32" | TPrim TInt32 -> [%expr Int32.logand [%e a] [%e b]]
      | TReg "int64" -> [%expr Int64.logand [%e a] [%e b]]
      | _ -> [%expr [%e a] land [%e b]])
  | Sarek_ast.Lor -> (
      match ty_repr with
      | TReg "int32" | TPrim TInt32 -> [%expr Int32.logor [%e a] [%e b]]
      | TReg "int64" -> [%expr Int64.logor [%e a] [%e b]]
      | _ -> [%expr [%e a] lor [%e b]])
  | Sarek_ast.Lxor -> (
      match ty_repr with
      | TReg "int32" | TPrim TInt32 -> [%expr Int32.logxor [%e a] [%e b]]
      | TReg "int64" -> [%expr Int64.logxor [%e a] [%e b]]
      | _ -> [%expr [%e a] lxor [%e b]])
  | Sarek_ast.Lsl -> (
      match ty_repr with
      | TReg "int32" | TPrim TInt32 ->
          [%expr Int32.shift_left [%e a] (Int32.to_int [%e b])]
      | TReg "int64" -> [%expr Int64.shift_left [%e a] (Int64.to_int [%e b])]
      | _ -> [%expr [%e a] lsl [%e b]])
  | Sarek_ast.Lsr -> (
      match ty_repr with
      | TReg "int32" | TPrim TInt32 ->
          [%expr Int32.shift_right_logical [%e a] (Int32.to_int [%e b])]
      | TReg "int64" ->
          [%expr Int64.shift_right_logical [%e a] (Int64.to_int [%e b])]
      | _ -> [%expr [%e a] lsr [%e b]])
  | Sarek_ast.Asr -> (
      match ty_repr with
      | TReg "int32" | TPrim TInt32 ->
          [%expr Int32.shift_right [%e a] (Int32.to_int [%e b])]
      | TReg "int64" -> [%expr Int64.shift_right [%e a] (Int64.to_int [%e b])]
      | _ -> [%expr [%e a] asr [%e b]])

(** Generate unary operation *)
and gen_unop ~loc op a ty : expression =
  let ty_repr = repr ty in
  match op with
  | Sarek_ast.Neg -> (
      match ty_repr with
      | TReg "float32" | TReg "float64" -> [%expr -.[%e a]]
      | TReg "int32" | TPrim TInt32 -> [%expr Int32.neg [%e a]]
      | TReg "int64" -> [%expr Int64.neg [%e a]]
      | _ -> [%expr -[%e a]])
  | Sarek_ast.Not -> [%expr not [%e a]]
  | Sarek_ast.Lnot -> (
      match ty_repr with
      | TReg "int32" | TPrim TInt32 -> [%expr Int32.lognot [%e a]]
      | TReg "int64" -> [%expr Int64.lognot [%e a]]
      | _ -> [%expr lnot [%e a]])

(** Top-level entry point for generating expressions. Starts with empty mutable
    variable set. *)
let gen_expr ~loc e = gen_expr_impl ~loc ~mut_vars:IntSet.empty e

(** {1 Module Item Generation} *)

(** Generate a module-level function (TMFun) as a let binding. *)
let gen_module_fun ~loc (name : string) (params : tparam list) (body : texpr) :
    pattern * expression =
  let fn_pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
  let body_e = gen_expr ~loc body in

  (* Build function from parameters - use parameter names with type constraints
     to help OCaml resolve record fields and other types *)
  let fn_e =
    List.fold_right
      (fun param acc ->
        let var_pat =
          Ast_builder.Default.ppat_var ~loc {txt = param.tparam_name; loc}
        in
        let ty = core_type_of_typ ~loc param.tparam_type in
        let param_pat = Ast_builder.Default.ppat_constraint ~loc var_pat ty in
        [%expr fun [%p param_pat] -> [%e acc]])
      params
      body_e
  in
  (fn_pat, fn_e)

(** Generate a module-level constant (TMConst) as a let binding. *)
let gen_module_const ~loc (name : string) (_id : int) (_typ : typ)
    (value : texpr) : pattern * expression =
  (* Use the constant name, not the ID *)
  let const_pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
  let value_e = gen_expr ~loc value in
  (const_pat, value_e)

(** {1 Type Declaration Generation} *)

(** Generate a type declaration for a record type *)
let gen_type_decl_record ~loc (name : string)
    (fields : (string * typ * bool) list) : structure_item =
  let field_decls =
    List.map
      (fun (fname, fty, is_mutable) ->
        let ty = core_type_of_typ ~loc fty in
        Ast_builder.Default.label_declaration
          ~loc
          ~name:{txt = fname; loc}
          ~mutable_:(if is_mutable then Mutable else Immutable)
          ~type_:ty)
      fields
  in
  Ast_builder.Default.pstr_type
    ~loc
    Recursive
    [
      Ast_builder.Default.type_declaration
        ~loc
        ~name:{txt = name; loc}
        ~params:[]
        ~cstrs:[]
        ~kind:(Ptype_record field_decls)
        ~private_:Public
        ~manifest:None;
    ]

(** Generate a type declaration for a variant type *)
let gen_type_decl_variant ~loc (name : string)
    (constrs : (string * typ option) list) : structure_item =
  let constr_decls =
    List.map
      (fun (cname, arg_opt) ->
        let args =
          match arg_opt with
          | None -> Pcstr_tuple []
          | Some ty -> Pcstr_tuple [core_type_of_typ ~loc ty]
        in
        Ast_builder.Default.constructor_declaration
          ~loc
          ~name:{txt = cname; loc}
          ~args
          ~res:None)
      constrs
  in
  Ast_builder.Default.pstr_type
    ~loc
    Recursive
    [
      Ast_builder.Default.type_declaration
        ~loc
        ~name:{txt = name; loc}
        ~params:[]
        ~cstrs:[]
        ~kind:(Ptype_variant constr_decls)
        ~private_:Public
        ~manifest:None;
    ]

(** Generate a structure item from a typed type declaration *)
let gen_type_decl_item ~loc (decl : ttype_decl) : structure_item =
  match decl with
  | TTypeRecord {tdecl_name; tdecl_fields; _} ->
      gen_type_decl_record ~loc tdecl_name tdecl_fields
  | TTypeVariant {tdecl_name; tdecl_constructors; _} ->
      gen_type_decl_variant ~loc tdecl_name tdecl_constructors

(** Wrap expression with type declarations using a local module. Generates: let
    module Types = struct [@@@warning "-34-37"] type t = ... end in let open
    Types in body The warning suppression avoids unused-type and
    unused-constructor warnings for registered types that may not be used by
    this specific kernel. *)
let wrap_type_decls ~loc (decls : ttype_decl list) (body : expression) :
    expression =
  if decls = [] then body
  else
    (* Generate structure items for each type declaration *)
    let type_items = List.map (gen_type_decl_item ~loc) decls in
    (* Add warning suppression at the start of the module:
       [@@@warning "-34-37"] suppresses unused-type-declaration and unused-constructor *)
    let warning_attr =
      Ast_builder.Default.pstr_attribute
        ~loc
        (Ast_builder.Default.attribute
           ~loc
           ~name:{txt = "warning"; loc}
           ~payload:
             (PStr
                [
                  Ast_builder.Default.pstr_eval
                    ~loc
                    (Ast_builder.Default.estring ~loc "-34-37")
                    [];
                ]))
    in
    (* Create the module structure with warning suppression first *)
    let mod_struct =
      Ast_builder.Default.pmod_structure ~loc (warning_attr :: type_items)
    in
    (* Wrap with let module Types = struct ... end in let open Types in body *)
    let open_types =
      Ast_builder.Default.pexp_open
        ~loc
        (Ast_builder.Default.open_infos
           ~loc
           ~override:Fresh
           ~expr:
             (Ast_builder.Default.pmod_ident ~loc {txt = Lident "Types"; loc}))
        body
    in
    Ast_builder.Default.pexp_letmodule
      ~loc
      {txt = Some "Types"; loc}
      mod_struct
      open_types

(** Wrap expression with module item bindings. *)
let wrap_module_items ~loc (items : tmodule_item list) (body : expression) :
    expression =
  List.fold_right
    (fun item acc ->
      match item with
      | TMFun (name, params, item_body) ->
          let pat, expr = gen_module_fun ~loc name params item_body in
          [%expr
            let [%p pat] = [%e expr] in
            [%e acc]]
      | TMConst (name, id, typ, value) ->
          let pat, expr = gen_module_const ~loc name id typ value in
          [%expr
            let [%p pat] = [%e expr] in
            [%e acc]])
    items
    body

(** {1 Kernel Generation} *)

(** Generate the cpu_kern function from a typed kernel.

    The generated function has type: Sarek_cpu_runtime.thread_state -> (arg1,
    arg2, ...) -> unit

    Parameters are passed as a tuple of bigarrays and scalars. *)
let gen_cpu_kern ~loc (kernel : tkernel) : expression =
  let body_e = gen_expr ~loc kernel.tkern_body in

  (* Generate inline module items only.
     External registered functions ([@sarek.module]) are the first N items,
     where N = tkern_external_item_count. We skip those since they're already
     available via the OCaml module system.
     The remaining items are inline helpers defined within the kernel payload. *)
  let inline_items =
    let all_items = kernel.tkern_module_items in
    let skip_count = kernel.tkern_external_item_count in
    (* Drop the first skip_count items (external ones) *)
    let rec drop n lst =
      if n <= 0 then lst
      else match lst with [] -> [] | _ :: tl -> drop (n - 1) tl
    in
    drop skip_count all_items
  in
  let body_with_items = wrap_module_items ~loc inline_items body_e in

  (* Build parameter tuple pattern - use parameter names with type constraints
     This is essential for int32 parameters to be typed correctly, otherwise
     OCaml infers them as int due to literal comparisons. *)
  let param_pats =
    List.map
      (fun p ->
        let var_pat =
          Ast_builder.Default.ppat_var ~loc {txt = p.tparam_name; loc}
        in
        let ty = core_type_of_typ ~loc p.tparam_type in
        Ast_builder.Default.ppat_constraint ~loc var_pat ty)
      kernel.tkern_params
  in
  let params_pat =
    match param_pats with
    | [] -> [%pat? ()]
    | [p] -> p
    | ps -> Ast_builder.Default.ppat_tuple ~loc ps
  in

  (* Build the function:
     fun __state __shared (arg0, arg1, ...) -> body
     We suppress warning 27 (unused-var-strict) since kernel parameters
     may not always be used (e.g., barrier-only kernels), and warnings
     32/33 for unused value bindings. *)
  let state_pat = Ast_builder.Default.ppat_var ~loc {txt = state_var; loc} in
  let shared_pat = Ast_builder.Default.ppat_var ~loc {txt = shared_var; loc} in

  (* The function expression with warning suppression attribute on the body *)
  let inner_fun =
    [%expr
      fun ([%p state_pat] : Sarek.Sarek_cpu_runtime.thread_state)
          ([%p shared_pat] : Sarek.Sarek_cpu_runtime.shared_mem)
          [%p params_pat] -> [%e body_with_items]]
  in
  (* Add warning suppression attribute to the function *)
  let fun_with_warnings =
    {
      inner_fun with
      pexp_attributes =
        [
          Ast_builder.Default.attribute
            ~loc
            ~name:{txt = "warning"; loc}
            ~payload:
              (PStr
                 [
                   Ast_builder.Default.pstr_eval
                     ~loc
                     (Ast_builder.Default.estring ~loc "-27-32-33")
                     [];
                 ]);
        ];
    }
  in
  fun_with_warnings

(** Generate a type cast expression for extracting a kernel argument.

    For vectors: cast Obj.t to the appropriate bigarray type For scalars: cast
    Obj.t to the primitive type *)
let gen_arg_cast ~loc (param : tparam) (idx : int) : expression =
  let arr_access =
    [%expr Array.get __args [%e Ast_builder.Default.eint ~loc idx]]
  in
  match repr param.tparam_type with
  | TVec elem_ty -> (
      (* Vector types - cast to appropriate bigarray *)
      match repr elem_ty with
      | TReg "float32" ->
          [%expr
            (Obj.obj [%e arr_access]
              : ( float,
                  Bigarray.float32_elt,
                  Bigarray.c_layout )
                Bigarray.Array1.t)]
      | TReg "float64" ->
          [%expr
            (Obj.obj [%e arr_access]
              : ( float,
                  Bigarray.float64_elt,
                  Bigarray.c_layout )
                Bigarray.Array1.t)]
      | TReg "int32" | TPrim TInt32 ->
          [%expr
            (Obj.obj [%e arr_access]
              : (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t)]
      | TReg "int64" ->
          [%expr
            (Obj.obj [%e arr_access]
              : (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t)]
      | _ ->
          (* Default to float32 for unknown types *)
          [%expr
            (Obj.obj [%e arr_access]
              : ( float,
                  Bigarray.float32_elt,
                  Bigarray.c_layout )
                Bigarray.Array1.t)])
  | TReg "float32" -> [%expr (Obj.obj [%e arr_access] : float)]
  | TReg "float64" -> [%expr (Obj.obj [%e arr_access] : float)]
  | TReg "int32" | TPrim TInt32 -> [%expr (Obj.obj [%e arr_access] : int32)]
  | TReg "int64" -> [%expr (Obj.obj [%e arr_access] : int64)]
  | TPrim TBool -> [%expr (Obj.obj [%e arr_access] : bool)]
  | _ ->
      (* Default - just return as Obj.t and let caller deal with it *)
      arr_access

(** Generate the cpu_kern wrapper that matches the kirc_kernel.cpu_kern type.

    Generated function type: block:int*int*int -> grid:int*int*int -> Obj.t
    array -> unit

    This wrapper: 1. Extracts and casts each argument from the Obj.t array 2.
    Builds the args tuple 3. Calls run_sequential with the native kernel

    Note: Kernels with inline type declarations (ktype) or vectors of record
    types cannot be executed natively because OCaml's type system doesn't
    support local type definitions that escape their scope. These kernels will
    raise an exception when executed on CPU. *)
let gen_cpu_kern_wrapper ~loc (kernel : tkernel) : expression =
  (* Check if the kernel uses features that prevent native execution *)
  let has_inline_types = kernel.tkern_type_decls <> [] in
  let has_record_vector_params =
    List.exists
      (fun p ->
        match repr p.tparam_type with
        | TVec elem_ty -> (
            match repr elem_ty with
            | TRecord _ | TVariant _ -> true
            | _ -> false)
        | _ -> false)
      kernel.tkern_params
  in
  if has_inline_types || has_record_vector_params then
    (* Generate a stub that raises an error - native execution not supported *)
    [%expr
      fun ~block:_ ~grid:_ _ ->
        failwith
          "Native CPU execution not supported for kernels with inline types or \
           record/variant vectors"]
  else
    let native_kern = gen_cpu_kern ~loc kernel in

    (* Generate argument extraction bindings - use parameter names *)
    let arg_bindings =
      List.mapi
        (fun i param ->
          let var_pat =
            Ast_builder.Default.ppat_var ~loc {txt = param.tparam_name; loc}
          in
          let cast_expr = gen_arg_cast ~loc param i in
          (var_pat, cast_expr))
        kernel.tkern_params
    in

    (* Build the args tuple expression - use parameter names *)
    let args_tuple =
      let arg_exprs =
        List.map (fun p -> evar ~loc p.tparam_name) kernel.tkern_params
      in
      match arg_exprs with
      | [] -> [%expr ()]
      | [e] -> e
      | es -> Ast_builder.Default.pexp_tuple ~loc es
    in

    (* Build the nested let bindings *)
    let body_with_bindings =
      List.fold_right
        (fun (pat, expr) body ->
          [%expr
            let [%p pat] = [%e expr] in
            [%e body]])
        arg_bindings
        [%expr
          Sarek.Sarek_cpu_runtime.run_sequential
            ~block
            ~grid
            __native_kern
            [%e args_tuple]]
    in

    [%expr
      fun ~block ~grid __args ->
        let __native_kern = [%e native_kern] in
        [%e body_with_bindings]]
