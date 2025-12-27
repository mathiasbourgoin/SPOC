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
  | Sarek_env.IntrinsicRef (path, name) ->
      (* For intrinsic constants from stdlib modules, look up via module path *)
      evar_qualified ~loc path name

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
      (* Call the OCaml implementation from the stdlib module *)
      let fn = evar_qualified ~loc path name in
      match args with
      | [] -> fn
      | _ ->
          Ast_builder.Default.pexp_apply
            ~loc
            fn
            (List.map (fun a -> (Nolabel, a)) args))

(** {1 Expression Generation} *)

(** Generate OCaml expression from typed Sarek expression *)
let rec gen_expr ~loc:_ (te : texpr) : expression =
  let loc = ppxlib_loc_of_sarek te.te_loc in
  match te.te with
  (* Literals *)
  | TEUnit -> [%expr ()]
  | TEBool b -> if b then [%expr true] else [%expr false]
  | TEInt n -> Ast_builder.Default.eint ~loc n
  | TEInt32 n -> [%expr [%e Ast_builder.Default.eint ~loc (Int32.to_int n)]]
  | TEInt64 n -> [%expr [%e Ast_builder.Default.eint ~loc (Int64.to_int n)]]
  | TEFloat f | TEDouble f ->
      Ast_builder.Default.efloat ~loc (string_of_float f)
  (* Variables *)
  | TEVar (_name, id) ->
      (* Local variables are stored by their id *)
      evar ~loc (var_name id)
  (* Vector/array access *)
  | TEVecGet (vec, idx) ->
      let vec_e = gen_expr ~loc vec in
      let idx_e = gen_expr ~loc idx in
      [%expr Bigarray.Array1.get [%e vec_e] [%e idx_e]]
  | TEVecSet (vec, idx, value) ->
      let vec_e = gen_expr ~loc vec in
      let idx_e = gen_expr ~loc idx in
      let val_e = gen_expr ~loc value in
      [%expr Bigarray.Array1.set [%e vec_e] [%e idx_e] [%e val_e]]
  | TEArrGet (arr, idx) ->
      let arr_e = gen_expr ~loc arr in
      let idx_e = gen_expr ~loc idx in
      [%expr Bigarray.Array1.get [%e arr_e] [%e idx_e]]
  | TEArrSet (arr, idx, value) ->
      let arr_e = gen_expr ~loc arr in
      let idx_e = gen_expr ~loc idx in
      let val_e = gen_expr ~loc value in
      [%expr Bigarray.Array1.set [%e arr_e] [%e idx_e] [%e val_e]]
  (* Record field access *)
  | TEFieldGet (record, field_name, _field_idx) ->
      let rec_e = gen_expr ~loc record in
      let field_lid = {txt = Lident field_name; loc} in
      Ast_builder.Default.pexp_field ~loc rec_e field_lid
  | TEFieldSet (record, field_name, _field_idx, value) ->
      let rec_e = gen_expr ~loc record in
      let val_e = gen_expr ~loc value in
      let field_lid = {txt = Lident field_name; loc} in
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
  | TEAssign (_name, id, value) ->
      let val_e = gen_expr ~loc value in
      let var_e = evar ~loc (mut_var_name id) in
      [%expr [%e var_e] := [%e val_e]]
  (* Let binding *)
  | TELet (_name, id, value, body) ->
      let val_e = gen_expr ~loc value in
      let body_e = gen_expr ~loc body in
      let pat = Ast_builder.Default.ppat_var ~loc {txt = var_name id; loc} in
      [%expr
        let [%p pat] = [%e val_e] in
        [%e body_e]]
  (* Mutable let binding *)
  | TELetMut (_name, id, value, body) ->
      let val_e = gen_expr ~loc value in
      let body_e = gen_expr ~loc body in
      let pat =
        Ast_builder.Default.ppat_var ~loc {txt = mut_var_name id; loc}
      in
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
  (* For loop *)
  | TEFor (_var_name, var_id, lo, hi, dir, body) -> (
      let lo_e = gen_expr ~loc lo in
      let hi_e = gen_expr ~loc hi in
      let body_e = gen_expr ~loc body in
      let var_pat =
        Ast_builder.Default.ppat_var ~loc {txt = var_name var_id; loc}
      in
      match dir with
      | Sarek_ast.Upto ->
          [%expr
            for [%p var_pat] = [%e lo_e] to [%e hi_e] - 1 do
              [%e body_e]
            done]
      | Sarek_ast.Downto ->
          [%expr
            for [%p var_pat] = [%e hi_e] - 1 downto [%e lo_e] do
              [%e body_e]
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
  (* Record construction *)
  | TERecord (_type_name, fields) ->
      let fields_e =
        List.map
          (fun (name, expr) -> ({txt = Lident name; loc}, gen_expr ~loc expr))
          fields
      in
      Ast_builder.Default.pexp_record ~loc fields_e None
  (* Variant construction *)
  | TEConstr (_type_name, constr_name, arg) ->
      let arg_e = Option.map (gen_expr ~loc) arg in
      Ast_builder.Default.pexp_construct
        ~loc
        {txt = Lident constr_name; loc}
        arg_e
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
  | TELetShared (name, id, elem_ty, size_opt, body) ->
      let size_e =
        match size_opt with
        | Some s -> gen_expr ~loc s
        | None ->
            (* Default to block_dim_x *)
            let state = evar ~loc state_var in
            [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_x]
      in
      let alloc_fn =
        match repr elem_ty with
        | TReg "float32" -> [%expr Sarek.Sarek_cpu_runtime.alloc_shared_float32]
        | TPrim TInt32 -> [%expr Sarek.Sarek_cpu_runtime.alloc_shared_int32]
        | _ -> [%expr Sarek.Sarek_cpu_runtime.alloc_shared_float32]
      in
      let shared = evar ~loc shared_var in
      let body_e = gen_expr ~loc body in
      let pat = Ast_builder.Default.ppat_var ~loc {txt = var_name id; loc} in
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

(** Generate pattern from typed pattern *)
and gen_pattern ~loc:_ (tpat : tpattern) : pattern =
  let loc = ppxlib_loc_of_sarek tpat.tpat_loc in
  match tpat.tpat with
  | TPAny -> Ast_builder.Default.ppat_any ~loc
  | TPVar (_name, id) ->
      Ast_builder.Default.ppat_var ~loc {txt = var_name id; loc}
  | TPConstr (_type_name, constr_name, arg) ->
      let arg_p = Option.map (gen_pattern ~loc) arg in
      Ast_builder.Default.ppat_construct
        ~loc
        {txt = Lident constr_name; loc}
        arg_p
  | TPTuple pats ->
      let pats_p = List.map (gen_pattern ~loc) pats in
      Ast_builder.Default.ppat_tuple ~loc pats_p

(** Generate binary operation *)
and gen_binop ~loc op a b ty : expression =
  match op with
  | Sarek_ast.Add -> (
      match repr ty with
      | TReg "float32" | TReg "float64" -> [%expr [%e a] +. [%e b]]
      | _ -> [%expr [%e a] + [%e b]])
  | Sarek_ast.Sub -> (
      match repr ty with
      | TReg "float32" | TReg "float64" -> [%expr [%e a] -. [%e b]]
      | _ -> [%expr [%e a] - [%e b]])
  | Sarek_ast.Mul -> (
      match repr ty with
      | TReg "float32" | TReg "float64" -> [%expr [%e a] *. [%e b]]
      | _ -> [%expr [%e a] * [%e b]])
  | Sarek_ast.Div -> (
      match repr ty with
      | TReg "float32" | TReg "float64" -> [%expr [%e a] /. [%e b]]
      | _ -> [%expr [%e a] / [%e b]])
  | Sarek_ast.Mod -> [%expr [%e a] mod [%e b]]
  | Sarek_ast.And -> [%expr [%e a] && [%e b]]
  | Sarek_ast.Or -> [%expr [%e a] || [%e b]]
  | Sarek_ast.Eq -> [%expr [%e a] = [%e b]]
  | Sarek_ast.Ne -> [%expr [%e a] <> [%e b]]
  | Sarek_ast.Lt -> [%expr [%e a] < [%e b]]
  | Sarek_ast.Gt -> [%expr [%e a] > [%e b]]
  | Sarek_ast.Le -> [%expr [%e a] <= [%e b]]
  | Sarek_ast.Ge -> [%expr [%e a] >= [%e b]]
  | Sarek_ast.Land -> [%expr [%e a] land [%e b]]
  | Sarek_ast.Lor -> [%expr [%e a] lor [%e b]]
  | Sarek_ast.Lxor -> [%expr [%e a] lxor [%e b]]
  | Sarek_ast.Lsl -> [%expr [%e a] lsl [%e b]]
  | Sarek_ast.Lsr -> [%expr [%e a] lsr [%e b]]
  | Sarek_ast.Asr -> [%expr [%e a] asr [%e b]]

(** Generate unary operation *)
and gen_unop ~loc op a ty : expression =
  match op with
  | Sarek_ast.Neg -> (
      match repr ty with
      | TReg "float32" | TReg "float64" -> [%expr -.[%e a]]
      | _ -> [%expr -[%e a]])
  | Sarek_ast.Not -> [%expr not [%e a]]
  | Sarek_ast.Lnot -> [%expr lnot [%e a]]

(** {1 Kernel Generation} *)

(** Generate the cpu_kern function from a typed kernel.

    The generated function has type: Sarek_cpu_runtime.thread_state -> (arg1,
    arg2, ...) -> unit

    Parameters are passed as a tuple of bigarrays and scalars. *)
let gen_cpu_kern ~loc (kernel : tkernel) : expression =
  let body_e = gen_expr ~loc kernel.tkern_body in

  (* Build parameter tuple pattern *)
  let param_pats =
    List.map
      (fun p ->
        Ast_builder.Default.ppat_var ~loc {txt = var_name p.tparam_id; loc})
      kernel.tkern_params
  in
  let params_pat =
    match param_pats with
    | [] -> [%pat? ()]
    | [p] -> p
    | ps -> Ast_builder.Default.ppat_tuple ~loc ps
  in

  (* Build the function:
     fun __state (arg0, arg1, ...) -> body *)
  let state_pat = Ast_builder.Default.ppat_var ~loc {txt = state_var; loc} in

  [%expr
    fun ([%p state_pat] : Sarek.Sarek_cpu_runtime.thread_state)
        [%p params_pat] -> [%e body_e]]
