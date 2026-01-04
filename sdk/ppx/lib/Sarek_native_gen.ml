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

(** {1 Intrinsic Mapping}

    Map Sarek intrinsics to their OCaml equivalents. For cpu_kern, we call the
    OCaml implementations directly rather than generating GPU code. *)

(** Kernel generation mode for simple vs full execution - defined early for use
    below *)
type gen_mode =
  | FullMode  (** Standard mode - uses thread_state for all indices *)
  | Simple1DMode  (** Simple 1D - gid_x passed directly as int32 *)
  | Simple2DMode  (** Simple 2D - gid_x, gid_y passed directly *)
  | Simple3DMode  (** Simple 3D - gid_x, gid_y, gid_z passed directly *)

(** Variable names for simple mode global indices *)
let gid_x_var = "__gid_x"

let gid_y_var = "__gid_y"

let gid_z_var = "__gid_z"

(** Map stdlib module paths to their runtime module paths. Sarek stdlib modules
    need to be mapped to their actual OCaml locations. *)
let map_stdlib_path path =
  match path with
  | ["Float32"] | ["Sarek_stdlib"; "Float32"] ->
      ["Sarek"; "Sarek_cpu_runtime"; "Float32"]
  | ["Float64"] | ["Sarek_stdlib"; "Float64"] ->
      (* Float64 is just OCaml float, use stdlib *)
      ["Float"]
  | ["Int32"] | ["Sarek_stdlib"; "Int32"] -> ["Int32"]
  | ["Int64"] | ["Sarek_stdlib"; "Int64"] -> ["Int64"]
  | _ -> path

(** Generate intrinsic constant based on generation mode. For simple modes,
    global indices are passed as direct parameters. For full mode, all indices
    come from thread_state. *)
let gen_intrinsic_const ~loc ~gen_mode (ref : Sarek_env.intrinsic_ref) :
    expression =
  (* Helper for simple mode - use direct gid variables for global indices *)
  let use_simple_gid name =
    match (gen_mode, name) with
    (* In simple modes, global_idx_x/global_thread_id is the __gid_x parameter *)
    | ( (Simple1DMode | Simple2DMode | Simple3DMode),
        ("global_idx_x" | "global_thread_id") ) ->
        Some (evar ~loc gid_x_var)
    | (Simple2DMode | Simple3DMode), "global_idx_y" ->
        Some (evar ~loc gid_y_var)
    | Simple3DMode, "global_idx_z" -> Some (evar ~loc gid_z_var)
    | _ -> None
  in
  match ref with
  | Sarek_env.CorePrimitiveRef name -> (
      (* Check if this can be simplified for simple modes *)
      match use_simple_gid name with
      | Some e -> e
      | None -> (
          let state = evar ~loc state_var in
          match name with
          | "thread_idx_x" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.thread_idx_x]
          | "thread_idx_y" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.thread_idx_y]
          | "thread_idx_z" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.thread_idx_z]
          | "block_idx_x" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.block_idx_x]
          | "block_idx_y" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.block_idx_y]
          | "block_idx_z" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.block_idx_z]
          | "block_dim_x" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_x]
          | "block_dim_y" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_y]
          | "block_dim_z" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_z]
          | "grid_dim_x" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.grid_dim_x]
          | "grid_dim_y" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.grid_dim_y]
          | "grid_dim_z" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.grid_dim_z]
          | "global_idx_x" ->
              [%expr Sarek.Sarek_cpu_runtime.global_idx_x [%e state]]
          | "global_idx_y" ->
              [%expr Sarek.Sarek_cpu_runtime.global_idx_y [%e state]]
          | "global_idx_z" ->
              [%expr Sarek.Sarek_cpu_runtime.global_idx_z [%e state]]
          | _ ->
              (* Unknown - try to generate call to Gpu module *)
              evar_qualified ~loc ["Gpu"] name))
  | Sarek_env.IntrinsicRef (path, name) -> (
      (* Check if this can be simplified for simple modes *)
      match use_simple_gid name with
      | Some e -> e
      | None -> (
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
          | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "global_idx_x" ->
              [%expr Sarek.Sarek_cpu_runtime.global_idx_x [%e state]]
          | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "global_idx_y" ->
              [%expr Sarek.Sarek_cpu_runtime.global_idx_y [%e state]]
          | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "global_idx_z" ->
              [%expr Sarek.Sarek_cpu_runtime.global_idx_z [%e state]]
          | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "global_thread_id" ->
              (* global_thread_id is alias for global_idx_x *)
              [%expr Sarek.Sarek_cpu_runtime.global_idx_x [%e state]]
          | _ ->
              (* For other intrinsic constants from stdlib modules, look up via module path *)
              evar_qualified ~loc (map_stdlib_path path) name))

(** Generate intrinsic function call based on generation mode. For simple modes,
    global indices are direct parameters. For full mode, indices come from
    thread_state. *)
let gen_intrinsic_fun ~loc ~gen_mode (ref : Sarek_env.intrinsic_ref)
    (args : expression list) : expression =
  (* Helper for simple mode global index functions *)
  let use_simple_gid_fn name =
    match (gen_mode, name) with
    | ( (Simple1DMode | Simple2DMode | Simple3DMode),
        ("global_idx" | "global_idx_x" | "global_thread_id") ) ->
        Some (evar ~loc gid_x_var)
    | (Simple2DMode | Simple3DMode), "global_idx_y" ->
        Some (evar ~loc gid_y_var)
    | Simple3DMode, "global_idx_z" -> Some (evar ~loc gid_z_var)
    | _ -> None
  in
  match ref with
  | Sarek_env.CorePrimitiveRef name -> (
      match use_simple_gid_fn name with
      | Some e -> e
      | None -> (
          let state = evar ~loc state_var in
          match name with
          | "block_barrier" | "warp_barrier" ->
              (* Call the barrier function from thread state *)
              [%expr [%e state].Sarek.Sarek_cpu_runtime.barrier ()]
          | "global_idx" | "global_thread_id" ->
              [%expr Sarek.Sarek_cpu_runtime.global_idx_x [%e state]]
          | "global_size" ->
              [%expr Sarek.Sarek_cpu_runtime.global_size_x [%e state]]
          | _ ->
              (* Try Gpu module *)
              let fn = evar_qualified ~loc ["Gpu"] name in
              Ast_builder.Default.pexp_apply
                ~loc
                fn
                (List.map (fun a -> (Nolabel, a)) args)))
  | Sarek_env.IntrinsicRef (path, name) -> (
      match use_simple_gid_fn name with
      | Some e -> e
      | None -> (
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
              let fn = evar_qualified ~loc (map_stdlib_path path) name in
              match args with
              | [] -> fn
              | _ ->
                  Ast_builder.Default.pexp_apply
                    ~loc
                    fn
                    (List.map (fun a -> (Nolabel, a)) args))))

(** {1 Expression Generation} *)

(** Mutable variable tracking. We track which variable IDs are mutable so that
    TEVar can dereference them. *)
module IntSet = Set.Make (Int)

(** Set of inline type names for first-class module approach *)
module StringSet = Set.Make (String)

(** Expression generation context *)
type gen_context = {
  mut_vars : IntSet.t;
      (** Variable IDs that are mutable (need dereferencing) *)
  inline_types : StringSet.t;
      (** Type names that use first-class module accessors *)
  current_module : string option;
      (** Current module name for same-module type detection *)
  gen_mode : gen_mode;
      (** Generation mode - affects how thread indices are accessed *)
}

(** Empty generation context *)
let empty_ctx =
  {
    mut_vars = IntSet.empty;
    inline_types = StringSet.empty;
    current_module = None;
    gen_mode = FullMode;
  }

(** Check if a qualified type name is from the current module. For
    "Test_registered_variant.color", returns true if current_module is
    "Test_registered_variant". *)
let is_same_module ctx type_name =
  match ctx.current_module with
  | None -> false
  | Some cur_mod -> (
      match String.rindex_opt type_name '.' with
      | Some idx ->
          let type_mod = String.sub type_name 0 idx in
          String.equal type_mod cur_mod
      | None -> false)

(** The name of the first-class module variable *)
let types_module_var = "__types"

(** {1 First-Class Module Name Helpers} *)

(** Generate accessor function name for a field getter *)
let field_getter_name type_name field_name =
  Printf.sprintf "get_%s_%s" type_name field_name

(** Generate constructor function name for a record *)
let record_maker_name type_name = Printf.sprintf "make_%s" type_name

(** Generate constructor function name for a variant constructor *)
let variant_ctor_name type_name ctor_name =
  Printf.sprintf "make_%s_%s" type_name ctor_name

(** Generate OCaml expression from typed Sarek expression.
    @param ctx Generation context with mutable vars and inline types *)
let rec gen_expr_impl ~loc:_ ~ctx (te : texpr) : expression =
  let loc = ppxlib_loc_of_sarek te.te_loc in
  (* Helper to recursively generate, passing ctx *)
  let gen_expr ~loc e = gen_expr_impl ~loc ~ctx e in
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
         - Qualified names (like "Visibility_lib.public_add")
         OCaml handles shadowing correctly, so using original names is safe. *)
      let var_e =
        if String.contains name '.' then
          (* Qualified name - build a proper Ldot path.
             For stdlib modules like "Float32.of_float", we need to map
             to the runtime path. *)
          let parts = String.split_on_char '.' name in
          (* Split into module path and function name *)
          let module_path, func_name =
            match List.rev parts with
            | fn :: rest -> (List.rev rest, fn)
            | [] -> assert false
          in
          (* Map stdlib module paths to runtime locations *)
          let mapped_path = map_stdlib_path module_path in
          evar_qualified ~loc mapped_path func_name
        else evar ~loc name
      in
      if IntSet.mem id ctx.mut_vars then
        (* Mutable variable - dereference the ref *)
        [%expr ![%e var_e]]
      else var_e
  (* Vector/array access - V2 only *)
  | TEVecGet (vec, idx) ->
      let vec_e = gen_expr ~loc vec in
      let idx_e = gen_expr ~loc idx in
      [%expr Spoc_core.Vector.get [%e vec_e] (Int32.to_int [%e idx_e])]
  | TEVecSet (vec, idx, value) ->
      let vec_e = gen_expr ~loc vec in
      let idx_e = gen_expr ~loc idx in
      let val_e = gen_expr ~loc value in
      [%expr
        Spoc_core.Vector.set [%e vec_e] (Int32.to_int [%e idx_e]) [%e val_e]]
  (* Array access - for shared memory (regular OCaml arrays) *)
  | TEArrGet (arr, idx) ->
      let arr_e = gen_expr ~loc arr in
      let idx_e = gen_expr ~loc idx in
      [%expr [%e arr_e].(Int32.to_int [%e idx_e])]
  | TEArrSet (arr, idx, value) ->
      let arr_e = gen_expr ~loc arr in
      let idx_e = gen_expr ~loc idx in
      let val_e = gen_expr ~loc value in
      [%expr [%e arr_e].(Int32.to_int [%e idx_e]) <- [%e val_e]]
  (* Record field access - qualify field with module path from record type.
     For Geometry_lib.point, we need p.Geometry_lib.x, not p.x.
     For inline types using first-class modules, call __types.get_type_field.
     For same-module types, use unqualified field names. *)
  | TEFieldGet (record, field_name, _field_idx) -> (
      let rec_e = gen_expr ~loc record in
      match repr record.ty with
      | TRecord (type_name, _) -> (
          (* type_name may be "Module.type" or just "type" *)
          match String.rindex_opt type_name '.' with
          | Some _ when is_same_module ctx type_name ->
              (* Same-module type - use unqualified field access *)
              let field_lid = {txt = Lident field_name; loc} in
              Ast_builder.Default.pexp_field ~loc rec_e field_lid
          | Some idx ->
              (* External qualified type - use qualified field access *)
              let module_path = String.sub type_name 0 idx in
              let parts = String.split_on_char '.' module_path in
              let rec build_lid = function
                | [] -> Lident field_name
                | [m] -> Ldot (Lident m, field_name)
                | m :: rest -> Ldot (build_lid rest, m)
              in
              let field_lid = {txt = build_lid (List.rev parts); loc} in
              Ast_builder.Default.pexp_field ~loc rec_e field_lid
          | None ->
              (* Inline type - check if using first-class module approach *)
              if StringSet.mem type_name ctx.inline_types then
                (* Use accessor: __types#get_typename_fieldname record *)
                let fn_name = field_getter_name type_name field_name in
                let method_call =
                  Ast_builder.Default.pexp_send
                    ~loc
                    (evar ~loc types_module_var)
                    {txt = fn_name; loc}
                in
                Ast_builder.Default.pexp_apply
                  ~loc
                  method_call
                  [(Nolabel, rec_e)]
              else
                (* Direct field access *)
                let field_lid = {txt = Lident field_name; loc} in
                Ast_builder.Default.pexp_field ~loc rec_e field_lid)
      | _ ->
          (* Not a record type - shouldn't happen, but use unqualified name *)
          let field_lid = {txt = Lident field_name; loc} in
          Ast_builder.Default.pexp_field ~loc rec_e field_lid)
  | TEFieldSet (record, field_name, _field_idx, value) ->
      let rec_e = gen_expr ~loc record in
      let val_e = gen_expr ~loc value in
      (* Note: mutable record fields with first-class modules would need setters.
         For now we don't support mutable fields in inline types with FCM. *)
      let field_lid =
        match repr record.ty with
        | TRecord (type_name, _) -> (
            match String.rindex_opt type_name '.' with
            | Some _ when is_same_module ctx type_name ->
                (* Same-module type - use unqualified field access *)
                {txt = Lident field_name; loc}
            | Some idx ->
                (* External qualified type - use qualified field access *)
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
      let ctx' = {ctx with mut_vars = IntSet.add id ctx.mut_vars} in
      let body_e = gen_expr_impl ~loc ~ctx:ctx' body in
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
          (* OCaml for loops are inclusive on both ends, just like Sarek.
             for i = 0 to k - 1l means iterate from 0 to k-1 inclusive. *)
          [%expr
            for
              [%p int_var_pat] = Int32.to_int [%e lo_e]
              to Int32.to_int [%e hi_e]
            do
              [%e wrapped_body]
            done]
      | Sarek_ast.Downto ->
          [%expr
            for
              [%p int_var_pat] = Int32.to_int [%e hi_e]
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
            let pat_e = gen_pattern_impl ~loc ~ctx pat in
            let body_e = gen_expr ~loc body in
            Ast_builder.Default.case ~lhs:pat_e ~guard:None ~rhs:body_e)
          cases
      in
      Ast_builder.Default.pexp_match ~loc scrut_e cases_e
  (* Record construction - qualify field names with module path from type_name.
     For inline types with FCM, use __types.make_typename ~field1:v1 ~field2:v2.
     For same-module types, use unqualified field names. *)
  | TERecord (type_name, fields) -> (
      match String.rindex_opt type_name '.' with
      | Some _ when is_same_module ctx type_name ->
          (* Same-module type - use unqualified record construction *)
          let fields_e =
            List.map
              (fun (name, expr) ->
                ({txt = Lident name; loc}, gen_expr ~loc expr))
              fields
          in
          Ast_builder.Default.pexp_record ~loc fields_e None
      | Some idx ->
          (* External qualified type - use qualified field names *)
          let module_path = String.sub type_name 0 idx in
          let parts = String.split_on_char '.' module_path in
          let field_lid name =
            let rec build_lid = function
              | [] -> Lident name
              | [m] -> Ldot (Lident m, name)
              | m :: rest -> Ldot (build_lid rest, m)
            in
            {txt = build_lid (List.rev parts); loc}
          in
          let fields_e =
            List.map
              (fun (name, expr) -> (field_lid name, gen_expr ~loc expr))
              fields
          in
          Ast_builder.Default.pexp_record ~loc fields_e None
      | None ->
          (* Inline type - check if using first-class module approach *)
          if StringSet.mem type_name ctx.inline_types then
            (* Use maker: __types#make_typename ~field1:v1 ~field2:v2 *)
            let fn_name = record_maker_name type_name in
            let method_call =
              Ast_builder.Default.pexp_send
                ~loc
                (evar ~loc types_module_var)
                {txt = fn_name; loc}
            in
            (* Build labelled arguments *)
            let args =
              List.map
                (fun (name, expr) -> (Labelled name, gen_expr ~loc expr))
                fields
            in
            Ast_builder.Default.pexp_apply ~loc method_call args
          else
            (* Direct record construction *)
            let fields_e =
              List.map
                (fun (name, expr) ->
                  ({txt = Lident name; loc}, gen_expr ~loc expr))
                fields
            in
            Ast_builder.Default.pexp_record ~loc fields_e None)
  (* Variant construction - qualify constructor with module path.
     For inline types with FCM, use __types.make_typename_Ctor arg.
     For same-module types, use unqualified constructors. *)
  | TEConstr (type_name, constr_name, arg) -> (
      match String.rindex_opt type_name '.' with
      | Some _ when is_same_module ctx type_name ->
          (* Same-module type - use unqualified constructor *)
          let arg_e = Option.map (gen_expr ~loc) arg in
          Ast_builder.Default.pexp_construct
            ~loc
            {txt = Lident constr_name; loc}
            arg_e
      | Some idx ->
          (* External qualified type - use qualified constructor *)
          let module_path = String.sub type_name 0 idx in
          let parts = String.split_on_char '.' module_path in
          let rec build_lid = function
            | [] -> Lident constr_name
            | [m] -> Ldot (Lident m, constr_name)
            | m :: rest -> Ldot (build_lid rest, m)
          in
          let constr_lid = build_lid (List.rev parts) in
          let arg_e = Option.map (gen_expr ~loc) arg in
          Ast_builder.Default.pexp_construct ~loc {txt = constr_lid; loc} arg_e
      | None ->
          (* Inline type - check if using first-class module approach *)
          if StringSet.mem type_name ctx.inline_types then
            (* Use maker: __types#make_typename_Ctor arg or __types#make_typename_Ctor () *)
            let fn_name = variant_ctor_name type_name constr_name in
            let method_call =
              Ast_builder.Default.pexp_send
                ~loc
                (evar ~loc types_module_var)
                {txt = fn_name; loc}
            in
            let arg_e =
              match arg with Some a -> gen_expr ~loc a | None -> [%expr ()]
            in
            Ast_builder.Default.pexp_apply ~loc method_call [(Nolabel, arg_e)]
          else
            (* Direct constructor *)
            let arg_e = Option.map (gen_expr ~loc) arg in
            Ast_builder.Default.pexp_construct
              ~loc
              {txt = Lident constr_name; loc}
              arg_e)
  (* Tuple *)
  | TETuple exprs ->
      let exprs_e = List.map (gen_expr ~loc) exprs in
      Ast_builder.Default.pexp_tuple ~loc exprs_e
  (* Return - just evaluate the expression *)
  | TEReturn e -> gen_expr ~loc e
  (* Create local array - use regular OCaml arrays for native mode *)
  | TECreateArray (size, elem_ty, _memspace) ->
      let size_e = gen_expr ~loc size in
      (* Generate default value based on element type *)
      let default_e =
        match repr elem_ty with
        | TReg "float32" | TReg "float64" -> [%expr 0.0]
        | TPrim TInt32 | TReg "int32" -> [%expr 0l]
        | TReg "int64" -> [%expr 0L]
        | TReg "char" -> [%expr '\000']
        | _ -> [%expr Obj.magic 0]
        (* Fallback for custom types *)
      in
      [%expr Array.make [%e size_e] [%e default_e]]
  (* Global ref - reference to external value *)
  | TEGlobalRef (name, _typ) ->
      (* Dereference the ref *)
      let var_e = evar ~loc name in
      [%expr ![%e var_e]]
  (* Native code with OCaml fallback - use the OCaml expression directly *)
  | TENative {ocaml; _} ->
      (* The ocaml expression is a function that will be applied to arguments.
         Return it as-is; TEApp will handle the application. *)
      ocaml
  (* Pragma - just evaluate body (pragmas are hints for GPU) *)
  | TEPragma (_opts, body) -> gen_expr ~loc body
  (* Intrinsic constant - thread indices, etc. *)
  | TEIntrinsicConst ref -> gen_intrinsic_const ~loc ~gen_mode:ctx.gen_mode ref
  (* Intrinsic function - math functions, barriers, etc. *)
  | TEIntrinsicFun (ref, _convergence, args) ->
      let args_e = List.map (gen_expr ~loc) args in
      gen_intrinsic_fun ~loc ~gen_mode:ctx.gen_mode ref args_e
  (* BSP let%shared - allocate shared memory using OCaml arrays *)
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
      let shared = evar ~loc shared_var in
      let body_e = gen_expr ~loc body in
      let pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
      let name_e = Ast_builder.Default.estring ~loc name in
      (* Use typed allocators for common types, generic for custom types *)
      let alloc_expr =
        match repr elem_ty with
        | TReg "float32" | TReg "float64" ->
            [%expr
              Sarek.Sarek_cpu_runtime.alloc_shared_float
                [%e shared]
                [%e name_e]
                [%e size_e]
                0.0]
        | TPrim TInt32 | TReg "int32" ->
            [%expr
              Sarek.Sarek_cpu_runtime.alloc_shared_int32
                [%e shared]
                [%e name_e]
                [%e size_e]
                0l]
        | TReg "int64" ->
            [%expr
              Sarek.Sarek_cpu_runtime.alloc_shared_int64
                [%e shared]
                [%e name_e]
                [%e size_e]
                0L]
        | TReg "int" ->
            [%expr
              Sarek.Sarek_cpu_runtime.alloc_shared_int
                [%e shared]
                [%e name_e]
                [%e size_e]
                0]
        | TReg "char" ->
            [%expr
              Sarek.Sarek_cpu_runtime.alloc_shared
                [%e shared]
                [%e name_e]
                [%e size_e]
                '\000']
        | _ ->
            (* Fallback for custom types - uses Obj.magic *)
            [%expr
              Sarek.Sarek_cpu_runtime.alloc_shared
                [%e shared]
                [%e name_e]
                [%e size_e]
                (Obj.magic 0)]
      in
      [%expr
        let [%p pat] = [%e alloc_expr] in
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
  | TELetRec (name, _id, params, fn_body, cont) ->
      (* Generate: let rec name p1 p2 ... = body in cont *)
      let fn_body_e = gen_expr ~loc fn_body in
      let cont_e = gen_expr ~loc cont in
      (* Create function with parameters *)
      let fn_expr =
        List.fold_right
          (fun p acc ->
            let pvar =
              Ast_builder.Default.ppat_var ~loc {txt = p.tparam_name; loc}
            in
            Ast_builder.Default.pexp_fun ~loc Nolabel None pvar acc)
          params
          fn_body_e
      in
      let binding =
        Ast_builder.Default.value_binding
          ~loc
          ~pat:(Ast_builder.Default.ppat_var ~loc {txt = name; loc})
          ~expr:fn_expr
      in
      Ast_builder.Default.pexp_let ~loc Recursive [binding] cont_e

(** Generate pattern from typed pattern. Takes context to detect same-module
    types that shouldn't be qualified. *)
and gen_pattern_impl ~loc:_ ~ctx (tpat : tpattern) : pattern =
  let loc = ppxlib_loc_of_sarek tpat.tpat_loc in
  match tpat.tpat with
  | TPAny -> Ast_builder.Default.ppat_any ~loc
  | TPVar (name, _id) ->
      (* Use original name for pattern variables *)
      Ast_builder.Default.ppat_var ~loc {txt = name; loc}
  | TPConstr (type_name, constr_name, arg) ->
      let arg_p = Option.map (gen_pattern_impl ~loc ~ctx) arg in
      (* Qualify constructor with module path from type_name if present.
         For "Geometry_lib.shape", we need Geometry_lib.Circle, not Circle.
         For inline types or same-module types, use unqualified name. *)
      let constr_lid =
        match String.rindex_opt type_name '.' with
        | Some idx ->
            (* Check if this is from the current module - use unqualified if so *)
            if is_same_module ctx type_name then Lident constr_name
            else
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
      let pats_p = List.map (gen_pattern_impl ~loc ~ctx) pats in
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

(** Extract module name from a Sarek location (file path). For
    "/path/to/test_registered_variant.ml", returns "Test_registered_variant". *)
let module_name_of_sarek_loc (loc : Sarek_ast.loc) : string =
  let file = loc.loc_file in
  let base = Filename.(remove_extension (basename file)) in
  String.capitalize_ascii base

(** Top-level entry point for generating expressions. Starts with empty context.
*)
let gen_expr ~loc e = gen_expr_impl ~loc ~ctx:empty_ctx e

(** Generate expression with inline types context for first-class modules. *)
let gen_expr_with_inline_types ~loc ~inline_type_names ~current_module e =
  let ctx = {empty_ctx with inline_types = inline_type_names; current_module} in
  gen_expr_impl ~loc ~ctx e

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

(** Wrap expression with module item bindings. *)
let wrap_module_items ~loc (items : tmodule_item list) (body : expression) :
    expression =
  List.fold_right
    (fun item acc ->
      match item with
      | TMFun (name, is_rec, params, item_body) ->
          let pat, expr = gen_module_fun ~loc name params item_body in
          if is_rec then
            (* Generate let rec for recursive functions *)
            [%expr
              let rec [%p pat] = [%e expr] in
              [%e acc]]
          else
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

(** {1 First-Class Module Approach for Inline Types}

    To avoid "type escapes its scope" errors, we use first-class modules to
    encapsulate inline types. The approach:

    1. Generate a module signature KERNEL_TYPES with abstract types and
    accessors 2. Generate a concrete module implementation with the actual types
    3. Transform the kernel body to use T.get_field and T.make_type accessors 4.
    Pass (module T : KERNEL_TYPES) as a parameter to the kernel

    This keeps the concrete types hidden behind an existential type. *)

(** Generate a concrete module implementation for inline types (types only).
    Example for record type point with fields x and y: struct type point =
    record with fields x: float and y: float end

    Note: We only generate type declarations here, not accessor functions. The
    accessor functions are generated as object methods by gen_types_object. *)
let gen_module_impl ~loc (decls : ttype_decl list) : module_expr =
  let struct_items =
    List.map
      (fun decl ->
        match decl with
        | TTypeRecord {tdecl_name; tdecl_fields; _} ->
            gen_type_decl_record ~loc tdecl_name tdecl_fields
        | TTypeVariant {tdecl_name; tdecl_constructors; _} ->
            gen_type_decl_variant ~loc tdecl_name tdecl_constructors)
      decls
  in
  Ast_builder.Default.pmod_structure ~loc struct_items

(** Get only inline type declarations (those without module prefix).
    External/registered types have qualified names like "Module.type". *)
let inline_type_decls (decls : ttype_decl list) : ttype_decl list =
  List.filter
    (fun decl ->
      let name =
        match decl with
        | TTypeRecord {tdecl_name; _} -> tdecl_name
        | TTypeVariant {tdecl_name; _} -> tdecl_name
      in
      (* Only include types without '.' - these are inline definitions *)
      not (String.contains name '.'))
    decls

(** Check if a kernel has inline types that need first-class module handling *)
let has_inline_types (kernel : tkernel) : bool =
  inline_type_decls kernel.tkern_type_decls <> []

(** {1 Kernel Generation} *)

(** Convert execution strategy to generation mode *)
let gen_mode_of_exec_strategy = function
  | Sarek_convergence.Simple1D -> Simple1DMode
  | Sarek_convergence.Simple2D -> Simple2DMode
  | Sarek_convergence.Simple3D -> Simple3DMode
  | Sarek_convergence.FullState -> FullMode

(** Generate the cpu_kern function from a typed kernel.

    The generated function has type: Sarek_cpu_runtime.thread_state -> (arg1,
    arg2, ...) -> unit

    Parameters are passed as a tuple of bigarrays and scalars.

    For kernels with inline types, we use first-class modules:
    - The kernel takes an extra __types parameter of type (module KERNEL_TYPES)
    - Field access uses __types.get_type_field
    - Record construction uses __types.make_type *)
let gen_cpu_kern ~loc (kernel : tkernel) : expression =
  (* Check if we need first-class modules for inline types *)
  let use_fcm = has_inline_types kernel in

  (* Build set of inline type names for the context *)
  let inline_type_names =
    if use_fcm then
      List.fold_left
        (fun acc decl ->
          let name =
            match decl with
            | TTypeRecord {tdecl_name; _} -> tdecl_name
            | TTypeVariant {tdecl_name; _} -> tdecl_name
          in
          (* Only include unqualified names (inline types) *)
          if not (String.contains name '.') then StringSet.add name acc else acc)
        StringSet.empty
        kernel.tkern_type_decls
    else StringSet.empty
  in

  (* Extract current module name for same-module type detection *)
  let current_module = Some (module_name_of_sarek_loc kernel.tkern_loc) in

  (* Generate body with appropriate context *)
  let body_e =
    if use_fcm then
      gen_expr_with_inline_types
        ~loc
        ~inline_type_names
        ~current_module
        kernel.tkern_body
    else
      (* Even without FCM, we need current_module for same-module type detection *)
      let ctx = {empty_ctx with current_module} in
      gen_expr_impl ~loc ~ctx kernel.tkern_body
  in

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
     For regular kernels: fun __state __shared (arg0, arg1, ...) -> body
     For FCM kernels: fun __types __state __shared (arg0, arg1, ...) -> body
     where __types is a record with accessor functions.

     We suppress warning 27 (unused-var-strict) since kernel parameters
     may not always be used (e.g., barrier-only kernels), and warnings
     32/33 for unused value bindings. *)
  let state_pat = Ast_builder.Default.ppat_var ~loc {txt = state_var; loc} in
  let shared_pat = Ast_builder.Default.ppat_var ~loc {txt = shared_var; loc} in

  (* The function expression with warning suppression attribute on the body *)
  let inner_fun =
    if use_fcm then
      (* FCM kernel: add __types parameter as first argument *)
      let types_pat =
        Ast_builder.Default.ppat_var ~loc {txt = types_module_var; loc}
      in
      [%expr
        fun [%p types_pat]
            ([%p state_pat] : Sarek.Sarek_cpu_runtime.thread_state)
            ([%p shared_pat] : Sarek.Sarek_cpu_runtime.shared_mem)
            [%p params_pat] -> [%e body_with_items]]
    else
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

(** Generate a simplified cpu_kern for simple 1D/2D/3D kernels.

    For Simple1D: fun __gid_x (args) -> body For Simple2D: fun __gid_x __gid_y
    (args) -> body For Simple3D: fun __gid_x __gid_y __gid_z (args) -> body

    These kernels don't need thread_state or shared memory. The body is
    generated with the appropriate gen_mode so that global_idx_x/y/z references
    are replaced with the __gid_x/y/z parameters. *)
let gen_simple_cpu_kern ~loc ~exec_strategy (kernel : tkernel) : expression =
  (* Build set of inline type names for the context *)
  let use_fcm = has_inline_types kernel in
  let inline_type_names =
    if use_fcm then
      List.fold_left
        (fun acc decl ->
          let name =
            match decl with
            | TTypeRecord {tdecl_name; _} -> tdecl_name
            | TTypeVariant {tdecl_name; _} -> tdecl_name
          in
          if not (String.contains name '.') then StringSet.add name acc else acc)
        StringSet.empty
        kernel.tkern_type_decls
    else StringSet.empty
  in

  (* Extract current module name for same-module type detection *)
  let current_module = Some (module_name_of_sarek_loc kernel.tkern_loc) in

  (* Set generation mode based on strategy *)
  let gen_mode = gen_mode_of_exec_strategy exec_strategy in

  (* Generate body with the simple mode context *)
  let ctx =
    {empty_ctx with current_module; inline_types = inline_type_names; gen_mode}
  in
  let body_e = gen_expr_impl ~loc ~ctx kernel.tkern_body in

  (* Generate inline module items *)
  let inline_items =
    let all_items = kernel.tkern_module_items in
    let skip_count = kernel.tkern_external_item_count in
    let rec drop n lst =
      if n <= 0 then lst
      else match lst with [] -> [] | _ :: tl -> drop (n - 1) tl
    in
    drop skip_count all_items
  in
  let body_with_items = wrap_module_items ~loc inline_items body_e in

  (* Build parameter tuple pattern - use parameter names with type constraints *)
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

  (* Build the function with gid parameters based on strategy:
     Simple1D: fun __gid_x (args) -> body
     Simple2D: fun __gid_x __gid_y (args) -> body
     Simple3D: fun __gid_x __gid_y __gid_z (args) -> body *)
  let gid_x_pat = Ast_builder.Default.ppat_var ~loc {txt = gid_x_var; loc} in
  let gid_y_pat = Ast_builder.Default.ppat_var ~loc {txt = gid_y_var; loc} in
  let gid_z_pat = Ast_builder.Default.ppat_var ~loc {txt = gid_z_var; loc} in

  let inner_fun =
    if use_fcm then
      let types_pat =
        Ast_builder.Default.ppat_var ~loc {txt = types_module_var; loc}
      in
      match exec_strategy with
      | Sarek_convergence.Simple1D ->
          [%expr
            fun [%p types_pat] ([%p gid_x_pat] : int32) [%p params_pat] ->
              [%e body_with_items]]
      | Sarek_convergence.Simple2D ->
          [%expr
            fun [%p types_pat]
                ([%p gid_x_pat] : int32)
                ([%p gid_y_pat] : int32)
                [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.Simple3D ->
          [%expr
            fun [%p types_pat]
                ([%p gid_x_pat] : int32)
                ([%p gid_y_pat] : int32)
                ([%p gid_z_pat] : int32)
                [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.FullState ->
          failwith "gen_simple_cpu_kern called with FullState strategy"
    else
      match exec_strategy with
      | Sarek_convergence.Simple1D ->
          [%expr
            fun ([%p gid_x_pat] : int32) [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.Simple2D ->
          [%expr
            fun ([%p gid_x_pat] : int32)
                ([%p gid_y_pat] : int32)
                [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.Simple3D ->
          [%expr
            fun ([%p gid_x_pat] : int32)
                ([%p gid_y_pat] : int32)
                ([%p gid_z_pat] : int32)
                [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.FullState ->
          failwith "gen_simple_cpu_kern called with FullState strategy"
  in
  (* Add warning suppression attribute to the function *)
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

(** Generate a type cast expression for extracting a kernel argument.

    For vectors: cast Obj.t to Spoc_core.Vector.t (V2 path). For scalars: cast
    Obj.t to the primitive type.

    V2 uses type-safe vectors that abstract over storage. *)
let gen_arg_cast ~loc (param : tparam) (idx : int) : expression =
  let arr_access =
    [%expr Array.get __args [%e Ast_builder.Default.eint ~loc idx]]
  in
  match repr param.tparam_type with
  | TVec _ ->
      (* All vector types use the same generic cast - Vector.get/set handle the types *)
      [%expr (Obj.obj [%e arr_access] : (_, _) Spoc_core.Vector.t)]
  | TReg "float32" -> [%expr (Obj.obj [%e arr_access] : float)]
  | TReg "float64" -> [%expr (Obj.obj [%e arr_access] : float)]
  | TReg "int32" | TPrim TInt32 -> [%expr (Obj.obj [%e arr_access] : int32)]
  | TReg "int64" -> [%expr (Obj.obj [%e arr_access] : int64)]
  | TPrim TBool -> [%expr (Obj.obj [%e arr_access] : bool)]
  | _ ->
      (* Default - just return as Obj.t and let caller deal with it *)
      arr_access

(** Generate an object expression with accessor methods for FCM. Example for
    type point with fields x and y: object method get_point_x r = r.x method
    get_point_y r = r.y method make_point ~x ~y = record with fields x and y end

    Using an object avoids needing to define a record type for the accessors. *)
let gen_types_object ~loc (decls : ttype_decl list) : expression =
  let methods =
    List.concat_map
      (fun decl ->
        match decl with
        | TTypeRecord {tdecl_name; tdecl_fields; _} ->
            (* Getters *)
            let getters =
              List.map
                (fun (fname, _fty, _is_mut) ->
                  let fn_name = field_getter_name tdecl_name fname in
                  let field_lid = {txt = Lident fname; loc} in
                  let fn_expr =
                    [%expr
                      fun __r ->
                        [%e
                          Ast_builder.Default.pexp_field
                            ~loc
                            [%expr __r]
                            field_lid]]
                  in
                  Ast_builder.Default.pcf_method
                    ~loc
                    ({txt = fn_name; loc}, Public, Cfk_concrete (Fresh, fn_expr)))
                tdecl_fields
            in
            (* Maker *)
            let maker =
              let fn_name = record_maker_name tdecl_name in
              let param_pats =
                List.map
                  (fun (fname, _fty, _) ->
                    ( Labelled fname,
                      Ast_builder.Default.ppat_var ~loc {txt = fname; loc} ))
                  tdecl_fields
              in
              let record_fields =
                List.map
                  (fun (fname, _, _) ->
                    ( {txt = Lident fname; loc},
                      Ast_builder.Default.pexp_ident
                        ~loc
                        {txt = Lident fname; loc} ))
                  tdecl_fields
              in
              let record_expr =
                Ast_builder.Default.pexp_record ~loc record_fields None
              in
              let fn_expr =
                List.fold_right
                  (fun (lbl, pat) body ->
                    Ast_builder.Default.pexp_fun ~loc lbl None pat body)
                  param_pats
                  record_expr
              in
              Ast_builder.Default.pcf_method
                ~loc
                ({txt = fn_name; loc}, Public, Cfk_concrete (Fresh, fn_expr))
            in
            getters @ [maker]
        | TTypeVariant {tdecl_name; tdecl_constructors; _} ->
            (* Constructor functions *)
            List.map
              (fun (cname, arg_opt) ->
                let fn_name = variant_ctor_name tdecl_name cname in
                let ctor_lid = {txt = Lident cname; loc} in
                let fn_expr =
                  match arg_opt with
                  | None ->
                      [%expr
                        fun () ->
                          [%e
                            Ast_builder.Default.pexp_construct
                              ~loc
                              ctor_lid
                              None]]
                  | Some _ ->
                      [%expr
                        fun __x ->
                          [%e
                            Ast_builder.Default.pexp_construct
                              ~loc
                              ctor_lid
                              (Some [%expr __x])]]
                in
                Ast_builder.Default.pcf_method
                  ~loc
                  ({txt = fn_name; loc}, Public, Cfk_concrete (Fresh, fn_expr)))
              tdecl_constructors)
      decls
  in
  Ast_builder.Default.pexp_object
    ~loc
    (Ast_builder.Default.class_structure
       ~self:(Ast_builder.Default.ppat_any ~loc)
       ~fields:methods)

(** Generate the cpu_kern wrapper that matches the kirc_kernel.cpu_kern type.

    Generated function type: block:int*int*int -> grid:int*int*int -> Obj.t
    array -> unit

    This wrapper: 1. Extracts and casts each argument from the Obj.t array 2.
    Builds the args tuple 3. Calls run_sequential with the native kernel

    For kernels with inline types, we use first-class modules to avoid "type
    escapes its scope" errors. The types module is created inside the wrapper
    and passed to the kernel.

    OPTIMIZATION: For simple 1D/2D/3D kernels that don't use block/thread
    indices, shared memory, or barriers, we generate an optimized path that uses
    run_1d/2d/3d_threadpool. This eliminates the per-element thread_state
    overhead. *)
let gen_cpu_kern_wrapper ~loc (kernel : tkernel) : expression =
  (* V2: Uses Spoc_core.Vector with get/set for type-safe access.
     Record/variant vectors are supported via custom_type descriptors. *)
  let use_fcm = has_inline_types kernel in
  let native_kern = gen_cpu_kern ~loc kernel in

  (* Detect execution strategy for optimization *)
  let exec_strategy = Sarek_convergence.kernel_exec_strategy kernel in

  (* Detect barrier usage at compile time - passed to runtime *)
  let has_barriers = Sarek_convergence.kernel_uses_barriers kernel in
  let has_barriers_expr = Ast_builder.Default.ebool ~loc has_barriers in

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

  (* Generate the simple kernel expression if needed *)
  let simple_kern_opt =
    match exec_strategy with
    | Sarek_convergence.Simple1D | Sarek_convergence.Simple2D
    | Sarek_convergence.Simple3D ->
        Some (gen_simple_cpu_kern ~loc ~exec_strategy kernel)
    | Sarek_convergence.FullState -> None
  in

  (* For simple kernels, generate optimized threadpool call:
     - Simple1D: run_1d_threadpool ~total_x:(bx*gx) simple_kern args
     - Simple2D: run_2d_threadpool ~width:(bx*gx) ~height:(by*gy) simple_kern args
     - Simple3D: run_3d_threadpool ~width ~height ~depth simple_kern args
     For FCM kernels, partially apply __types_rec first.
     Note: __simple_kern is bound in the wrapper body. *)
  let gen_simple_threadpool_call () =
    match exec_strategy with
    | Sarek_convergence.Simple1D ->
        if use_fcm then
          [%expr
            let bx, _, _ = block in
            let gx, _, _ = grid in
            Sarek.Sarek_cpu_runtime.run_1d_threadpool
              ~total_x:(bx * gx)
              (fun gid_x args -> __simple_kern __types_rec gid_x args)
              [%e args_tuple]]
        else
          [%expr
            let bx, _, _ = block in
            let gx, _, _ = grid in
            Sarek.Sarek_cpu_runtime.run_1d_threadpool
              ~total_x:(bx * gx)
              __simple_kern
              [%e args_tuple]]
    | Sarek_convergence.Simple2D ->
        if use_fcm then
          [%expr
            let bx, by, _ = block in
            let gx, gy, _ = grid in
            Sarek.Sarek_cpu_runtime.run_2d_threadpool
              ~width:(bx * gx)
              ~height:(by * gy)
              (fun gid_x gid_y args ->
                __simple_kern __types_rec gid_x gid_y args)
              [%e args_tuple]]
        else
          [%expr
            let bx, by, _ = block in
            let gx, gy, _ = grid in
            Sarek.Sarek_cpu_runtime.run_2d_threadpool
              ~width:(bx * gx)
              ~height:(by * gy)
              __simple_kern
              [%e args_tuple]]
    | Sarek_convergence.Simple3D ->
        if use_fcm then
          [%expr
            let bx, by, bz = block in
            let gx, gy, gz = grid in
            Sarek.Sarek_cpu_runtime.run_3d_threadpool
              ~width:(bx * gx)
              ~height:(by * gy)
              ~depth:(bz * gz)
              (fun gid_x gid_y gid_z args ->
                __simple_kern __types_rec gid_x gid_y gid_z args)
              [%e args_tuple]]
        else
          [%expr
            let bx, by, bz = block in
            let gx, gy, gz = grid in
            Sarek.Sarek_cpu_runtime.run_3d_threadpool
              ~width:(bx * gx)
              ~height:(by * gy)
              ~depth:(bz * gz)
              __simple_kern
              [%e args_tuple]]
    | Sarek_convergence.FullState ->
        failwith "gen_simple_threadpool_call called with FullState"
  in

  (* For simple kernels in Threadpool mode, use the optimized path.
     For all other cases, use the standard path. *)
  let run_call =
    match exec_strategy with
    | Sarek_convergence.Simple1D | Sarek_convergence.Simple2D
    | Sarek_convergence.Simple3D ->
        (* Simple kernel - use optimized path for Threadpool *)
        let simple_call = gen_simple_threadpool_call () in
        if use_fcm then
          [%expr
            match __mode with
            | Sarek.Sarek_cpu_runtime.Sequential ->
                Sarek.Sarek_cpu_runtime.run_sequential
                  ~block
                  ~grid
                  (__native_kern __types_rec)
                  [%e args_tuple]
            | Sarek.Sarek_cpu_runtime.Threadpool -> [%e simple_call]
            | Sarek.Sarek_cpu_runtime.Parallel ->
                Sarek.Sarek_cpu_runtime.run_parallel
                  ~block
                  ~grid
                  (__native_kern __types_rec)
                  [%e args_tuple]]
        else
          [%expr
            match __mode with
            | Sarek.Sarek_cpu_runtime.Sequential ->
                Sarek.Sarek_cpu_runtime.run_sequential
                  ~block
                  ~grid
                  __native_kern
                  [%e args_tuple]
            | Sarek.Sarek_cpu_runtime.Threadpool -> [%e simple_call]
            | Sarek.Sarek_cpu_runtime.Parallel ->
                Sarek.Sarek_cpu_runtime.run_parallel
                  ~block
                  ~grid
                  __native_kern
                  [%e args_tuple]]
    | Sarek_convergence.FullState ->
        (* Complex kernel - use standard path for all modes *)
        if use_fcm then
          [%expr
            match __mode with
            | Sarek.Sarek_cpu_runtime.Sequential ->
                Sarek.Sarek_cpu_runtime.run_sequential
                  ~block
                  ~grid
                  (__native_kern __types_rec)
                  [%e args_tuple]
            | Sarek.Sarek_cpu_runtime.Threadpool ->
                Sarek.Sarek_cpu_runtime.run_threadpool
                  ~has_barriers:[%e has_barriers_expr]
                  ~block
                  ~grid
                  (__native_kern __types_rec)
                  [%e args_tuple]
            | Sarek.Sarek_cpu_runtime.Parallel ->
                Sarek.Sarek_cpu_runtime.run_parallel
                  ~block
                  ~grid
                  (__native_kern __types_rec)
                  [%e args_tuple]]
        else
          [%expr
            match __mode with
            | Sarek.Sarek_cpu_runtime.Sequential ->
                Sarek.Sarek_cpu_runtime.run_sequential
                  ~block
                  ~grid
                  __native_kern
                  [%e args_tuple]
            | Sarek.Sarek_cpu_runtime.Threadpool ->
                Sarek.Sarek_cpu_runtime.run_threadpool
                  ~has_barriers:[%e has_barriers_expr]
                  ~block
                  ~grid
                  __native_kern
                  [%e args_tuple]
            | Sarek.Sarek_cpu_runtime.Parallel ->
                Sarek.Sarek_cpu_runtime.run_parallel
                  ~block
                  ~grid
                  __native_kern
                  [%e args_tuple]]
  in

  (* Build the nested let bindings *)
  let body_with_bindings =
    List.fold_right
      (fun (pat, expr) body ->
        [%expr
          let [%p pat] = [%e expr] in
          [%e body]])
      arg_bindings
      run_call
  in

  if use_fcm then
    (* For FCM kernels, create the types object inside a local module
       to define the concrete types, then extract the accessor object.
       Only include inline types (not external/registered types). *)
    let inline_decls = inline_type_decls kernel.tkern_type_decls in
    let types_object = gen_types_object ~loc inline_decls in
    let types_impl = gen_module_impl ~loc inline_decls in
    (* Build: let module __Types = <impl> in let open __Types in ... *)
    let inner_body =
      (* let open __Types in let __types_rec = ... *)
      Ast_builder.Default.pexp_open
        ~loc
        (Ast_builder.Default.open_infos
           ~loc
           ~override:Fresh
           ~expr:
             (Ast_builder.Default.pmod_ident ~loc {txt = Lident "__Types"; loc}))
        (match simple_kern_opt with
        | Some simple_kern ->
            [%expr
              let __types_rec = [%e types_object] in
              let __native_kern = [%e native_kern] in
              let __simple_kern = [%e simple_kern] in
              [%e body_with_bindings]]
        | None ->
            [%expr
              let __types_rec = [%e types_object] in
              let __native_kern = [%e native_kern] in
              [%e body_with_bindings]])
    in
    let with_module =
      Ast_builder.Default.pexp_letmodule
        ~loc
        {txt = Some "__Types"; loc}
        types_impl
        inner_body
    in
    [%expr
      fun ~mode:(__mode : Sarek.Sarek_cpu_runtime.exec_mode)
          ~block
          ~grid
          __args -> [%e with_module]]
  else
    match simple_kern_opt with
    | Some simple_kern ->
        [%expr
          fun ~mode:(__mode : Sarek.Sarek_cpu_runtime.exec_mode)
              ~block
              ~grid
              __args ->
            let __native_kern = [%e native_kern] in
            let __simple_kern = [%e simple_kern] in
            [%e body_with_bindings]]
    | None ->
        [%expr
          fun ~mode:(__mode : Sarek.Sarek_cpu_runtime.exec_mode)
              ~block
              ~grid
              __args ->
            let __native_kern = [%e native_kern] in
            [%e body_with_bindings]]

(** Generate V2 cpu_kern - uses Spoc_core.Vector.get/set instead of Spoc.Mem.

    Generated signature: thread_state -> shared_mem -> args_tuple -> unit This
    matches the signature expected by run_parallel/run_sequential. *)
let gen_cpu_kern_native ~loc (kernel : tkernel) : expression =
  let use_fcm = has_inline_types kernel in
  let inline_type_names =
    if use_fcm then
      List.fold_left
        (fun acc decl ->
          let name =
            match decl with
            | TTypeRecord {tdecl_name; _} -> tdecl_name
            | TTypeVariant {tdecl_name; _} -> tdecl_name
          in
          if not (String.contains name '.') then StringSet.add name acc else acc)
        StringSet.empty
        kernel.tkern_type_decls
    else StringSet.empty
  in
  let current_module = Some (module_name_of_sarek_loc kernel.tkern_loc) in

  let body_e =
    if use_fcm then
      gen_expr_with_inline_types
        ~loc
        ~inline_type_names
        ~current_module
        kernel.tkern_body
    else
      let ctx = {empty_ctx with current_module} in
      gen_expr_impl ~loc ~ctx kernel.tkern_body
  in

  (* Generate inline module items *)
  let inline_items =
    let all_items = kernel.tkern_module_items in
    let skip_count = kernel.tkern_external_item_count in
    let rec drop n lst =
      if n <= 0 then lst
      else match lst with [] -> [] | _ :: tl -> drop (n - 1) tl
    in
    drop skip_count all_items
  in
  let body_with_items = wrap_module_items ~loc inline_items body_e in

  (* Build parameter tuple pattern with type constraints *)
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

  let state_pat = Ast_builder.Default.ppat_var ~loc {txt = state_var; loc} in
  let shared_pat = Ast_builder.Default.ppat_var ~loc {txt = shared_var; loc} in

  let inner_fun =
    if use_fcm then
      let types_pat =
        Ast_builder.Default.ppat_var ~loc {txt = types_module_var; loc}
      in
      [%expr
        fun [%p types_pat]
            ([%p state_pat] : Sarek.Sarek_cpu_runtime.thread_state)
            ([%p shared_pat] : Sarek.Sarek_cpu_runtime.shared_mem)
            [%p params_pat] -> [%e body_with_items]]
    else
      [%expr
        fun ([%p state_pat] : Sarek.Sarek_cpu_runtime.thread_state)
            ([%p shared_pat] : Sarek.Sarek_cpu_runtime.shared_mem)
            [%p params_pat] -> [%e body_with_items]]
  in
  (* Add warning suppression attribute *)
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

(** Generate simple kernel for optimized threadpool execution using the modern
    vector path (Spoc_core.Vector). *)
let gen_simple_cpu_kern_native ~loc ~exec_strategy (kernel : tkernel) :
    expression =
  let use_fcm = has_inline_types kernel in
  let inline_type_names =
    if use_fcm then
      List.fold_left
        (fun acc decl ->
          let name =
            match decl with
            | TTypeRecord {tdecl_name; _} -> tdecl_name
            | TTypeVariant {tdecl_name; _} -> tdecl_name
          in
          if not (String.contains name '.') then StringSet.add name acc else acc)
        StringSet.empty
        kernel.tkern_type_decls
    else StringSet.empty
  in
  let current_module = Some (module_name_of_sarek_loc kernel.tkern_loc) in
  let gen_mode = gen_mode_of_exec_strategy exec_strategy in

  let ctx =
    {empty_ctx with current_module; inline_types = inline_type_names; gen_mode}
  in
  let body_e = gen_expr_impl ~loc ~ctx kernel.tkern_body in

  (* Generate inline module items *)
  let inline_items =
    let all_items = kernel.tkern_module_items in
    let skip_count = kernel.tkern_external_item_count in
    let rec drop n lst =
      if n <= 0 then lst
      else match lst with [] -> [] | _ :: tl -> drop (n - 1) tl
    in
    drop skip_count all_items
  in
  let body_with_items = wrap_module_items ~loc inline_items body_e in

  (* Build parameter tuple pattern with type constraints *)
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

  let gid_x_pat = Ast_builder.Default.ppat_var ~loc {txt = gid_x_var; loc} in
  let gid_y_pat = Ast_builder.Default.ppat_var ~loc {txt = gid_y_var; loc} in
  let gid_z_pat = Ast_builder.Default.ppat_var ~loc {txt = gid_z_var; loc} in

  let inner_fun =
    if use_fcm then
      let types_pat =
        Ast_builder.Default.ppat_var ~loc {txt = types_module_var; loc}
      in
      match exec_strategy with
      | Sarek_convergence.Simple1D ->
          [%expr
            fun [%p types_pat] ([%p gid_x_pat] : int32) [%p params_pat] ->
              [%e body_with_items]]
      | Sarek_convergence.Simple2D ->
          [%expr
            fun [%p types_pat]
                ([%p gid_x_pat] : int32)
                ([%p gid_y_pat] : int32)
                [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.Simple3D ->
          [%expr
            fun [%p types_pat]
                ([%p gid_x_pat] : int32)
                ([%p gid_y_pat] : int32)
                ([%p gid_z_pat] : int32)
                [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.FullState ->
          failwith "gen_simple_cpu_kern_native called with FullState strategy"
    else
      match exec_strategy with
      | Sarek_convergence.Simple1D ->
          [%expr
            fun ([%p gid_x_pat] : int32) [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.Simple2D ->
          [%expr
            fun ([%p gid_x_pat] : int32)
                ([%p gid_y_pat] : int32)
                [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.Simple3D ->
          [%expr
            fun ([%p gid_x_pat] : int32)
                ([%p gid_y_pat] : int32)
                ([%p gid_z_pat] : int32)
                [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.FullState ->
          failwith "gen_simple_cpu_kern_native called with FullState strategy"
  in
  (* Add warning suppression attribute *)
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

(** Generate the cpu_kern wrapper for use with native_fn_t.

    Generated function type: parallel:bool -> block:int*int*int ->
    grid:int*int*int -> Obj.t array -> unit

    Uses Spoc_core.Vector instead of Spoc.Vector. Expects vectors directly in
    the Obj.t array (not expanded buffer/length pairs).

    This is a port of gen_cpu_kern_wrapper with updated callsites:
    - Uses gen_cpu_kern_native instead of gen_cpu_kern
    - Uses gen_simple_cpu_kern_native instead of gen_simple_cpu_kern
    - Uses parallel:bool instead of mode:exec_mode *)
let gen_cpu_kern_native_wrapper ~loc (kernel : tkernel) : expression =
  let use_fcm = has_inline_types kernel in
  let native_kern = gen_cpu_kern_native ~loc kernel in

  (* Detect execution strategy for optimization *)
  let exec_strategy = Sarek_convergence.kernel_exec_strategy kernel in

  (* Detect barrier usage at compile time - passed to runtime *)
  let has_barriers = Sarek_convergence.kernel_uses_barriers kernel in
  let has_barriers_expr = Ast_builder.Default.ebool ~loc has_barriers in

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

  (* Generate the simple kernel expression if needed *)
  let simple_kern_opt =
    match exec_strategy with
    | Sarek_convergence.Simple1D | Sarek_convergence.Simple2D
    | Sarek_convergence.Simple3D ->
        Some (gen_simple_cpu_kern_native ~loc ~exec_strategy kernel)
    | Sarek_convergence.FullState -> None
  in

  (* For simple kernels, generate optimized threadpool call *)
  let gen_simple_threadpool_call () =
    match exec_strategy with
    | Sarek_convergence.Simple1D ->
        if use_fcm then
          [%expr
            let bx, _, _ = block in
            let gx, _, _ = grid in
            Sarek.Sarek_cpu_runtime.run_1d_threadpool
              ~total_x:(bx * gx)
              (fun gid_x args -> __simple_kern __types_rec gid_x args)
              [%e args_tuple]]
        else
          [%expr
            let bx, _, _ = block in
            let gx, _, _ = grid in
            Sarek.Sarek_cpu_runtime.run_1d_threadpool
              ~total_x:(bx * gx)
              __simple_kern
              [%e args_tuple]]
    | Sarek_convergence.Simple2D ->
        if use_fcm then
          [%expr
            let bx, by, _ = block in
            let gx, gy, _ = grid in
            Sarek.Sarek_cpu_runtime.run_2d_threadpool
              ~width:(bx * gx)
              ~height:(by * gy)
              (fun gid_x gid_y args ->
                __simple_kern __types_rec gid_x gid_y args)
              [%e args_tuple]]
        else
          [%expr
            let bx, by, _ = block in
            let gx, gy, _ = grid in
            Sarek.Sarek_cpu_runtime.run_2d_threadpool
              ~width:(bx * gx)
              ~height:(by * gy)
              __simple_kern
              [%e args_tuple]]
    | Sarek_convergence.Simple3D ->
        if use_fcm then
          [%expr
            let bx, by, bz = block in
            let gx, gy, gz = grid in
            Sarek.Sarek_cpu_runtime.run_3d_threadpool
              ~width:(bx * gx)
              ~height:(by * gy)
              ~depth:(bz * gz)
              (fun gid_x gid_y gid_z args ->
                __simple_kern __types_rec gid_x gid_y gid_z args)
              [%e args_tuple]]
        else
          [%expr
            let bx, by, bz = block in
            let gx, gy, gz = grid in
            Sarek.Sarek_cpu_runtime.run_3d_threadpool
              ~width:(bx * gx)
              ~height:(by * gy)
              ~depth:(bz * gz)
              __simple_kern
              [%e args_tuple]]
    | Sarek_convergence.FullState ->
        failwith "gen_simple_threadpool_call called with FullState"
  in

  (* V2 uses parallel:bool - map to Sequential/Parallel/Threadpool:
     parallel=false -> Sequential
     parallel=true -> Threadpool for simple kernels, Parallel for complex *)
  let run_call =
    match exec_strategy with
    | Sarek_convergence.Simple1D | Sarek_convergence.Simple2D
    | Sarek_convergence.Simple3D ->
        (* Simple kernel - use optimized threadpool path when parallel *)
        let simple_call = gen_simple_threadpool_call () in
        if use_fcm then
          [%expr
            if __parallel then [%e simple_call]
            else
              Sarek.Sarek_cpu_runtime.run_sequential
                ~block
                ~grid
                (__native_kern __types_rec)
                [%e args_tuple]]
        else
          [%expr
            if __parallel then [%e simple_call]
            else
              Sarek.Sarek_cpu_runtime.run_sequential
                ~block
                ~grid
                __native_kern
                [%e args_tuple]]
    | Sarek_convergence.FullState ->
        (* Complex kernel - use threadpool with barrier support when parallel *)
        if use_fcm then
          [%expr
            if __parallel then
              Sarek.Sarek_cpu_runtime.run_threadpool
                ~has_barriers:[%e has_barriers_expr]
                ~block
                ~grid
                (__native_kern __types_rec)
                [%e args_tuple]
            else
              Sarek.Sarek_cpu_runtime.run_sequential
                ~block
                ~grid
                (__native_kern __types_rec)
                [%e args_tuple]]
        else
          [%expr
            if __parallel then
              Sarek.Sarek_cpu_runtime.run_threadpool
                ~has_barriers:[%e has_barriers_expr]
                ~block
                ~grid
                __native_kern
                [%e args_tuple]
            else
              Sarek.Sarek_cpu_runtime.run_sequential
                ~block
                ~grid
                __native_kern
                [%e args_tuple]]
  in

  (* Build the nested let bindings *)
  let body_with_bindings =
    List.fold_right
      (fun (pat, expr) body ->
        [%expr
          let [%p pat] = [%e expr] in
          [%e body]])
      arg_bindings
      run_call
  in

  if use_fcm then
    (* For FCM kernels, create the types object inside a local module *)
    let inline_decls = inline_type_decls kernel.tkern_type_decls in
    let types_object = gen_types_object ~loc inline_decls in
    let types_impl = gen_module_impl ~loc inline_decls in
    let inner_body =
      Ast_builder.Default.pexp_open
        ~loc
        (Ast_builder.Default.open_infos
           ~loc
           ~override:Fresh
           ~expr:
             (Ast_builder.Default.pmod_ident ~loc {txt = Lident "__Types"; loc}))
        (match simple_kern_opt with
        | Some simple_kern ->
            [%expr
              let __types_rec = [%e types_object] in
              let __native_kern = [%e native_kern] in
              let __simple_kern = [%e simple_kern] in
              [%e body_with_bindings]]
        | None ->
            [%expr
              let __types_rec = [%e types_object] in
              let __native_kern = [%e native_kern] in
              [%e body_with_bindings]])
    in
    let with_module =
      Ast_builder.Default.pexp_letmodule
        ~loc
        {txt = Some "__Types"; loc}
        types_impl
        inner_body
    in
    [%expr
      fun ~parallel:(__parallel : bool) ~block ~grid __args -> [%e with_module]]
  else
    match simple_kern_opt with
    | Some simple_kern ->
        [%expr
          fun ~parallel:(__parallel : bool) ~block ~grid __args ->
            let __native_kern = [%e native_kern] in
            let __simple_kern = [%e simple_kern] in
            [%e body_with_bindings]]
    | None ->
        [%expr
          fun ~parallel:(__parallel : bool) ~block ~grid __args ->
            let __native_kern = [%e native_kern] in
            [%e body_with_bindings]]
