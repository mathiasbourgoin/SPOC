(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This is the PPX entry point. It registers the [%kernel ...] extension
 * and orchestrates parsing, type checking, lowering, and code generation.
 *
 * Type and intrinsic information comes from:
 * 1. Sarek_env.with_stdlib - hardcoded core types and intrinsics
 * 2. [@@sarek.type] attributes - user-defined record/variant types
 * 3. Runtime Sarek_registry - populated by %sarek_intrinsic at load time
 ******************************************************************************)

open Ppxlib
open Sarek_ppx_lib

(* Registry of types declared in the current compilation unit via [@@sarek.type] *)
let registered_types : Sarek_ast.type_decl list ref = ref []

(* Registry of module items declared in the current compilation unit *)
let registered_mods : Sarek_ast.module_item list ref = ref []

let tdecl_key = function
  | Sarek_ast.Type_record {tdecl_name; tdecl_module; _}
  | Sarek_ast.Type_variant {tdecl_name; tdecl_module; _} -> (
      match tdecl_module with
      | Some m -> m ^ "." ^ tdecl_name
      | None -> tdecl_name)

let dedup_tdecls decls =
  let module S = Set.Make (String) in
  let _, revs =
    List.fold_left
      (fun (seen, acc) d ->
        let key = tdecl_key d in
        if S.mem key seen then (seen, acc) else (S.add key seen, d :: acc))
      (S.empty, [])
      decls
  in
  List.rev revs

let dedup_mods mods =
  let module S = Set.Make (String) in
  let _, revs =
    List.fold_left
      (fun (seen, acc) m ->
        let key =
          match m with
          | Sarek_ast.MConst (n, _, _) | Sarek_ast.MFun (n, _, _) -> n
        in
        if S.mem key seen then (seen, acc) else (S.add key seen, m :: acc))
      (S.empty, [])
      mods
  in
  List.rev revs

let rec flatten_longident = function
  | Lident s -> [s]
  | Ldot (li, s) -> flatten_longident li @ [s]
  | Lapply _ -> []

let rec core_type_to_sarek_type_expr ~loc (ct : core_type) =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt; _}, args) ->
      let name = String.concat "." (flatten_longident txt) in
      let args = List.map (core_type_to_sarek_type_expr ~loc) args in
      Sarek_ast.TEConstr (name, args)
  | Ptyp_var v -> Sarek_ast.TEVar v
  | Ptyp_arrow (_, a, b) ->
      Sarek_ast.TEArrow
        ( core_type_to_sarek_type_expr ~loc a,
          core_type_to_sarek_type_expr ~loc b )
  | Ptyp_tuple l ->
      Sarek_ast.TETuple (List.map (core_type_to_sarek_type_expr ~loc) l)
  | _ ->
      Location.raise_errorf ~loc "Unsupported type expression in [@@sarek.type]"

let module_name_of_loc loc =
  let file = loc.loc_start.pos_fname in
  let base = Filename.(remove_extension (basename file)) in
  String.capitalize_ascii base

let register_sarek_type_decl ~loc (td : type_declaration) =
  let tdecl =
    match td.ptype_kind with
    | Ptype_record labels ->
        let fields =
          List.map
            (fun ld ->
              ( ld.pld_name.txt,
                ld.pld_mutable = Mutable,
                core_type_to_sarek_type_expr ~loc ld.pld_type ))
            labels
        in
        Sarek_ast.Type_record
          {
            tdecl_name = td.ptype_name.txt;
            tdecl_module = Some (module_name_of_loc loc);
            tdecl_fields = fields;
            tdecl_loc = Sarek_ast.loc_of_ppxlib loc;
          }
    | Ptype_variant constrs ->
        let cons =
          List.map
            (fun cd ->
              match cd.pcd_args with
              | Pcstr_tuple [] -> (cd.pcd_name.txt, None)
              | Pcstr_tuple [ct] ->
                  (cd.pcd_name.txt, Some (core_type_to_sarek_type_expr ~loc ct))
              | Pcstr_tuple _ | Pcstr_record _ ->
                  Location.raise_errorf
                    ~loc
                    "Only zero or single-argument constructors supported in \
                     [@@sarek.type]")
            constrs
        in
        Sarek_ast.Type_variant
          {
            tdecl_name = td.ptype_name.txt;
            tdecl_module = Some (module_name_of_loc loc);
            tdecl_constructors = cons;
            tdecl_loc = Sarek_ast.loc_of_ppxlib loc;
          }
    | _ ->
        Location.raise_errorf
          ~loc
          "Only record or variant types can be used with [@@sarek.type]"
  in
  registered_types := tdecl :: !registered_types ;
  ()

let register_sarek_module_item ~loc:_ item =
  (match item with
  | Sarek_ast.MFun (name, _, _) ->
      Format.eprintf "Sarek PPX: register module fun %s@." name
  | Sarek_ast.MConst (name, _, _) ->
      Format.eprintf "Sarek PPX: register module const %s@." name) ;
  registered_mods := item :: !registered_mods

let scan_dir_for_sarek_types directory =
  Array.iter
    (fun fname ->
      if Filename.check_suffix fname ".ml" then
        let path = Filename.concat directory fname in
        try
          let ic = open_in path in
          let lexbuf = Lexing.from_channel ic in
          lexbuf.lex_curr_p <-
            {lexbuf.lex_curr_p with pos_fname = path; pos_bol = 0; pos_lnum = 1} ;
          let st = Parse.implementation lexbuf in
          close_in ic ;
          List.iter
            (fun item ->
              match item.pstr_desc with
              | Pstr_type (_rf, decls) ->
                  List.iter
                    (fun d ->
                      let has_attr name a = String.equal a.attr_name.txt name in
                      if
                        List.exists (has_attr "sarek.type") d.ptype_attributes
                        || List.exists
                             (has_attr "sarek.type_private")
                             d.ptype_attributes
                      then register_sarek_type_decl ~loc:d.ptype_loc d)
                    decls
              | Pstr_value (Nonrecursive, vbs) ->
                  List.iter
                    (fun vb ->
                      let has_attr name a = String.equal a.attr_name.txt name in
                      if
                        List.exists (has_attr "sarek.module") vb.pvb_attributes
                        || List.exists
                             (has_attr "sarek.module_private")
                             vb.pvb_attributes
                      then (
                        Format.eprintf
                          "Sarek PPX: sarek.module binding %s@."
                          (Option.value
                             (Sarek_parse.extract_name_from_pattern vb.pvb_pat)
                             ~default:"<anon>") ;
                        let name =
                          match
                            Sarek_parse.extract_name_from_pattern vb.pvb_pat
                          with
                          | Some n -> n
                          | None ->
                              Location.raise_errorf
                                ~loc:vb.pvb_pat.ppat_loc
                                "Expected variable name"
                        in
                        let ty =
                          Sarek_parse.extract_type_from_pattern vb.pvb_pat
                        in
                        let item =
                          match vb.pvb_expr.pexp_desc with
                          | Pexp_function (params, _, Pfunction_body body_expr)
                            ->
                              let params =
                                List.map
                                  Sarek_parse.extract_param_from_pparam
                                  params
                              in
                              let body =
                                Sarek_parse.parse_expression body_expr
                              in
                              Sarek_ast.MFun (name, params, body)
                          | _ ->
                              let value =
                                Sarek_parse.parse_expression vb.pvb_expr
                              in
                              let ty =
                                match ty with
                                | Some t -> t
                                | None ->
                                    Location.raise_errorf
                                      ~loc:vb.pvb_pat.ppat_loc
                                      "[@sarek.module] constants require a \
                                       type annotation"
                              in
                              Sarek_ast.MConst (name, ty, value)
                        in
                        register_sarek_module_item ~loc:vb.pvb_loc item))
                    vbs
              | _ -> ())
            st
        with _ -> ())
    (try Sys.readdir directory with Sys_error _ -> [||])

(* Attribute used to mark Sarek-visible type declarations *)
let sarek_type_attr =
  Attribute.declare
    "sarek.type"
    Attribute.Context.type_declaration
    Ast_pattern.(pstr nil)
    ()

let sarek_type_private_attr =
  Attribute.declare
    "sarek.type_private"
    Attribute.Context.type_declaration
    Ast_pattern.(pstr nil)
    ()

(** Generate field accessor functions for a record type.
    For type point = { x: float32; y: float32 }, generates:
      let sarek_get_point_x (p : point) : float32 = p.x
      let sarek_get_point_y (p : point) : float32 = p.y *)
let generate_field_accessors ~loc (td : type_declaration) : structure_item list
    =
  match td.ptype_kind with
  | Ptype_record labels ->
      let type_name = td.ptype_name.txt in
      let type_lid = {txt = Lident type_name; loc} in
      let type_ct = Ast_builder.Default.ptyp_constr ~loc type_lid [] in
      List.map
        (fun (ld : label_declaration) ->
          let field_name = ld.pld_name.txt in
          let field_type = ld.pld_type in
          let fn_name = Printf.sprintf "sarek_get_%s_%s" type_name field_name in
          let param_pat =
            Ast_builder.Default.ppat_constraint
              ~loc
              (Ast_builder.Default.pvar ~loc "p")
              type_ct
          in
          let field_access =
            Ast_builder.Default.pexp_field
              ~loc
              (Ast_builder.Default.evar ~loc "p")
              {txt = Lident field_name; loc}
          in
          let body =
            Ast_builder.Default.pexp_constraint ~loc field_access field_type
          in
          let fn_expr =
            Ast_builder.Default.pexp_fun ~loc Nolabel None param_pat body
          in
          let fn_pat = Ast_builder.Default.pvar ~loc fn_name in
          [%stri let [%p fn_pat] = [%e fn_expr]])
        labels
  | _ -> []

(** Extract type name from a core_type for registry registration *)
let type_name_of_core_type (ct : core_type) : string =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident name; _}, _) -> name
  | Ptyp_constr ({txt = Ldot (_, name); _}, _) -> name
  | _ -> "unknown"

(** Calculate the size in bytes of a sarek type based on its fields. Uses 4
    bytes for int32/float32, 8 bytes for int64/float64. *)
let calc_type_size (labels : label_declaration list) : int =
  List.fold_left
    (fun acc ld ->
      let field_size =
        match ld.pld_type.ptyp_desc with
        | Ptyp_constr ({txt = Lident "int32"; _}, _) -> 4
        | Ptyp_constr ({txt = Lident "int64"; _}, _) -> 8
        | Ptyp_constr ({txt = Lident "float32"; _}, _) -> 4
        | Ptyp_constr ({txt = Lident "float"; _}, _) -> 8
        | Ptyp_constr ({txt = Lident "int"; _}, _) -> 4
        | _ -> 4
      in
      acc + field_size)
    0
    labels

(** Get the accessor function for a field type *)
let get_accessor_for_type ~loc (ct : core_type) : expression =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident "float32"; _}, _)
  | Ptyp_constr ({txt = Lident "float"; _}, _) ->
      [%expr Spoc.Tools.float32get]
  | Ptyp_constr ({txt = Lident "int32"; _}, _)
  | Ptyp_constr ({txt = Lident "int"; _}, _) ->
      (* For int32, we'd need custom_int32get - for now use float32 as placeholder *)
      [%expr fun arr idx -> Int32.of_float (Spoc.Tools.float32get arr idx)]
  | _ -> [%expr Spoc.Tools.float32get]

(** Get the setter function for a field type *)
let set_accessor_for_type ~loc (ct : core_type) : expression =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident "float32"; _}, _)
  | Ptyp_constr ({txt = Lident "float"; _}, _) ->
      [%expr Spoc.Tools.float32set]
  | Ptyp_constr ({txt = Lident "int32"; _}, _)
  | Ptyp_constr ({txt = Lident "int"; _}, _) ->
      [%expr fun arr idx v -> Spoc.Tools.float32set arr idx (Int32.to_float v)]
  | _ -> [%expr Spoc.Tools.float32set]

(** Get the field count (number of primitive fields, counting nested as 1 for
    now) *)
let field_element_count (ct : core_type) : int =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident "float32"; _}, _) -> 1
  | Ptyp_constr ({txt = Lident "float"; _}, _) -> 1
  | Ptyp_constr ({txt = Lident "int32"; _}, _) -> 1
  | Ptyp_constr ({txt = Lident "int"; _}, _) -> 1
  | _ -> 1

(** Generate a <name>_custom value for Vector.Custom.
    For type float4 = { mutable x: float32; mutable y: float32; ... } [@@sarek.type],
    generates proper get/set functions using Spoc.Tools.float32get/set *)
let generate_custom_value ~loc (td : type_declaration) : structure_item list =
  match td.ptype_kind with
  | Ptype_record labels ->
      let type_name = td.ptype_name.txt in
      let custom_name = type_name ^ "_custom" in
      let size = calc_type_size labels in
      let size_expr = Ast_builder.Default.eint ~loc size in
      let custom_pat = Ast_builder.Default.pvar ~loc custom_name in

      (* Calculate field offsets in terms of primitive count *)
      let field_infos =
        let rec calc_offsets offset = function
          | [] -> []
          | ld :: rest ->
              let count = field_element_count ld.pld_type in
              (ld.pld_name.txt, offset, ld.pld_type)
              :: calc_offsets (offset + count) rest
        in
        calc_offsets 0 labels
      in

      (* Generate get function: fun arr idx -> { x = get arr (idx*n + 0); ... } *)
      let num_fields = List.length labels in
      let num_fields_expr = Ast_builder.Default.eint ~loc num_fields in
      let get_fields =
        List.map
          (fun (name, offset, ftype) ->
            let offset_expr = Ast_builder.Default.eint ~loc offset in
            let getter = get_accessor_for_type ~loc ftype in
            let field_expr =
              [%expr [%e getter] arr (base + [%e offset_expr])]
            in
            ({txt = Lident name; loc}, field_expr))
          field_infos
      in
      let get_record = Ast_builder.Default.pexp_record ~loc get_fields None in
      let get_body =
        [%expr
          let base = idx * [%e num_fields_expr] in
          [%e get_record]]
      in
      let get_fn = [%expr fun arr idx -> [%e get_body]] in

      (* Generate set function: fun arr idx v -> set arr (idx*n+0) v.x; ... *)
      let set_stmts =
        List.map
          (fun (name, offset, ftype) ->
            let offset_expr = Ast_builder.Default.eint ~loc offset in
            let setter = set_accessor_for_type ~loc ftype in
            let field_access =
              Ast_builder.Default.pexp_field
                ~loc
                [%expr v]
                {txt = Lident name; loc}
            in
            [%expr [%e setter] arr (base + [%e offset_expr]) [%e field_access]])
          field_infos
      in
      let set_body =
        match set_stmts with
        | [] -> [%expr ()]
        | [stmt] ->
            [%expr
              let base = idx * [%e num_fields_expr] in
              [%e stmt]]
        | stmt :: rest ->
            let seq =
              List.fold_left
                (fun acc s ->
                  [%expr
                    [%e acc] ;
                    [%e s]])
                stmt
                rest
            in
            [%expr
              let base = idx * [%e num_fields_expr] in
              [%e seq]]
      in
      let set_fn = [%expr fun arr idx v -> [%e set_body]] in

      [
        [%stri
          let [%p custom_pat] =
            {
              Spoc.Vector.size = [%e size_expr];
              get = [%e get_fn];
              set = [%e set_fn];
            }];
      ]
  | _ -> []

(** Generate runtime registration code for a type.

    This implements the ppx_deriving-style composability pattern:
    - The PPX generates OCaml code that calls Sarek.Sarek_registry.register_*
    - This code runs at module initialization time (when the library is linked)
    - When another module depends on this library, the type is already registered
    - At JIT time, the code generator can look up the type info

    For a record type:
      type point = { x: float32; y: float32 } [@@sarek.type]

    Generates:
      let () = Sarek.Sarek_registry.register_record
        "Module.point"
        ~fields:[{field_name="x"; field_type="float32"; field_mutable=false}; ...]
        ~size:8

    For a variant type:
      type shape = Circle of float32 | Square of float32 [@@sarek.type]

    Generates:
      let () = Sarek.Sarek_registry.register_variant
        "Module.shape"
        ~constructors:[{ctor_name="Circle"; ctor_arg_type=Some "float32"}; ...]
*)
let generate_type_registration ~loc (td : type_declaration) :
    structure_item list =
  let type_name = td.ptype_name.txt in
  (* Get module name from file path for fully qualified name *)
  let module_name = module_name_of_loc loc in
  let full_name = module_name ^ "." ^ type_name in
  let full_name_expr = Ast_builder.Default.estring ~loc full_name in

  match td.ptype_kind with
  | Ptype_record labels ->
      (* Build field info list *)
      let field_exprs =
        List.map
          (fun (ld : label_declaration) ->
            let fname = Ast_builder.Default.estring ~loc ld.pld_name.txt in
            let ftype =
              Ast_builder.Default.estring
                ~loc
                (type_name_of_core_type ld.pld_type)
            in
            let fmut =
              Ast_builder.Default.ebool ~loc (ld.pld_mutable = Mutable)
            in
            [%expr
              {
                Sarek.Sarek_registry.field_name = [%e fname];
                field_type = [%e ftype];
                field_mutable = [%e fmut];
              }])
          labels
      in
      let fields_list = Ast_builder.Default.elist ~loc field_exprs in
      let size = calc_type_size labels in
      let size_expr = Ast_builder.Default.eint ~loc size in
      [
        [%stri
          let () =
            Sarek.Sarek_registry.register_record
              [%e full_name_expr]
              ~fields:[%e fields_list]
              ~size:[%e size_expr]];
      ]
  | Ptype_variant constrs ->
      (* Build constructor info list *)
      let ctor_exprs =
        List.map
          (fun (cd : constructor_declaration) ->
            let cname = Ast_builder.Default.estring ~loc cd.pcd_name.txt in
            let carg =
              match cd.pcd_args with
              | Pcstr_tuple [] -> [%expr None]
              | Pcstr_tuple [t] ->
                  let tname =
                    Ast_builder.Default.estring ~loc (type_name_of_core_type t)
                  in
                  [%expr Some [%e tname]]
              | Pcstr_tuple _ -> [%expr Some "tuple"] (* Simplified for now *)
              | Pcstr_record _ -> [%expr Some "record"]
              (* Inline record *)
            in
            [%expr
              {
                Sarek.Sarek_registry.ctor_name = [%e cname];
                ctor_arg_type = [%e carg];
              }])
          constrs
      in
      let ctors_list = Ast_builder.Default.elist ~loc ctor_exprs in
      [
        [%stri
          let () =
            Sarek.Sarek_registry.register_variant
              [%e full_name_expr]
              ~constructors:[%e ctors_list]];
      ]
  | _ -> []

(* Context-free rule to capture [@@sarek.type] before kernel expansion *)
let sarek_type_rule =
  Context_free.Rule.attr_str_type_decl
    sarek_type_attr
    (fun ~ctxt _rec_flag decls payloads ->
      let loc = Expansion_context.Deriver.derived_item_loc ctxt in
      let generated =
        List.concat_map
          (fun (td, payload) ->
            match payload with
            | Some () ->
                register_sarek_type_decl ~loc:td.ptype_loc td ;
                (* Generate field accessor functions for compile-time validation *)
                let accessors = generate_field_accessors ~loc td in
                (* Generate <name>_custom value for Vector.Custom *)
                let custom_val = generate_custom_value ~loc td in
                (* Generate runtime registration code for cross-module composability.
                   This follows the ppx_deriving pattern: the PPX generates OCaml code
                   that registers the type at module initialization time. When another
                   module depends on this library, the registration runs before any
                   kernels are JIT-compiled, making the type info available. *)
                let registration = generate_type_registration ~loc td in
                accessors @ custom_val @ registration
            | None -> [])
          (List.combine decls payloads)
      in
      generated)

let sarek_type_private_rule =
  Context_free.Rule.attr_str_type_decl
    sarek_type_private_attr
    (fun ~ctxt:_ _rec_flag decls payloads ->
      List.iter2
        (fun td payload ->
          match payload with
          | Some () -> register_sarek_type_decl ~loc:td.ptype_loc td
          | None -> ())
        decls
        payloads ;
      [])

(******************************************************************************
 * Intrinsic Type and Function Declarations
 *
 * These extensions allow libraries to define primitive types and functions
 * that map directly to GPU intrinsics:
 *
 *   type%sarek_intrinsic float32 = {
 *     device = (fun _ -> "float");
 *     ctype = Ctypes.float;
 *   }
 *
 *   let%sarek_intrinsic rsqrt : float32 -> float32 = {
 *     device = (fun dev -> match dev... with ...);
 *     ocaml = (fun x -> 1.0 /. sqrt x);
 *   }
 ******************************************************************************)

(** Extension for type%sarek_intrinsic *)
let expand_sarek_intrinsic_type ~ctxt payload =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in
  match payload with
  | PStr
      [
        {
          pstr_desc =
            Pstr_value
              ( _,
                [
                  {
                    pvb_pat = {ppat_desc = Ppat_var {txt = type_name; _}; _};
                    pvb_expr = record_expr;
                    _;
                  };
                ] );
          _;
        };
      ] ->
      (* Extract device and ctype fields from the record expression *)
      let device_expr, ctype_expr =
        match record_expr.pexp_desc with
        | Pexp_record (fields, None) -> (
            let find_field name =
              List.find_map
                (fun (lid, expr) ->
                  match lid.txt with
                  | Lident n when String.equal n name -> Some expr
                  | _ -> None)
                fields
            in
            ( (match find_field "device" with
              | Some e -> e
              | None ->
                  Location.raise_errorf
                    ~loc
                    "sarek_intrinsic type requires 'device' field"),
              match find_field "ctype" with
              | Some e -> e
              | None ->
                  Location.raise_errorf
                    ~loc
                    "sarek_intrinsic type requires 'ctype' field" ))
        | _ ->
            Location.raise_errorf
              ~loc
              "sarek_intrinsic type expects a record { device = ...; ctype = \
               ... }"
      in
      (* Generate runtime registration code *)
      let type_name_str = Ast_builder.Default.estring ~loc type_name in
      [
        [%stri
          let () =
            Sarek.Sarek_registry.register_type
              [%e type_name_str]
              ~device:[%e device_expr]
              ~size:(Ctypes.sizeof [%e ctype_expr])];
      ]
  | _ ->
      Location.raise_errorf
        ~loc
        "type%%sarek_intrinsic expects: type%%sarek_intrinsic name = { device \
         = ...; ctype = ... }"

(** Extract argument types and return type from a function type *)
let rec extract_fun_types (ct : core_type) : string list * string =
  match ct.ptyp_desc with
  | Ptyp_arrow (_, arg_type, ret_type) ->
      let more_args, final_ret = extract_fun_types ret_type in
      (type_name_of_core_type arg_type :: more_args, final_ret)
  | _ -> ([], type_name_of_core_type ct)

(** Extension for let%sarek_intrinsic *)
let expand_sarek_intrinsic_fun ~ctxt payload =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in
  match payload with
  | PStr
      [
        {
          pstr_desc =
            Pstr_value
              ( _,
                [
                  {
                    pvb_pat =
                      {
                        ppat_desc =
                          Ppat_constraint
                            ( {ppat_desc = Ppat_var {txt = fun_name; _}; _},
                              type_constraint );
                        _;
                      };
                    pvb_expr = record_expr;
                    _;
                  };
                ] );
          _;
        };
      ] ->
      (* Extract type information from constraint *)
      let arg_types, ret_type = extract_fun_types type_constraint in
      let arity = List.length arg_types in
      (* Extract device and ocaml fields from the record expression *)
      let device_expr, ocaml_expr =
        match record_expr.pexp_desc with
        | Pexp_record (fields, None) -> (
            let find_field name =
              List.find_map
                (fun (lid, expr) ->
                  match lid.txt with
                  | Lident n when String.equal n name -> Some expr
                  | _ -> None)
                fields
            in
            ( (match find_field "device" with
              | Some e -> e
              | None ->
                  Location.raise_errorf
                    ~loc
                    "sarek_intrinsic function requires 'device' field"),
              match find_field "ocaml" with
              | Some e -> e
              | None ->
                  Location.raise_errorf
                    ~loc
                    "sarek_intrinsic function requires 'ocaml' field" ))
        | _ ->
            Location.raise_errorf
              ~loc
              "sarek_intrinsic function expects a record { device = ...; ocaml \
               = ... }"
      in
      (* Generate device function, registration, and OCaml binding *)
      let fun_name_str = Ast_builder.Default.estring ~loc fun_name in
      let fun_name_pat = Ast_builder.Default.pvar ~loc fun_name in
      let device_fun_name = fun_name ^ "_device" in
      let device_fun_pat = Ast_builder.Default.pvar ~loc device_fun_name in
      let arity_expr = Ast_builder.Default.eint ~loc arity in
      let arg_types_expr =
        Ast_builder.Default.elist
          ~loc
          (List.map (Ast_builder.Default.estring ~loc) arg_types)
      in
      let ret_type_expr = Ast_builder.Default.estring ~loc ret_type in
      let device_fun_ref_name = fun_name ^ "_device_ref" in
      let device_fun_ref_pat =
        Ast_builder.Default.pvar ~loc device_fun_ref_name
      in
      let device_fun_ref_expr =
        Ast_builder.Default.evar ~loc device_fun_ref_name
      in
      [
        (* Expose the device function for extensions to chain to *)
        [%stri
          let [%p device_fun_pat] : Spoc.Devices.device -> string =
            [%e device_expr]];
        (* Mutable ref for extension chaining *)
        [%stri
          let [%p device_fun_ref_pat] : (Spoc.Devices.device -> string) ref =
            ref [%e Ast_builder.Default.evar ~loc device_fun_name]];
        (* Register the intrinsic for code generation - uses the ref for extensibility *)
        [%stri
          let () =
            Sarek.Sarek_registry.register_fun
              [%e fun_name_str]
              ~arity:[%e arity_expr]
              ~device:(fun dev -> ![%e device_fun_ref_expr] dev)
              ~arg_types:[%e arg_types_expr]
              ~ret_type:[%e ret_type_expr]];
        (* Expose the OCaml implementation for host-side use *)
        [%stri let [%p fun_name_pat] = [%e ocaml_expr]];
      ]
  | _ ->
      Location.raise_errorf
        ~loc
        "let%%sarek_intrinsic expects: let%%sarek_intrinsic (name : type) = { \
         device = ...; ocaml = ... }"

(** Combined extension for %sarek_intrinsic - handles both types and functions
*)
let expand_sarek_intrinsic ~ctxt payload =
  let _loc = Expansion_context.Extension.extension_point_loc ctxt in
  match payload with
  (* Try type first - pattern: let name = { device = ...; ctype = ... } *)
  | PStr
      [
        {
          pstr_desc =
            Pstr_value
              ( _,
                [
                  {
                    pvb_pat = {ppat_desc = Ppat_var _; _};
                    pvb_expr = {pexp_desc = Pexp_record (fields, None); _};
                    _;
                  };
                ] );
          _;
        };
      ]
    when List.exists
           (fun (lid, _) ->
             match lid.txt with Lident "ctype" -> true | _ -> false)
           fields ->
      expand_sarek_intrinsic_type ~ctxt payload
  (* Otherwise try function - pattern: let (name : type) = { device = ...; ocaml = ... } *)
  | _ -> expand_sarek_intrinsic_fun ~ctxt payload

let sarek_intrinsic_extension =
  Extension.V3.declare
    "sarek_intrinsic"
    Extension.Context.structure_item
    Ast_pattern.(pstr __)
    (fun ~ctxt payload ->
      let loc = Expansion_context.Extension.extension_point_loc ctxt in
      let items = expand_sarek_intrinsic ~ctxt (PStr payload) in
      (* Wrap multiple items in include struct ... end *)
      Ast_builder.Default.pstr_include
        ~loc
        {
          pincl_mod = Ast_builder.Default.pmod_structure ~loc items;
          pincl_loc = loc;
          pincl_attributes = [];
        })

(** Extension for %sarek_extend - allows extending intrinsics for new backends.

    Syntax: let%sarek_extend Module.func = fun dev -> if ... then "new_code"
    else Module.func_device dev

    This updates the Module.func_device_ref to point to the new function, which
    chains to the original via Module.func_device. *)
let expand_sarek_extend ~ctxt payload =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in
  match payload with
  | PStr
      [
        {
          pstr_desc =
            Pstr_value
              ( _,
                [
                  {
                    pvb_pat = {ppat_desc = Ppat_var {txt = name; _}; _};
                    pvb_expr = new_device_fun;
                    _;
                  };
                ] );
          _;
        };
      ] ->
      (* name is like "Float32.sin" - we need to update Float32.sin_device_ref *)
      let device_ref_name = name ^ "_device_ref" in
      let device_ref_expr =
        (* Parse "Float32.sin_device_ref" into a proper longident expression *)
        let parts = String.split_on_char '.' device_ref_name in
        match parts with
        | [single] -> Ast_builder.Default.evar ~loc single
        | _ ->
            let rec build_longident = function
              | [] -> assert false
              | [x] -> Lident x
              | x :: rest -> Ldot (build_longident rest, x)
            in
            Ast_builder.Default.pexp_ident
              ~loc
              {txt = build_longident (List.rev parts); loc}
      in
      [%stri let () = [%e device_ref_expr] := [%e new_device_fun]]
  | _ ->
      Location.raise_errorf
        ~loc
        "let%%sarek_extend expects: let%%sarek_extend Module.func = fun dev -> \
         ..."

let sarek_extend_extension =
  Extension.V3.declare
    "sarek_extend"
    Extension.Context.structure_item
    Ast_pattern.(pstr __)
    (fun ~ctxt payload -> expand_sarek_extend ~ctxt (PStr payload))

(** The main kernel expansion function *)
let expand_kernel ~ctxt payload =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in

  try
    let dir = Filename.dirname loc.loc_start.pos_fname in
    scan_dir_for_sarek_types dir ;
    let real_dir = Filename.dirname (Unix.realpath loc.loc_start.pos_fname) in
    if not (String.equal dir real_dir) then scan_dir_for_sarek_types real_dir ;
    (match Sys.getenv_opt "PWD" with
    | Some cwd -> (
        let source_dir = Filename.concat cwd dir in
        try
          if Sys.is_directory source_dir then
            scan_dir_for_sarek_types source_dir
        with Sys_error _ -> ())
    | None -> ()) ;
    (* Types and module items registered in the current compilation unit *)
    let pre_types = dedup_tdecls !registered_types in
    let pre_mods = dedup_mods !registered_mods in
    (* 1. Parse the PPX payload to Sarek AST *)
    let ast = Sarek_parse.parse_payload payload in
    let ast =
      {
        ast with
        Sarek_ast.kern_types = pre_types @ ast.kern_types;
        kern_module_items = pre_mods @ ast.kern_module_items;
      }
    in

    (* 2. Set up the typing environment with stdlib *)
    let env = Sarek_env.(empty |> with_stdlib) in

    (* 3. Type inference *)
    match Sarek_typer.infer_kernel env ast with
    | Error errors ->
        (* Report the first error with location *)
        Sarek_error.report_errors errors ;
        (* If we get here, generate dummy expression *)
        [%expr assert false]
    | Ok tkernel ->
        (* 4. Lower to Kirc_Ast *)
        let ir, constructors = Sarek_lower.lower_kernel tkernel in
        let ret_val = Sarek_lower.lower_return_value tkernel in

        (* 5. Quote the IR back to OCaml *)
        Sarek_quote.quote_kernel ~loc tkernel ir constructors ret_val
  with
  | Sarek_parse.Parse_error_exn (msg, ploc) ->
      Location.raise_errorf ~loc:ploc "Sarek parse error: %s" msg
  | Sarek_error.Sarek_error e ->
      let eloc = Sarek_ast.loc_to_ppxlib (Sarek_error.error_loc e) in
      Location.raise_errorf ~loc:eloc "%a" Sarek_error.pp_error e
  | e ->
      Location.raise_errorf
        ~loc
        "Sarek internal error: %s"
        (Printexc.to_string e)

(** The [%kernel ...] extension for expressions *)
let kernel_extension =
  Extension.V3.declare
    "kernel"
    Extension.Context.expression
    Ast_pattern.(single_expr_payload __)
    expand_kernel

(* Register top-level Sarek type declarations *)
let expand_sarek_type ~ctxt payload =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in
  match payload with
  | PStr ([{pstr_desc = Pstr_type (_rf, [tdecl]); _}] as items) ->
      let ctors =
        Sarek_lower.constructor_strings_of_core_type_decl ~loc tdecl
      in
      let register =
        [%stri
          let () = List.iter Sarek.Kirc.register_constructor_string [%e ctors]]
      in
      items @ [register]
  | _ ->
      Location.raise_errorf
        ~loc
        "%%sarek.type expects a single type declaration"

let sarek_type_extension = ()

(** Register sarek.module bindings on any structure we process, so libraries can
    publish module items for use in kernels. *)
let process_structure_for_module_items (str : structure) : structure =
  let process_vb vb =
    let has_attr name a = String.equal a.attr_name.txt name in
    if
      List.exists (has_attr "sarek.module") vb.pvb_attributes
      || List.exists (has_attr "sarek.module_private") vb.pvb_attributes
    then (
      let loc = vb.pvb_loc in
      let name =
        match Sarek_parse.extract_name_from_pattern vb.pvb_pat with
        | Some n -> n
        | None ->
            Location.raise_errorf
              ~loc:vb.pvb_pat.ppat_loc
              "Expected variable name"
      in
      let ty = Sarek_parse.extract_type_from_pattern vb.pvb_pat in
      let item =
        match vb.pvb_expr.pexp_desc with
        | Pexp_function (params, _, Pfunction_body body_expr) ->
            let params =
              List.map Sarek_parse.extract_param_from_pparam params
            in
            let body = Sarek_parse.parse_expression body_expr in
            Sarek_ast.MFun (name, params, body)
        | _ ->
            let value = Sarek_parse.parse_expression vb.pvb_expr in
            let ty =
              match ty with
              | Some t -> t
              | None ->
                  Location.raise_errorf
                    ~loc:vb.pvb_pat.ppat_loc
                    "[@sarek.module] constants require a type annotation"
            in
            Sarek_ast.MConst (name, ty, value)
      in
      register_sarek_module_item ~loc item ;
      None)
    else None
  in
  let extra_items =
    List.filter_map
      (fun item ->
        match item.pstr_desc with
        | Pstr_value (_rf, vbs) ->
            let registrations = List.filter_map process_vb vbs in
            if registrations = [] then None else Some registrations
        | _ -> None)
      str
    |> List.flatten
  in
  str @ extra_items

(** Register the transformation *)
let () =
  let rules =
    [
      sarek_type_rule;
      sarek_type_private_rule;
      Context_free.Rule.extension kernel_extension;
      Context_free.Rule.extension sarek_intrinsic_extension;
    ]
  in
  Driver.register_transformation
    ~rules
    ~impl:process_structure_for_module_items
    "sarek_ppx"
