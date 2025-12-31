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

(* Force stdlib module initialization to populate the PPX registry.
   This must happen before any kernel expansion uses Sarek_env.with_stdlib(). *)
let () = Sarek_stdlib.force_init ()

(* Force Float64 module initialization to populate the PPX registry.
   Float64 is in a separate library (sarek.float64) for devices that support it.
   The PPX knows about Float64 intrinsics, but runtime code only links if needed. *)
let () = ignore Sarek_float64.Float64.sin

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
          | Sarek_ast.MConst (n, _, _) | Sarek_ast.MFun (n, _, _, _) -> n
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
  | Sarek_ast.MFun (name, _, _, _) ->
      Format.eprintf "Sarek PPX: register module fun %s@." name
  | Sarek_ast.MConst (name, _, _) ->
      Format.eprintf "Sarek PPX: register module const %s@." name) ;
  registered_mods := item :: !registered_mods

(** Scan a single .ml file for [@@sarek.type] and [@sarek.module] declarations
*)
let scan_file_for_sarek_types path =
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
        | Pstr_value (rec_flag, vbs) ->
            let is_rec = rec_flag = Recursive in
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
                        Sarek_ast.MFun (name, is_rec, params, body)
                    | _ ->
                        let value = Sarek_parse.parse_expression vb.pvb_expr in
                        let ty =
                          match ty with
                          | Some t -> t
                          | None ->
                              Location.raise_errorf
                                ~loc:vb.pvb_pat.ppat_loc
                                "[@sarek.module] constants require a type \
                                 annotation"
                        in
                        Sarek_ast.MConst (name, ty, value)
                  in
                  register_sarek_module_item ~loc:vb.pvb_loc item))
              vbs
        | _ -> ())
      st
  with _ -> ()

(** Scan a directory for .ml files with Sarek declarations, or scan a single
    file *)
let scan_dir_for_sarek_types ?single_file directory =
  match single_file with
  | Some path -> scan_file_for_sarek_types path
  | None ->
      Array.iter
        (fun fname ->
          if Filename.check_suffix fname ".ml" then
            let path = Filename.concat directory fname in
            scan_file_for_sarek_types path)
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

(* NOTE: %sarek_intrinsic and %sarek_extend are handled by sarek_ppx_intrinsic.
   This allows breaking the circular dependency between sarek_ppx and sarek_stdlib.
   sarek_stdlib uses sarek_ppx_intrinsic, and sarek_ppx links sarek_stdlib. *)

(** The main kernel expansion function *)
let expand_kernel ~ctxt payload =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in
  Sarek_debug.log "expand_kernel: %s" loc.loc_start.pos_fname ;

  try
    (* Scan the current file for [@sarek.module] and [@@sarek.type] declarations.
       This is needed because the impl pass (process_structure_for_module_items)
       runs AFTER extensions are expanded, so we need to scan now. *)
    let current_file = loc.loc_start.pos_fname in
    if Sys.file_exists current_file then scan_file_for_sarek_types current_file ;
    (* Types and module items registered in the current compilation unit. *)
    let pre_types = dedup_tdecls !registered_types in
    let local_mods = dedup_mods !registered_mods in
    (* Also include module items from the registry (populated by linked libraries) *)
    let registry_mods =
      List.map
        (fun (info : Sarek_ppx_registry.module_item_info) -> info.mi_item)
        (Sarek_ppx_registry.all_module_items ())
    in
    let pre_mods = dedup_mods (local_mods @ registry_mods) in
    Sarek_debug.log
      "pre_mods count=%d (local=%d, registry=%d)"
      (List.length pre_mods)
      (List.length local_mods)
      (List.length registry_mods) ;
    (* 1. Parse the PPX payload to Sarek AST *)
    Sarek_debug.log_enter "parse_payload" ;
    let ast = Sarek_parse.parse_payload payload in
    Sarek_debug.log_exit "parse_payload" ;
    let ast =
      {
        ast with
        Sarek_ast.kern_types = pre_types @ ast.kern_types;
        kern_module_items = pre_mods @ ast.kern_module_items;
        (* External items are prepended first, then inline items from payload.
           Track how many are external so native code gen can skip them. *)
        kern_external_item_count = List.length pre_mods;
      }
    in

    (* 2. Set up the typing environment with stdlib *)
    let env = Sarek_env.(empty |> with_stdlib) in

    (* 3. Type inference *)
    Sarek_debug.log_enter "infer_kernel" ;
    match Sarek_typer.infer_kernel env ast with
    | Error errors ->
        (* Report the first error with location *)
        Sarek_error.report_errors errors ;
        (* If we get here, generate dummy expression *)
        [%expr assert false]
    | Ok tkernel -> (
        Sarek_debug.log_exit "infer_kernel" ;
        (* 4. Convergence analysis - check barrier safety *)
        match Sarek_convergence.check_kernel tkernel with
        | Error (err :: _) ->
            (* Raise as Sarek_error to be caught by the handler below *)
            raise (Sarek_error.Sarek_error err)
        | Error [] | Ok () ->
            (* 5. Monomorphization pass - specialize polymorphic functions *)
            Sarek_debug.log_enter "monomorphize" ;
            let tkernel = Sarek_mono.monomorphize tkernel in
            Sarek_debug.log_exit "monomorphize" ;

            (* 6. Tail recursion elimination pass (for GPU code) *)
            (* Keep original kernel for native OCaml which handles recursion *)
            let native_kernel = tkernel in
            Sarek_debug.log_enter "transform_kernel" ;
            let tkernel = Sarek_tailrec.transform_kernel tkernel in
            Sarek_debug.log_exit "transform_kernel" ;

            (* 7. Lower to Kirc_Ast (legacy) *)
            Sarek_debug.log_enter "lower_kernel" ;
            let ir, constructors = Sarek_lower.lower_kernel tkernel in
            Sarek_debug.log_exit "lower_kernel" ;
            let ret_val = Sarek_lower.lower_return_value tkernel in

            (* 7b. Lower to Sarek_ir (V2) - optional, fails gracefully *)
            let v2_kernel =
              try
                Sarek_debug.log_enter "lower_kernel_v2" ;
                let k, _constructors_v2 =
                  Sarek_lower_v2.lower_kernel tkernel
                in
                Sarek_debug.log_exit "lower_kernel_v2" ;
                Some k
              with _ ->
                Sarek_debug.log_exit "lower_kernel_v2 (failed)" ;
                None
            in

            (* 8. Quote the IR back to OCaml *)
            Sarek_quote.quote_kernel
              ~loc
              ~native_kernel
              ?ir_v2:v2_kernel
              tkernel
              ir
              constructors
              ret_val)
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
    publish module items for use in kernels.

    This generates registration code that runs at module initialization time,
    registering the items in Sarek_ppx_registry. This allows cross-module
    references: a library can define [@sarek.module] items and link with the PPX
    so they become available to kernels in other compilation units. *)
let process_structure_for_module_items (str : structure) : structure =
  let process_vb ~is_rec vb =
    let has_attr name a = String.equal a.attr_name.txt name in
    let is_private =
      List.exists (has_attr "sarek.module_private") vb.pvb_attributes
    in
    if List.exists (has_attr "sarek.module") vb.pvb_attributes || is_private
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
            Sarek_ast.MFun (name, is_rec, params, body)
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
      (* Register locally for this compilation unit *)
      register_sarek_module_item ~loc item ;
      (* For non-private items, generate registration code for cross-module use *)
      if is_private then None
      else
        let module_name = module_name_of_loc loc in
        let name_str = Ast_builder.Default.estring ~loc name in
        let module_name_str = Ast_builder.Default.estring ~loc module_name in
        let item_expr = Sarek_quote.quote_sarek_module_item ~loc item in
        Some
          [%stri
            let () =
              Sarek_ppx_lib.Sarek_ppx_registry.register_module_item
                (Sarek_ppx_lib.Sarek_ppx_registry.make_module_item_info
                   ~name:[%e name_str]
                   ~module_name:[%e module_name_str]
                   ~item:[%e item_expr])])
    else None
  in
  let extra_items =
    List.filter_map
      (fun item ->
        match item.pstr_desc with
        | Pstr_value (rf, vbs) ->
            let is_rec = rf = Recursive in
            let registrations = List.filter_map (process_vb ~is_rec) vbs in
            if registrations = [] then None else Some registrations
        | _ -> None)
      str
    |> List.flatten
  in
  str @ extra_items

(** [%sarek_include "path/to/file.ml"] - Include types and module items from
    another file.

    This scans the specified file for [@@sarek.type] and [@sarek.module]
    declarations and registers them for use in kernels in the current file. The
    path is relative to the current file's directory.

    Usage: [%sarek_include "registered_defs.ml"]

    let kernel =
    [%kernel fun ... -> let open Registered_defs in ... use types and functions
     from registered_defs.ml ... ] *)
let expand_sarek_include ~ctxt payload =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in
  (* Extract path string from various payload forms *)
  let extract_path () =
    match payload with
    (* [%sarek_include "path"] *)
    | PStr
        [
          {
            pstr_desc =
              Pstr_eval
                ({pexp_desc = Pexp_constant (Pconst_string (path, _, _)); _}, _);
            _;
          };
        ] ->
        Some path
    (* let%sarek_include _ = "path" *)
    | PStr
        [
          {
            pstr_desc =
              Pstr_value
                ( _,
                  [
                    {
                      pvb_expr =
                        {
                          pexp_desc = Pexp_constant (Pconst_string (path, _, _));
                          _;
                        };
                      _;
                    };
                  ] );
            _;
          };
        ] ->
        Some path
    | _ -> None
  in
  match extract_path () with
  | Some path ->
      (* Resolve path relative to current file *)
      let current_file = loc.loc_start.pos_fname in
      let dir = Filename.dirname current_file in
      let full_path =
        if Filename.is_relative path then Filename.concat dir path else path
      in
      (* Scan the file for types and module items *)
      (try scan_file_for_sarek_types full_path
       with e ->
         Location.raise_errorf
           ~loc
           "%%sarek_include: failed to scan %s: %s"
           full_path
           (Printexc.to_string e)) ;
      (* Return empty structure item - the side effect is registration *)
      [%stri let () = ()]
  | None ->
      Location.raise_errorf
        ~loc
        "%%sarek_include expects a string path, e.g. [%%sarek_include \
         \"file.ml\"]"

let sarek_include_extension =
  Extension.V3.declare
    "sarek_include"
    Extension.Context.structure_item
    Ast_pattern.(pstr __)
    (fun ~ctxt payload -> expand_sarek_include ~ctxt (PStr payload))

(** Register the transformation *)
let () =
  let rules =
    [
      sarek_type_rule;
      sarek_type_private_rule;
      Context_free.Rule.extension kernel_extension;
      Context_free.Rule.extension sarek_include_extension;
      (* NOTE: %sarek_intrinsic and %sarek_extend are handled by sarek_ppx_intrinsic *)
    ]
  in
  Driver.register_transformation
    ~rules
    ~impl:process_structure_for_module_items
    "sarek_ppx"
