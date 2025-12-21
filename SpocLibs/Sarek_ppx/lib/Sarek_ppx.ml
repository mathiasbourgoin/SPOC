(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This is the PPX entry point. It registers the [%kernel ...] extension
 * and orchestrates parsing, type checking, lowering, and code generation.
 ******************************************************************************)

open Ppxlib
open Sarek_ppx_lib

(* Registry of globally declared Sarek types (via [@@sarek.type]). *)
let registered_types : Sarek_ast.type_decl list ref = ref []
(* Registry of globally declared Sarek module items (functions/constants) *)
let registered_mods : Sarek_ast.module_item list ref = ref []

let registry_file _loc =
  match Sys.getenv_opt "SAREK_PPX_REGISTRY" with
  | Some p -> p
  | None ->
      Filename.concat (Filename.get_temp_dir_name ()) "sarek_types_registry"

let registry_mod_file _loc =
  match Sys.getenv_opt "SAREK_PPX_MODREG" with
  | Some p -> p
  | None ->
      Filename.concat (Filename.get_temp_dir_name ()) "sarek_module_registry"

let load_registry loc =
  try
    let ic = open_in_bin (registry_file loc) in
    let rec loop acc =
      try
        let v = Marshal.from_channel ic in
        loop (v :: acc)
      with End_of_file -> List.rev acc
    in
    let res = loop [] in
    close_in ic ;
    res
  with Sys_error _ -> []

let append_registry loc tdecl =
  try
    let oc = open_out_gen [Open_creat; Open_append; Open_binary] 0o644 (registry_file loc) in
    Marshal.to_channel oc tdecl [] ;
    close_out oc
  with Sys_error msg ->
    Format.eprintf "Sarek PPX: cannot persist registry (%s)@." msg

let load_mod_registry loc =
  try
    let ic = open_in_bin (registry_mod_file loc) in
    let rec loop acc =
      try
        let v = Marshal.from_channel ic in
        loop (v :: acc)
      with End_of_file -> List.rev acc
    in
    let res = loop [] in
    close_in ic ;
    res
  with Sys_error _ -> []

let append_mod_registry loc item =
  try
    let oc =
      open_out_gen [Open_creat; Open_append; Open_binary] 0o644
        (registry_mod_file loc)
    in
    Marshal.to_channel oc item [] ;
    close_out oc
  with Sys_error msg ->
    Format.eprintf "Sarek PPX: cannot persist mod registry (%s)@." msg

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
        if S.mem key seen then (seen, acc)
        else (S.add key seen, d :: acc))
      (S.empty, []) decls
  in
  List.rev revs

let dedup_mods mods =
  let module S = Set.Make (String) in
  let _, revs =
    List.fold_left
      (fun (seen, acc) m ->
        let key =
          match m with Sarek_ast.MConst (n, _, _) | Sarek_ast.MFun (n, _, _) -> n
        in
        if S.mem key seen then (seen, acc)
        else (S.add key seen, m :: acc))
      (S.empty, []) mods
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
  append_registry loc tdecl

let register_sarek_module_item ~loc item =
  registered_mods := item :: !registered_mods ;
  append_mod_registry loc item

let scan_dir_for_sarek_types directory =
  Array.iter
    (fun fname ->
      if Filename.check_suffix fname ".ml" then
        let path = Filename.concat directory fname in
        try
          let ic = open_in path in
          let lexbuf = Lexing.from_channel ic in
          lexbuf.lex_curr_p <-
            {
              lexbuf.lex_curr_p with
              pos_fname = path;
              pos_bol = 0;
              pos_lnum = 1;
            } ;
          let st = Parse.implementation lexbuf in
          close_in ic ;
          List.iter
            (fun item ->
              match item.pstr_desc with
              | Pstr_type (_rf, decls) ->
                  List.iter
                    (fun d ->
                      let has_attr a = String.equal a.attr_name.txt "sarek.type" in
                      if List.exists has_attr d.ptype_attributes then
                        register_sarek_type_decl ~loc:d.ptype_loc d)
                    decls
              | Pstr_value (Nonrecursive, vbs) ->
                  List.iter
                    (fun vb ->
                      let has_attr a =
                        String.equal a.attr_name.txt "sarek.module"
                      in
                      if List.exists has_attr vb.pvb_attributes then
                        let name =
                          match Sarek_parse.extract_name_from_pattern vb.pvb_pat with
                          | Some n -> n
                          | None ->
                              Location.raise_errorf
                                ~loc:vb.pvb_pat.ppat_loc "Expected variable name"
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
                                    Location.raise_errorf ~loc:vb.pvb_pat.ppat_loc
                                      "[@sarek.module] constants require a type annotation"
                              in
                              Sarek_ast.MConst (name, ty, value)
                        in
                        register_sarek_module_item ~loc:vb.pvb_loc item)
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

(* Context-free rule to capture [@@sarek.type] before kernel expansion *)
let sarek_type_rule =
  Context_free.Rule.attr_str_type_decl
    sarek_type_attr
    (fun ~ctxt:_ _rec_flag decls payloads ->
      List.iter2
        (fun td payload ->
          match payload with
          | Some () -> register_sarek_type_decl ~loc:td.ptype_loc td
          | None -> ())
        decls
        payloads ;
      [])

(** The main kernel expansion function *)
let expand_kernel ~ctxt payload =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in

  try
    let dir = Filename.dirname loc.loc_start.pos_fname in
    scan_dir_for_sarek_types dir ;
    let real_dir =
      Filename.dirname (Unix.realpath loc.loc_start.pos_fname)
    in
    if not (String.equal dir real_dir) then scan_dir_for_sarek_types real_dir ;
    (match Sys.getenv_opt "PWD" with
    | Some cwd -> (
        let source_dir = Filename.concat cwd dir in
        try if Sys.is_directory source_dir then scan_dir_for_sarek_types source_dir
        with Sys_error _ -> ())
    | None -> ());
    (* Pre-registered types (top-level sarek.type) *)
    let pre_types = dedup_tdecls (load_registry loc @ !registered_types) in
    let pre_mods = dedup_mods (load_mod_registry loc @ !registered_mods) in
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

(** The [%kernel ...] extension *)
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

(** Register the transformation *)
let () =
  let rules = [sarek_type_rule; Context_free.Rule.extension kernel_extension] in
  Driver.register_transformation ~rules "sarek_ppx"
