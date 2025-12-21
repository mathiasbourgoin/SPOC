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

let rec core_type_to_sarek_type_expr ~loc (ct : core_type) =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident name; _}, args) ->
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
            tdecl_constructors = cons;
            tdecl_loc = Sarek_ast.loc_of_ppxlib loc;
          }
    | _ ->
        Location.raise_errorf
          ~loc
          "Only record or variant types can be used with [@@sarek.type]"
  in
  registered_types := tdecl :: !registered_types

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
    (* Pre-registered types (top-level sarek.type) *)
    let file = loc.loc_start.pos_fname in
    let pre_types =
      List.filter
        (function
          | Sarek_ast.Type_record {tdecl_loc; _}
          | Sarek_ast.Type_variant {tdecl_loc; _} ->
              String.equal tdecl_loc.loc_file file)
        !registered_types
    in
    (* 1. Parse the PPX payload to Sarek AST *)
    let ast = Sarek_parse.parse_payload payload in
    let ast = {ast with Sarek_ast.kern_types = pre_types @ ast.kern_types} in

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
