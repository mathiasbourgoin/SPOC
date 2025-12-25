(******************************************************************************
 * Sarek PPX Intrinsic - Minimal PPX for intrinsic type and function declarations
 *
 * This PPX handles:
 * - %sarek_intrinsic for defining types and functions
 * - %sarek_extend for extending intrinsics for new backends
 *
 * It generates both:
 * 1. Runtime registration (Sarek.Sarek_registry) for JIT code generation
 * 2. PPX-time registration (Sarek_ppx_lib.Sarek_ppx_registry) for compile-time type checking
 *
 * This is kept separate from the main kernel PPX to break the circular dependency:
 * - sarek_stdlib uses sarek_ppx_intrinsic (to process %sarek_intrinsic)
 * - sarek_ppx uses sarek_stdlib (to have types registered at PPX load time)
 ******************************************************************************)

open Ppxlib

(** Extract type name from a core_type for registry registration *)
let type_name_of_core_type (ct : core_type) : string =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident name; _}, _) -> name
  | Ptyp_constr ({txt = Ldot (_, name); _}, _) -> name
  | _ -> "unknown"

(** Get the module name from a location (used for qualified names) *)
let module_name_of_loc loc =
  let file = loc.loc_start.pos_fname in
  let base = Filename.(remove_extension (basename file)) in
  String.capitalize_ascii base

(** Convert type name to Sarek_types representation *)
let sarek_type_of_name ~loc (name : string) : expression =
  match name with
  | "unit" -> [%expr Sarek_ppx_lib.Sarek_types.t_unit]
  | "bool" -> [%expr Sarek_ppx_lib.Sarek_types.t_bool]
  | "int" | "int32" -> [%expr Sarek_ppx_lib.Sarek_types.t_int32]
  | "int64" -> [%expr Sarek_ppx_lib.Sarek_types.t_int64]
  | "float32" -> [%expr Sarek_ppx_lib.Sarek_types.t_float32]
  | "float" | "float64" -> [%expr Sarek_ppx_lib.Sarek_types.t_float64]
  | _ ->
      (* For unknown types, use TReg with the name *)
      let name_expr = Ast_builder.Default.estring ~loc name in
      [%expr Sarek_ppx_lib.Sarek_types.TReg [%e name_expr]]

(** Build a function type from argument types and return type *)
let build_sarek_fun_type ~loc (arg_types : string list) (ret_type : string) :
    expression =
  match arg_types with
  | [] -> sarek_type_of_name ~loc ret_type
  | args ->
      let arg_exprs =
        Ast_builder.Default.elist ~loc (List.map (sarek_type_of_name ~loc) args)
      in
      let ret_expr = sarek_type_of_name ~loc ret_type in
      [%expr Sarek_ppx_lib.Sarek_types.TFun ([%e arg_exprs], [%e ret_expr])]

(** Extension for type%sarek_intrinsic - handles type registration *)
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
      let type_name_str = Ast_builder.Default.estring ~loc type_name in
      let _module_name = module_name_of_loc loc in
      let sarek_type = sarek_type_of_name ~loc type_name in
      [
        (* Runtime registration for JIT *)
        [%stri
          let () =
            Sarek.Sarek_registry.register_type
              [%e type_name_str]
              ~device:[%e device_expr]
              ~size:(Ctypes.sizeof [%e ctype_expr])];
        (* PPX-time registration for compile-time type checking *)
        [%stri
          let () =
            Sarek_ppx_lib.Sarek_ppx_registry.register_type
              (Sarek_ppx_lib.Sarek_ppx_registry.make_type_info
                 ~name:[%e type_name_str]
                 ~device:[%e device_expr]
                 ~size:(Ctypes.sizeof [%e ctype_expr])
                 ~sarek_type:[%e sarek_type])];
      ]
  | _ ->
      Location.raise_errorf
        ~loc
        "type%%sarek_intrinsic expects: let%%sarek_intrinsic name = { device = \
         ...; ctype = ... }"

(** Extract argument types and return type from a function type *)
let rec extract_fun_types (ct : core_type) : string list * string =
  match ct.ptyp_desc with
  | Ptyp_arrow (_, arg_type, ret_type) ->
      let more_args, final_ret = extract_fun_types ret_type in
      (type_name_of_core_type arg_type :: more_args, final_ret)
  | _ -> ([], type_name_of_core_type ct)

(** Extension for let%sarek_intrinsic - handles function registration *)
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
      (* Build names and expressions *)
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

      (* Module path for PPX-time registration *)
      let module_name = module_name_of_loc loc in
      let module_path_expr =
        Ast_builder.Default.elist
          ~loc
          [Ast_builder.Default.estring ~loc module_name]
      in

      (* Build Sarek type for PPX-time registration *)
      let sarek_fun_type = build_sarek_fun_type ~loc arg_types ret_type in

      [
        (* Expose the device function for extensions to chain to *)
        [%stri
          let [%p device_fun_pat] : Spoc.Devices.device -> string =
            [%e device_expr]];
        (* Mutable ref for extension chaining *)
        [%stri
          let [%p device_fun_ref_pat] : (Spoc.Devices.device -> string) ref =
            ref [%e Ast_builder.Default.evar ~loc device_fun_name]];
        (* Runtime registration for JIT - uses the ref for extensibility *)
        [%stri
          let () =
            Sarek.Sarek_registry.register_fun
              [%e fun_name_str]
              ~arity:[%e arity_expr]
              ~device:(fun dev -> ![%e device_fun_ref_expr] dev)
              ~arg_types:[%e arg_types_expr]
              ~ret_type:[%e ret_type_expr]];
        (* PPX-time registration for compile-time type checking *)
        [%stri
          let () =
            Sarek_ppx_lib.Sarek_ppx_registry.register_intrinsic
              (Sarek_ppx_lib.Sarek_ppx_registry.make_intrinsic_info
                 ~name:[%e fun_name_str]
                 ~module_path:[%e module_path_expr]
                 ~typ:[%e sarek_fun_type]
                 ~device:(fun dev -> ![%e device_fun_ref_expr] dev))];
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

(** Register the transformation *)
let () =
  Driver.register_transformation
    ~rules:
      [
        Context_free.Rule.extension sarek_intrinsic_extension;
        Context_free.Rule.extension sarek_extend_extension;
      ]
    "sarek_ppx_intrinsic"
