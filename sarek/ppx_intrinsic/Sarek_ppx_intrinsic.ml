(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX Intrinsic - Minimal PPX for intrinsic type and function declarations
 *
 * This PPX handles:
 * - %sarek_intrinsic for defining types and functions
 * - %sarek_extend for extending intrinsics for new backends
 *
 * It generates both:
 * 1. Runtime registration (Sarek_registry) for JIT code generation
 * 2. PPX-time registration (Sarek_ppx_registry) for compile-time type checking
 *
 * This is kept separate from the main kernel PPX to break the circular dependency:
 * - sarek_stdlib uses sarek_ppx_intrinsic (to process %sarek_intrinsic)
 * - sarek_ppx uses sarek_stdlib (to have types registered at PPX load time)
 ******************************************************************************)

open Ppxlib

(** Parsed type representation for intrinsic declarations *)
type parsed_type =
  | PTSimple of string (* e.g., "int32", "float" *)
  | PTArray of parsed_type (* e.g., "int32 array" - shared/local memory *)
  | PTVec of parsed_type (* e.g., "int32 vector" - global memory *)
  | PTArrow of parsed_type * parsed_type (* for function types if needed *)

(** Extract parsed type from a core_type *)
let rec parsed_type_of_core_type (ct : core_type) : parsed_type =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident "array"; _}, [elem_type]) ->
      PTArray (parsed_type_of_core_type elem_type)
  | Ptyp_constr ({txt = Lident "vector"; _}, [elem_type]) ->
      (* Vector is for global memory *)
      PTVec (parsed_type_of_core_type elem_type)
  | Ptyp_constr ({txt = Lident name; _}, _) -> PTSimple name
  | Ptyp_constr ({txt = Ldot (_, name); _}, _) -> PTSimple name
  | Ptyp_arrow (_, arg, ret) ->
      PTArrow (parsed_type_of_core_type arg, parsed_type_of_core_type ret)
  | _ -> PTSimple "unknown"

(** Extract type name from a core_type for registry registration (string
    representation) *)
let rec type_name_of_core_type (ct : core_type) : string =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident "array"; _}, [elem_type]) ->
      type_name_of_core_type elem_type ^ " array"
  | Ptyp_constr ({txt = Lident "vector"; _}, [elem_type]) ->
      type_name_of_core_type elem_type ^ " vector"
  | Ptyp_constr ({txt = Lident name; _}, _) -> name
  | Ptyp_constr ({txt = Ldot (_, name); _}, _) -> name
  | _ -> "unknown"

(** Name of the dune library whose module is being preprocessed, from the
    [library-name] cookie dune passes to every ppx driver it runs on a library's
    sources ([--cookie 'library-name="foo"']). [None] when the file is not part
    of a library (an executable or test stanza) or when the driver is run
    outside dune. *)
let library_name : string option ref = ref None

let () =
  Driver.Cookies.add_simple_handler
    "library-name"
    Ast_pattern.(estring __)
    ~f:(fun name -> library_name := name)

(** The OCaml path under which the module being preprocessed is reachable from a
    consumer, e.g. [["Sarek_stdlib"; "Float32"]] for
    [sarek/Sarek_stdlib/Float32.ml]. Both registries key the intrinsic on it,
    and the kernel PPX emits it verbatim as the existence witness and as the
    Native backend's call target, so it must be the REAL path.

    It comes from the library name dune hands the driver, under dune's default
    [(wrapped true)]: module [M] of library [lib] is [Lib.M], and the library's
    main module [lib.ml] is [Lib] itself. Outside a library it is [[M]].

    It used to be guessed from the source DIRECTORY: a directory name containing
    an underscore was taken to be the library name, any other was dropped. That
    held for sarek's own trees, whose directories are named after their
    libraries, and nowhere else: a downstream library [foo_intrinsics] living in
    [src/intrinsics/] registered [Bits.f] instead of [Foo_intrinsics.Bits.f],
    and renaming the directory changed the path.

    A [(wrapped false)] library is not detectable from the driver: its modules
    are top-level, but the path computed here still carries the library prefix,
    and the witness then fails to compile ("Unbound module"), loudly. Declare
    intrinsics in a wrapped library. *)
let module_path_of_loc loc =
  let file = loc.loc_start.pos_fname in
  let module_name =
    String.capitalize_ascii Filename.(remove_extension (basename file))
  in
  match !library_name with
  | None -> [module_name]
  | Some lib ->
      let lib = String.capitalize_ascii lib in
      if String.equal lib module_name then [lib] else [lib; module_name]

(** Convert simple type name to Sarek_types representation *)
let sarek_type_of_simple_name ~loc (name : string) : expression =
  match name with
  | "unit" -> [%expr Sarek_types.t_unit]
  | "bool" -> [%expr Sarek_types.t_bool]
  | "int" -> [%expr Sarek_types.t_int]
  | "int32" -> [%expr Sarek_types.t_int32]
  | "int64" -> [%expr Sarek_types.t_int64]
  | "float32" -> [%expr Sarek_types.t_float32]
  | "float" | "float64" -> [%expr Sarek_types.t_float64]
  | "char" -> [%expr Sarek_types.t_char]
  | _ ->
      (* For unknown types, use TReg (Custom name) *)
      let name_expr = Ast_builder.Default.estring ~loc name in
      [%expr Sarek_types.TReg (Sarek_types.Custom [%e name_expr])]

(** Flatten a parsed arrow type into (arg_types, return_type). E.g., PTArrow(a,
    PTArrow(b, c)) becomes ([a; b], c) *)
let rec flatten_arrow (pt : parsed_type) : parsed_type list * parsed_type =
  match pt with
  | PTArrow (arg, ret) ->
      let args, final_ret = flatten_arrow ret in
      (arg :: args, final_ret)
  | _ -> ([], pt)

(** Convert a non-arrow parsed type to Sarek_types representation *)
let rec sarek_type_of_simple_parsed ~loc (pt : parsed_type) : expression =
  match pt with
  | PTSimple name -> sarek_type_of_simple_name ~loc name
  | PTArray elem_type ->
      let elem_expr = sarek_type_of_simple_parsed ~loc elem_type in
      (* Use Local memory space for intrinsic array args (shared memory) *)
      [%expr Sarek_types.TArr ([%e elem_expr], Sarek_types.Local)]
  | PTVec elem_type ->
      let elem_expr = sarek_type_of_simple_parsed ~loc elem_type in
      (* Vector is global memory *)
      [%expr Sarek_types.TVec [%e elem_expr]]
  | PTArrow _ ->
      (* Should not happen after flattening, but handle gracefully *)
      sarek_type_of_simple_name ~loc "unknown"

(** Convert parsed type to Sarek_types representation. Flattens arrow types so a
    -> b -> c becomes TFun([a; b], c). *)
let sarek_type_of_parsed ~loc (pt : parsed_type) : expression =
  match pt with
  | PTArrow _ ->
      let args, ret = flatten_arrow pt in
      let arg_exprs =
        Ast_builder.Default.elist
          ~loc
          (List.map (sarek_type_of_simple_parsed ~loc) args)
      in
      let ret_expr = sarek_type_of_simple_parsed ~loc ret in
      [%expr Sarek_types.TFun ([%e arg_exprs], [%e ret_expr])]
  | _ -> sarek_type_of_simple_parsed ~loc pt

(** Convert type name string to Sarek_types representation (for backward compat)
*)
let sarek_type_of_name ~loc (name : string) : expression =
  (* Parse compound types like "int32 array" or "int32 vector" *)
  if
    String.length name > 6
    && String.sub name (String.length name - 6) 6 = " array"
  then
    let elem_name = String.sub name 0 (String.length name - 6) in
    let elem_expr = sarek_type_of_simple_name ~loc elem_name in
    [%expr Sarek_types.TArr ([%e elem_expr], Sarek_types.Local)]
  else if
    String.length name > 7
    && String.sub name (String.length name - 7) 7 = " vector"
  then
    let elem_name = String.sub name 0 (String.length name - 7) in
    let elem_expr = sarek_type_of_simple_name ~loc elem_name in
    [%expr Sarek_types.TVec [%e elem_expr]]
  else sarek_type_of_simple_name ~loc name

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
      [%expr Sarek_types.TFun ([%e arg_exprs], [%e ret_expr])]

(** Derive the size (in bytes) of a sarek type from its name alone. This avoids
    the Ctypes dependency for the pure metadata path. Sizes match the
    Ctypes.sizeof values for the corresponding C types: float32=4, float64=8,
    int32=4, int64=8, bool=4, char=1, int=4 (GPU int). *)
let size_of_type_name ~loc (name : string) : int =
  match name with
  | "float32" | "float" -> 4
  | "float64" -> 8
  | "int32" | "int" -> 4
  | "int64" -> 8
  | "bool" -> 4
  | "char" -> 1
  | "unit" -> 0
  | _ ->
      Location.raise_errorf
        ~loc
        "type%%sarek_intrinsic: cannot derive a byte size for unknown type %S \
         in the pure (ctype-absent) path. Add a `ctype` field (authoritative \
         size via Ctypes.sizeof) or extend size_of_type_name."
        name

(** Extension for type%%sarek_intrinsic - handles type registration. The [ctype]
    field is OPTIONAL.
    - When present: emits both Sarek_registry (JIT/FFI) and Sarek_ppx_registry
      registrations using Ctypes.sizeof for the authoritative byte-size.
    - When absent: emits only Sarek_ppx_registry registration using a pure size
      derived from the type name. No Ctypes dependency. *)
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
      ] -> (
      (* Extract device and optional ctype fields from the record expression *)
      let device_expr, ctype_expr_opt =
        match record_expr.pexp_desc with
        | Pexp_record (fields, None) ->
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
              find_field "ctype" )
        | _ ->
            Location.raise_errorf
              ~loc
              "sarek_intrinsic type expects a record { device = ...; ctype = \
               ... }"
      in
      let type_name_str = Ast_builder.Default.estring ~loc type_name in
      let _module_path = module_path_of_loc loc in
      let sarek_type = sarek_type_of_name ~loc type_name in
      match ctype_expr_opt with
      | Some ctype_expr ->
          (* FFI path: ctype present - emit both JIT and PPX-time registrations
             using Ctypes.sizeof for the authoritative byte-size. *)
          [
            (* Runtime registration for JIT *)
            [%stri
              let () =
                Sarek_registry.register_type
                  [%e type_name_str]
                  ~device:[%e device_expr]
                  ~size:(Ctypes.sizeof [%e ctype_expr])];
            (* PPX-time registration for compile-time type checking *)
            [%stri
              let () =
                Sarek_ppx_registry.register_type
                  (Sarek_ppx_registry.make_type_info
                     ~name:[%e type_name_str]
                     ~size:(Ctypes.sizeof [%e ctype_expr])
                     ~sarek_type:[%e sarek_type])];
          ]
      | None ->
          (* Pure metadata path: ctype absent - emit only the PPX-time
             registration, deriving size from the type name. No Ctypes used. *)
          let size_int = size_of_type_name ~loc type_name in
          let size_expr = Ast_builder.Default.eint ~loc size_int in
          [
            [%stri
              let () =
                ignore [%e device_expr] ;
                Sarek_ppx_registry.register_type
                  (Sarek_ppx_registry.make_type_info
                     ~name:[%e type_name_str]
                     ~size:[%e size_expr]
                     ~sarek_type:[%e sarek_type])];
          ])
  | _ ->
      Location.raise_errorf
        ~loc
        "type%%sarek_intrinsic expects: let%%sarek_intrinsic name = { device = \
         ...; ctype = ... } (ctype optional: omit it for the pure metadata \
         path, where the size is derived from the type name)"

(** Extract argument types and return type from a function type *)
let rec extract_fun_types (ct : core_type) : string list * string =
  match ct.ptyp_desc with
  | Ptyp_arrow (_, arg_type, ret_type) ->
      let more_args, final_ret = extract_fun_types ret_type in
      (type_name_of_core_type arg_type :: more_args, final_ret)
  | _ -> ([], type_name_of_core_type ct)

(** The {!Sarek_registry.host_value} constructor carrying a scalar of the given
    declared type, or [None] when the type is not a scalar the registry's host
    path can carry ([int], arrays, vectors, registered custom types). *)
let host_ctor_of_type_name = function
  | "int32" -> Some "HInt32"
  | "int64" -> Some "HInt64"
  | "float32" -> Some "HFloat32"
  | "float" | "float64" -> Some "HFloat64"
  | "bool" -> Some "HBool"
  | "unit" -> Some "HUnit"
  | _ -> None

(** Registration of the host implementation [fun_name] (already bound) for
    evaluators without a built-in entry for it (the interpreter). [None] when
    the signature has no argument or a non-scalar type: such an intrinsic is
    left to the evaluator's own table, as before. *)
let host_impl_registration ~loc ~module_path_expr ~fun_name arg_types ret_type =
  let ctors = List.map host_ctor_of_type_name (arg_types @ [ret_type]) in
  if arg_types = [] || List.mem None ctors then None
  else
    let ctor name = {txt = Ldot (Lident "Sarek_registry", name); loc} in
    let arg_ctors =
      List.map (fun t -> Option.get (host_ctor_of_type_name t)) arg_types
    in
    let ret_ctor = Option.get (host_ctor_of_type_name ret_type) in
    let pats, args =
      List.split
        (List.mapi
           (fun i c ->
             if String.equal c "HUnit" then
               ( Ast_builder.Default.ppat_construct ~loc (ctor c) None,
                 [%expr ()] )
             else
               let v = Printf.sprintf "a%d" i in
               ( Ast_builder.Default.ppat_construct
                   ~loc
                   (ctor c)
                   (Some (Ast_builder.Default.pvar ~loc v)),
                 Ast_builder.Default.evar ~loc v ))
           arg_ctors)
    in
    let list_pat =
      List.fold_right
        (fun p acc ->
          Ast_builder.Default.ppat_construct
            ~loc
            {txt = Lident "::"; loc}
            (Some (Ast_builder.Default.ppat_tuple ~loc [p; acc])))
        pats
        (Ast_builder.Default.ppat_construct ~loc {txt = Lident "[]"; loc} None)
    in
    let call =
      Ast_builder.Default.pexp_apply
        ~loc
        (Ast_builder.Default.evar ~loc fun_name)
        (List.map (fun a -> (Nolabel, a)) args)
    in
    let result =
      if String.equal ret_ctor "HUnit" then
        [%expr
          [%e call] ;
          Sarek_registry.HUnit]
      else Ast_builder.Default.pexp_construct ~loc (ctor ret_ctor) (Some call)
    in
    let mismatch =
      Ast_builder.Default.estring
        ~loc
        (fun_name ^ ": host arguments do not match the declared signature")
    in
    let name_expr = Ast_builder.Default.estring ~loc fun_name in
    Some
      [%stri
        let () =
          Sarek_registry.register_host_impl
            ~module_path:[%e module_path_expr]
            [%e name_expr]
            (function
            | [%p list_pat] -> [%e result]
            | _ -> invalid_arg [%e mismatch])]

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

      (* Module path for both runtime and PPX-time registration *)
      let module_path = module_path_of_loc loc in
      let module_path_expr =
        Ast_builder.Default.elist
          ~loc
          (List.map (Ast_builder.Default.estring ~loc) module_path)
      in

      (* Build Sarek type for PPX-time registration *)
      let sarek_fun_type = build_sarek_fun_type ~loc arg_types ret_type in

      [
        (* Expose the device function for extensions to chain to.
           The user provides the device function directly. *)
        [%stri let [%p device_fun_pat] : string -> string = [%e device_expr]];
        (* Mutable ref for extension chaining *)
        [%stri
          let [%p device_fun_ref_pat] : (string -> string) ref =
            ref [%e Ast_builder.Default.evar ~loc device_fun_name]];
        (* Runtime registration for JIT - uses the ref for extensibility *)
        [%stri
          let () =
            Sarek_registry.register_fun
              ~module_path:[%e module_path_expr]
              [%e fun_name_str]
              ~arity:[%e arity_expr]
              ~device:(fun dev -> ![%e device_fun_ref_expr] dev)
              ~arg_types:[%e arg_types_expr]
              ~ret_type:[%e ret_type_expr]];
        (* PPX-time registration for compile-time type checking.
           We register the device function ref so that extensions work correctly.
           The ref is dereferenced at lookup time, allowing %sarek_extend to work. *)
        [%stri
          let () =
            Sarek_ppx_registry.register_intrinsic
              (Sarek_ppx_registry.make_intrinsic_info
                 ~name:[%e fun_name_str]
                 ~module_path:[%e module_path_expr]
                 ~typ:[%e sarek_fun_type])];
        (* Expose the OCaml implementation for host-side use *)
        [%stri let [%p fun_name_pat] = [%e ocaml_expr]];
      ]
      @ Option.to_list
          (host_impl_registration
             ~loc
             ~module_path_expr
             ~fun_name
             arg_types
             ret_type)
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
  (* Type declaration: let name = { device = ...[; ctype = ...] }
     Ppat_var (no type constraint) identifies this as a type registration.
     ctype is optional: present in FFI libs, absent in pure metadata libs. *)
  | PStr
      [
        {
          pstr_desc =
            Pstr_value
              ( _,
                [
                  {
                    pvb_pat = {ppat_desc = Ppat_var _; _};
                    pvb_expr = {pexp_desc = Pexp_record (_, None); _};
                    _;
                  };
                ] );
          _;
        };
      ] ->
      expand_sarek_intrinsic_type ~ctxt payload
  (* Function declaration: let (name : type) = { device = ...; ocaml = ... } *)
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
