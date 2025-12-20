(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module parses OCaml AST (from ppxlib) into Sarek_ast.
 ******************************************************************************)

open Ppxlib

(** Convert ppxlib location to Sarek location *)
let loc_of_ppxlib = Sarek_ast.loc_of_ppxlib

(** Parse exception *)
exception Parse_error_exn of string * Location.t

(** Parse a core_type to type_expr *)
let rec parse_type (ct : core_type) : Sarek_ast.type_expr =
  match ct.ptyp_desc with
  | Ptyp_constr ({ txt = Lident name; _ }, []) ->
    Sarek_ast.TEConstr (name, [])
  | Ptyp_constr ({ txt = Lident name; _ }, args) ->
    Sarek_ast.TEConstr (name, List.map parse_type args)
  | Ptyp_constr ({ txt = Ldot (Lident _mod, name); _ }, args) ->
    Sarek_ast.TEConstr (name, List.map parse_type args)
  | Ptyp_var name ->
    Sarek_ast.TEVar name
  | Ptyp_arrow (_, t1, t2) ->
    Sarek_ast.TEArrow (parse_type t1, parse_type t2)
  | Ptyp_tuple ts ->
    Sarek_ast.TETuple (List.map parse_type ts)
  | _ ->
    Sarek_ast.TEConstr ("unknown", [])

(** Extract type annotation from a Ppxlib pattern if present *)
let rec extract_type_from_pattern (pat : Ppxlib.pattern) : Sarek_ast.type_expr option =
  match pat.ppat_desc with
  | Ppat_constraint (_, ct) -> Some (parse_type ct)
  | Ppat_alias (p, _) -> extract_type_from_pattern p
  | _ -> None

(** Extract variable name from a Ppxlib pattern *)
let rec extract_name_from_pattern (pat : Ppxlib.pattern) : string option =
  match pat.ppat_desc with
  | Ppat_var { txt; _ } -> Some txt
  | Ppat_constraint (p, _) -> extract_name_from_pattern p
  | Ppat_alias (_, { txt; _ }) -> Some txt
  | Ppat_any -> Some "_"
  | _ -> None

(** Parse a Ppxlib pattern to Sarek pattern *)
let rec parse_pattern (pat : Ppxlib.pattern) : Sarek_ast.pattern =
  let loc = loc_of_ppxlib pat.ppat_loc in
  let pat_desc = match pat.ppat_desc with
    | Ppat_any -> Sarek_ast.PAny
    | Ppat_var { txt; _ } -> Sarek_ast.PVar txt
    | Ppat_constraint (p, _) -> (parse_pattern p).Sarek_ast.pat
    | Ppat_construct ({ txt = Lident name; _ }, None) ->
      Sarek_ast.PConstr (name, None)
    | Ppat_construct ({ txt = Lident name; _ }, Some (_, arg)) ->
      Sarek_ast.PConstr (name, Some (parse_pattern arg))
    | Ppat_tuple ps ->
      Sarek_ast.PTuple (List.map parse_pattern ps)
    | _ ->
      raise (Parse_error_exn ("Unsupported pattern", pat.ppat_loc))
  in
  { Sarek_ast.pat = pat_desc; Sarek_ast.pat_loc = loc }

(** Parse a binary operator *)
let parse_binop (op : string) : Sarek_ast.binop option =
  match op with
  | "+" | "+." -> Some Sarek_ast.Add
  | "-" | "-." -> Some Sarek_ast.Sub
  | "*" | "*." -> Some Sarek_ast.Mul
  | "/" | "/." -> Some Sarek_ast.Div
  | "mod" -> Some Sarek_ast.Mod
  | "=" -> Some Sarek_ast.Eq
  | "<>" | "!=" -> Some Sarek_ast.Ne
  | "<" -> Some Sarek_ast.Lt
  | "<=" -> Some Sarek_ast.Le
  | ">" -> Some Sarek_ast.Gt
  | ">=" -> Some Sarek_ast.Ge
  | "&&" -> Some Sarek_ast.And
  | "||" -> Some Sarek_ast.Or
  | "land" -> Some Sarek_ast.Land
  | "lor" -> Some Sarek_ast.Lor
  | "lxor" -> Some Sarek_ast.Lxor
  | "lsl" -> Some Sarek_ast.Lsl
  | "lsr" -> Some Sarek_ast.Lsr
  | "asr" -> Some Sarek_ast.Asr
  | _ -> None

(** Parse a unary operator *)
let parse_unop (op : string) : Sarek_ast.unop option =
  match op with
  | "-" | "-." | "~-" | "~-." -> Some Sarek_ast.Neg
  | "not" -> Some Sarek_ast.Not
  | "lnot" -> Some Sarek_ast.Lnot
  | _ -> None

(** Parse an expression *)
let rec parse_expression (expr : expression) : Sarek_ast.expr =
  let loc = loc_of_ppxlib expr.pexp_loc in
  let e = match expr.pexp_desc with
    (* Unit *)
    | Pexp_construct ({ txt = Lident "()"; _ }, None) ->
      Sarek_ast.EUnit

    (* Boolean literals *)
    | Pexp_construct ({ txt = Lident "true"; _ }, None) ->
      Sarek_ast.EBool true
    | Pexp_construct ({ txt = Lident "false"; _ }, None) ->
      Sarek_ast.EBool false

    (* Integer literals *)
    | Pexp_constant (Pconst_integer (s, Some 'l')) ->
      Sarek_ast.EInt32 (Int32.of_string s)
    | Pexp_constant (Pconst_integer (s, Some 'L')) ->
      Sarek_ast.EInt64 (Int64.of_string s)
    | Pexp_constant (Pconst_integer (s, None)) ->
      Sarek_ast.EInt (int_of_string s)

    (* Float literals *)
    | Pexp_constant (Pconst_float (s, _)) ->
      Sarek_ast.EFloat (float_of_string s)

    (* Variables *)
    | Pexp_ident { txt = Lident name; _ } ->
      Sarek_ast.EVar name

    (* Module-qualified identifiers *)
    | Pexp_ident { txt = Ldot (Lident _mod, name); _ } ->
      Sarek_ast.EVar name  (* For now, just use the name *)

    (* Vector/array access: e.(i) or e.[i] *)
    | Pexp_apply (
        { pexp_desc = Pexp_ident { txt = Lident "Array.get"; _ }; _ },
        [(Nolabel, arr); (Nolabel, idx)]) ->
      Sarek_ast.EArrGet (parse_expression arr, parse_expression idx)

    | Pexp_apply (
        { pexp_desc = Pexp_ident { txt = Ldot (Lident "Array", "get"); _ }; _ },
        [(Nolabel, arr); (Nolabel, idx)]) ->
      Sarek_ast.EArrGet (parse_expression arr, parse_expression idx)

    (* Vector/array set: e.(i) <- x *)
    | Pexp_apply (
        { pexp_desc = Pexp_ident { txt = Ldot (Lident "Array", "set"); _ }; _ },
        [(Nolabel, arr); (Nolabel, idx); (Nolabel, value)]) ->
      Sarek_ast.EArrSet (parse_expression arr, parse_expression idx, parse_expression value)

    (* Mutable assignment: x := v *)
    | Pexp_apply (
        { pexp_desc = Pexp_ident { txt = Lident ":="; _ }; _ },
        [(Nolabel, lhs); (Nolabel, rhs)]) ->
      (match lhs.pexp_desc with
       | Pexp_ident { txt = Lident name; _ } ->
         Sarek_ast.EAssign (name, parse_expression rhs)
       | _ ->
         raise (Parse_error_exn ("Expected variable on left-hand side of :=", lhs.pexp_loc)))

    (* a.(i) syntax - array access *)
    | Pexp_apply (arr, [(Nolabel, idx)])
      when is_array_access expr ->
      Sarek_ast.EArrGet (parse_expression arr, parse_expression idx)

    (* Binary operators *)
    | Pexp_apply (
        { pexp_desc = Pexp_ident { txt = Lident op; _ }; _ },
        [(Nolabel, e1); (Nolabel, e2)]) ->
      (match parse_binop op with
       | Some binop -> Sarek_ast.EBinop (binop, parse_expression e1, parse_expression e2)
       | None ->
         (* Regular function application with infix *)
         Sarek_ast.EApp (parse_expression { expr with pexp_desc = Pexp_ident { txt = Lident op; loc = expr.pexp_loc } },
               [parse_expression e1; parse_expression e2]))

    (* Unary operators *)
    | Pexp_apply (
        { pexp_desc = Pexp_ident { txt = Lident op; _ }; _ },
        [(Nolabel, e)]) when parse_unop op <> None ->
      (match parse_unop op with
       | Some unop -> Sarek_ast.EUnop (unop, parse_expression e)
       | None -> assert false)

    (* Function application *)
    | Pexp_apply (fn, args) ->
      let fn_expr = parse_expression fn in
      let arg_exprs = List.map (fun (_, e) -> parse_expression e) args in
      Sarek_ast.EApp (fn_expr, arg_exprs)

    (* Let binding *)
    | Pexp_let (Nonrecursive, [{ pvb_pat; pvb_expr; _ }], body) ->
      let name = match extract_name_from_pattern pvb_pat with
        | Some n -> n
        | None -> raise (Parse_error_exn ("Expected variable pattern", pvb_pat.ppat_loc))
      in
      let ty = extract_type_from_pattern pvb_pat in
      let mut_expr =
        match pvb_expr.pexp_desc with
        | Pexp_apply ({ pexp_desc = Pexp_ident { txt = Lident "mut"; _ }; _ },
                      [(Nolabel, inner)]) ->
          Some inner
        | _ -> None
      in
      let is_mutable = Option.is_some mut_expr in
      let value_expr = match mut_expr with
        | Some inner -> inner
        | None -> pvb_expr
      in
      if is_mutable then
        Sarek_ast.ELetMut (name, ty, parse_expression value_expr, parse_expression body)
      else
        Sarek_ast.ELet (name, ty, parse_expression value_expr, parse_expression body)

    (* If-then-else *)
    | Pexp_ifthenelse (cond, then_e, else_opt) ->
      Sarek_ast.EIf (parse_expression cond,
           parse_expression then_e,
           Option.map parse_expression else_opt)

    (* For loop *)
    | Pexp_for ({ ppat_desc = Ppat_var { txt = var; _ }; _ },
                lo, hi, dir, body) ->
      let d = match dir with Upto -> Sarek_ast.Upto | Downto -> Sarek_ast.Downto in
      Sarek_ast.EFor (var, parse_expression lo, parse_expression hi, d, parse_expression body)

    (* While loop *)
    | Pexp_while (cond, body) ->
      Sarek_ast.EWhile (parse_expression cond, parse_expression body)

    (* Sequence *)
    | Pexp_sequence (e1, e2) ->
      Sarek_ast.ESeq (parse_expression e1, parse_expression e2)

    (* Match *)
    | Pexp_match (scrutinee, cases) ->
      let parsed_cases = List.map (fun case ->
          let pat = parse_pattern case.pc_lhs in
          let body = parse_expression case.pc_rhs in
          (pat, body)
        ) cases in
      Sarek_ast.EMatch (parse_expression scrutinee, parsed_cases)

    (* Record construction *)
    | Pexp_record (fields, _base) ->
      let parsed_fields = List.map (fun ({ txt; _ }, e) ->
          let name = match txt with
            | Lident n -> n
            | Ldot (_, n) -> n
            | _ -> "field"
          in
          (name, parse_expression e)
        ) fields in
      Sarek_ast.ERecord (None, parsed_fields)

    (* Field access *)
    | Pexp_field (record, { txt = Lident field; _ }) ->
      Sarek_ast.EFieldGet (parse_expression record, field)

    (* Field set (via setfield) *)
    | Pexp_setfield (record, { txt = Lident field; _ }, value) ->
      Sarek_ast.EFieldSet (parse_expression record, field, parse_expression value)

    (* Constructor application *)
    | Pexp_construct ({ txt = Lident name; _ }, arg_opt) ->
      Sarek_ast.EConstr (name, Option.map parse_expression arg_opt)

    (* Tuple *)
    | Pexp_tuple es ->
      Sarek_ast.ETuple (List.map parse_expression es)

    (* Type annotation *)
    | Pexp_constraint (e, ty) ->
      Sarek_ast.ETyped (parse_expression e, parse_type ty)

    (* Open expression *)
    | Pexp_open ({ popen_expr = { pmod_desc = Pmod_ident { txt; _ }; _ }; _ }, e) ->
      let path = match txt with
        | Lident n -> [n]
        | Ldot (Lident m, n) -> [m; n]
        | _ -> []
      in
      Sarek_ast.EOpen (path, parse_expression e)

    (* Lambda - for local functions in kernels (OCaml 5.2+ uses Pexp_function) *)
    | Pexp_function ([{ pparam_desc = Pparam_val (Nolabel, None, pat); _ }], None, Pfunction_body body) ->
      let name = match extract_name_from_pattern pat with
        | Some n -> n
        | None -> "_"
      in
      let ty = extract_type_from_pattern pat in
      (* Wrap as a let *)
      Sarek_ast.ELet (name, ty, parse_expression body, { Sarek_ast.e = Sarek_ast.EVar name; Sarek_ast.expr_loc = loc })

    (* Multi-parameter lambda *)
    | Pexp_function (params, None, Pfunction_body body) when List.length params > 0 ->
      (* Convert multi-param lambda to nested lets *)
      let rec make_nested_lets params body_expr =
        match params with
        | [] -> parse_expression body_expr
        | { pparam_desc = Pparam_val (Nolabel, None, pat); _ } :: rest ->
          let name = match extract_name_from_pattern pat with
            | Some n -> n
            | None -> "_"
          in
          let ty = extract_type_from_pattern pat in
          let inner = make_nested_lets rest body_expr in
          { Sarek_ast.e = Sarek_ast.ELet (name, ty, inner, { Sarek_ast.e = Sarek_ast.EVar name; Sarek_ast.expr_loc = loc });
            Sarek_ast.expr_loc = loc }
        | _ -> raise (Parse_error_exn ("Unsupported function parameter", expr.pexp_loc))
      in
      (make_nested_lets params body).Sarek_ast.e

    | _ ->
      raise (Parse_error_exn ("Unsupported expression", expr.pexp_loc))
  in
  { Sarek_ast.e; Sarek_ast.expr_loc = loc }

(** Check if an expression is an array access *)
and is_array_access (_expr : expression) : bool =
  (* This is a simplified check - real implementation would need more context *)
  false

(** Extract parameter from pparam_desc *)
let extract_param_from_pparam (pparam : function_param) : Sarek_ast.param =
  match pparam.pparam_desc with
  | Pparam_val (Nolabel, None, pat) ->
    let name = match extract_name_from_pattern pat with
      | Some n -> n
      | None -> raise (Parse_error_exn ("Expected named parameter", pat.ppat_loc))
    in
    let ty = match extract_type_from_pattern pat with
      | Some t -> t
      | None -> raise (Parse_error_exn ("Kernel parameters must have type annotations", pat.ppat_loc))
    in
    {
      Sarek_ast.param_name = name;
      Sarek_ast.param_type = ty;
      Sarek_ast.param_loc = loc_of_ppxlib pat.ppat_loc;
    }
  | Pparam_val (_, _, pat) ->
    raise (Parse_error_exn ("Labelled parameters not supported in kernels", pat.ppat_loc))
  | Pparam_newtype _ ->
    raise (Parse_error_exn ("Newtype parameters not supported in kernels", pparam.pparam_loc))

(** Parse a function expression into a kernel *)
let parse_kernel_function (expr : expression) : Sarek_ast.kernel =
  let loc = loc_of_ppxlib expr.pexp_loc in
  (* OCaml 5.2+ uses Pexp_function with all params in a list *)
  match expr.pexp_desc with
  | Pexp_function (params, _constraint, Pfunction_body body_expr) ->
    if params = [] then
      raise (Parse_error_exn ("Kernel must have at least one parameter", expr.pexp_loc));
    let parsed_params = List.map extract_param_from_pparam params in
    let body = parse_expression body_expr in
    {
      Sarek_ast.kern_name = None;
      Sarek_ast.kern_params = parsed_params;
      Sarek_ast.kern_body = body;
      Sarek_ast.kern_loc = loc;
    }
  | Pexp_function (_, _, Pfunction_cases _) ->
    raise (Parse_error_exn ("Pattern-matching functions not supported as kernels", expr.pexp_loc))
  | _ ->
    raise (Parse_error_exn ("Expected function expression for kernel", expr.pexp_loc))

(** Parse from ppxlib payload *)
let parse_payload (payload : expression) : Sarek_ast.kernel =
  let rec strip_wrappers e =
    match e.pexp_desc with
    | Pexp_letmodule (_name, _mod_expr, body) ->
      strip_wrappers body
    | _ -> e
  in
  parse_kernel_function (strip_wrappers payload)
