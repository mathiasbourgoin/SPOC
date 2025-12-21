(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module defines the typed AST. Every node has a resolved type (never
 * TVar with Unbound). This is the output of type inference.
 ******************************************************************************)

open Sarek_ast
open Sarek_types

(** Typed expression - every node has its type *)
type texpr = {
  te: texpr_desc;
  ty: typ;          (** Always resolved, never contains unbound TVar *)
  te_loc: loc;
}

and texpr_desc =
  (* Literals *)
  | TEUnit
  | TEBool of bool
  | TEInt of int
  | TEInt32 of int32
  | TEInt64 of int64
  | TEFloat of float
  | TEDouble of float

  (* Variables and access *)
  | TEVar of string * int           (** name, variable id *)
  | TEVecGet of texpr * texpr
  | TEVecSet of texpr * texpr * texpr
  | TEArrGet of texpr * texpr
  | TEArrSet of texpr * texpr * texpr
  | TEFieldGet of texpr * string * int  (** expr, field name, field index *)
  | TEFieldSet of texpr * string * int * texpr

  (* Operations - type is in ty field *)
  | TEBinop of binop * texpr * texpr
  | TEUnop of unop * texpr
  | TEApp of texpr * texpr list
  | TEAssign of string * int * texpr  (** name, var_id, value *)

  (* Binding and control *)
  | TELet of string * int * texpr * texpr  (** name, var_id, value, body *)
  | TELetMut of string * int * texpr * texpr
  | TEIf of texpr * texpr * texpr option
  | TEFor of string * int * texpr * texpr * for_dir * texpr  (** var, id, lo, hi, dir, body *)
  | TEWhile of texpr * texpr
  | TESeq of texpr list             (** Flattened sequence *)
  | TEMatch of texpr * (tpattern * texpr) list

  (* Records and variants *)
  | TERecord of string * (string * texpr) list  (** type_name, fields *)
  | TEConstr of string * string * texpr option  (** type_name, constr_name, arg *)
  | TETuple of texpr list

  (* Special forms *)
  | TEReturn of texpr
  | TECreateArray of texpr * typ * memspace
  | TEGlobalRef of string * typ     (** External ref with its type *)
  | TENative of string

  (* Intrinsics *)
  | TEIntrinsicConst of string * string  (** cuda code, opencl code *)
  | TEIntrinsicFun of string * string * texpr list  (** cuda, opencl, args *)

and tpattern = {
  tpat: tpattern_desc;
  tpat_ty: typ;
  tpat_loc: loc;
}

and tpattern_desc =
  | TPAny
  | TPVar of string * int           (** name, var_id *)
  | TPConstr of string * string * tpattern option  (** type_name, constr, arg *)
  | TPTuple of tpattern list

(** Typed kernel parameter *)
type tparam = {
  tparam_name: string;
  tparam_type: typ;
  tparam_index: int;
  tparam_is_vec: bool;              (** Is this a vector parameter? *)
  tparam_id: int;                   (** Variable ID for this parameter *)
}

(** Typed module item *)
type tmodule_item =
  | TMConst of string * int * typ * texpr    (** let name : ty = expr, var id *)
  | TMFun of string * tparam list * texpr    (** let name params = expr *)

(** Typed kernel definition *)
type tkernel = {
  tkern_name: string option;
  tkern_module_items: tmodule_item list;
  tkern_params: tparam list;
  tkern_body: texpr;
  tkern_return_type: typ;
  tkern_loc: loc;
}

(** Variable ID generator *)
let var_id_counter = ref 0

let fresh_var_id () =
  let id = !var_id_counter in
  incr var_id_counter;
  id

let reset_var_id_counter () = var_id_counter := 0

(** Create a simple typed expression *)
let mk_texpr te ty loc = { te; ty; te_loc = loc }

(** Get the fully resolved type (follow all links) *)
let rec resolve_type t =
  match t with
  | TVar { contents = Link t' } -> resolve_type t'
  | TVar { contents = Unbound _ } -> t  (* Should not happen after type inference *)
  | TPrim _ | TVec _ | TArr _ | TRecord _ | TVariant _ | TTuple _ -> t
  | TFun (args, ret) -> TFun (List.map resolve_type args, resolve_type ret)

(** Pretty printing *)
let rec pp_texpr fmt te =
  match te.te with
  | TEUnit -> Format.fprintf fmt "()"
  | TEBool b -> Format.fprintf fmt "%b" b
  | TEInt i -> Format.fprintf fmt "%d" i
  | TEInt32 i -> Format.fprintf fmt "%ldl" i
  | TEInt64 i -> Format.fprintf fmt "%LdL" i
  | TEFloat f -> Format.fprintf fmt "%ff" f
  | TEDouble f -> Format.fprintf fmt "%f" f
  | TEVar (name, id) -> Format.fprintf fmt "%s#%d" name id
  | TEVecGet (v, i) -> Format.fprintf fmt "%a.[%a]" pp_texpr v pp_texpr i
  | TEVecSet (v, i, x) -> Format.fprintf fmt "%a.[%a] <- %a" pp_texpr v pp_texpr i pp_texpr x
  | TEArrGet (a, i) -> Format.fprintf fmt "%a.(%a)" pp_texpr a pp_texpr i
  | TEArrSet (a, i, x) -> Format.fprintf fmt "%a.(%a) <- %a" pp_texpr a pp_texpr i pp_texpr x
  | TEFieldGet (r, f, _) -> Format.fprintf fmt "%a.%s" pp_texpr r f
  | TEFieldSet (r, f, _, x) -> Format.fprintf fmt "%a.%s <- %a" pp_texpr r f pp_texpr x
  | TEBinop (op, a, b) ->
    Format.fprintf fmt "(%a %a %a)" pp_texpr a Sarek_ast.pp_binop op pp_texpr b
  | TEUnop (op, a) ->
    Format.fprintf fmt "(%a %a)" Sarek_ast.pp_unop op pp_texpr a
  | TEApp (f, args) ->
    Format.fprintf fmt "(%a %a)" pp_texpr f
      (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt " ") pp_texpr) args
  | TEAssign (name, id, value) ->
    Format.fprintf fmt "(%s#%d := %a)" name id pp_texpr value
  | TELet (name, id, value, body) ->
    Format.fprintf fmt "(let %s#%d = %a in %a)" name id pp_texpr value pp_texpr body
  | TELetMut (name, id, value, body) ->
    Format.fprintf fmt "(let mut %s#%d = %a in %a)" name id pp_texpr value pp_texpr body
  | TEIf (c, t, None) ->
    Format.fprintf fmt "(if %a then %a)" pp_texpr c pp_texpr t
  | TEIf (c, t, Some e) ->
    Format.fprintf fmt "(if %a then %a else %a)" pp_texpr c pp_texpr t pp_texpr e
  | TEFor (var, id, lo, hi, dir, body) ->
    Format.fprintf fmt "(for %s#%d = %a %s %a do %a done)"
      var id pp_texpr lo
      (match dir with Upto -> "to" | Downto -> "downto")
      pp_texpr hi pp_texpr body
  | TEWhile (c, body) ->
    Format.fprintf fmt "(while %a do %a done)" pp_texpr c pp_texpr body
  | TESeq es ->
    Format.fprintf fmt "(%a)"
      (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt "; ") pp_texpr) es
  | TEMatch (e, cases) ->
    Format.fprintf fmt "(match %a with %a)"
      pp_texpr e
      (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt " | ")
         (fun fmt (p, e) -> Format.fprintf fmt "%a -> %a" pp_tpattern p pp_texpr e))
      cases
  | TERecord (name, fields) ->
    Format.fprintf fmt "%s{%a}" name
      (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt "; ")
         (fun fmt (f, e) -> Format.fprintf fmt "%s = %a" f pp_texpr e))
      fields
  | TEConstr (_, name, None) -> Format.fprintf fmt "%s" name
  | TEConstr (_, name, Some e) -> Format.fprintf fmt "(%s %a)" name pp_texpr e
  | TETuple es ->
    Format.fprintf fmt "(%a)"
      (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ", ") pp_texpr) es
  | TEReturn e -> Format.fprintf fmt "(return %a)" pp_texpr e
  | TECreateArray (size, ty, mem) ->
    Format.fprintf fmt "(create_array %a : %a %a)"
      pp_texpr size pp_typ ty pp_memspace mem
  | TEGlobalRef (name, ty) -> Format.fprintf fmt "@%s : %a" name pp_typ ty
  | TENative s -> Format.fprintf fmt "[%%native %S]" s
  | TEIntrinsicConst (cuda, _) -> Format.fprintf fmt "<intrinsic:%s>" cuda
  | TEIntrinsicFun (cuda, _, args) ->
    Format.fprintf fmt "<%s>(%a)" cuda
      (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ", ") pp_texpr) args

and pp_tpattern fmt p =
  match p.tpat with
  | TPAny -> Format.fprintf fmt "_"
  | TPVar (name, id) -> Format.fprintf fmt "%s#%d" name id
  | TPConstr (_, name, None) -> Format.fprintf fmt "%s" name
  | TPConstr (_, name, Some p) -> Format.fprintf fmt "(%s %a)" name pp_tpattern p
  | TPTuple ps ->
    Format.fprintf fmt "(%a)"
      (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ", ") pp_tpattern) ps

let pp_tparam fmt p =
  Format.fprintf fmt "(%s#%d : %a)" p.tparam_name p.tparam_id pp_typ p.tparam_type

let pp_tkernel fmt k =
  Format.fprintf fmt "kernel %a (%a) : %a = %a"
    (fun fmt -> function None -> Format.fprintf fmt "<anon>" | Some n -> Format.fprintf fmt "%s" n)
    k.tkern_name
    (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ", ") pp_tparam)
    k.tkern_params
    pp_typ k.tkern_return_type
    pp_texpr k.tkern_body
