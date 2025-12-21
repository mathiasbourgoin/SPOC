(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module defines the type representation used throughout compilation.
 * Types are framework-independent and support unification for type inference.
 ******************************************************************************)

(** Primitive types supported in GPU kernels *)
type prim_type = TUnit | TBool | TInt32 | TInt64 | TFloat32 | TFloat64

(** Memory spaces *)
type memspace =
  | Local  (** Thread-private memory *)
  | Shared  (** Block-shared memory *)
  | Global  (** Global device memory *)

(** Types *)
type typ =
  | TPrim of prim_type  (** Primitive types *)
  | TVar of tvar ref  (** Unification variable *)
  | TVec of typ  (** Vector type (GPU array parameter) *)
  | TArr of typ * memspace  (** Local array with memory space *)
  | TFun of typ list * typ  (** Function type *)
  | TRecord of string * (string * typ) list  (** Record type: name, fields *)
  | TVariant of string * (string * typ option) list
      (** Variant type: name, constructors *)
  | TTuple of typ list  (** Tuple type *)

and tvar =
  | Unbound of int * int  (** id, level for generalization *)
  | Link of typ  (** Resolved to this type *)

(** Generate fresh type variable IDs *)
let tvar_counter = ref 0

let fresh_tvar_id () =
  let id = !tvar_counter in
  incr tvar_counter ;
  id

(** Create a fresh unbound type variable at given level *)
let fresh_tvar ?(level = 0) () : typ =
  TVar (ref (Unbound (fresh_tvar_id (), level)))

(** Reset the type variable counter (for testing) *)
let reset_tvar_counter () = tvar_counter := 0

(** Follow links to get the actual type *)
let rec repr (t : typ) : typ =
  match t with TVar {contents = Link t'} -> repr t' | t -> t

(** Check if a type variable occurs in a type (for occurs check) *)
let rec occurs (id : int) (t : typ) : bool =
  match repr t with
  | TVar {contents = Unbound (id', _)} -> id = id'
  | TVar {contents = Link _} ->
      assert false (* repr should have followed links *)
  | TPrim _ -> false
  | TVec t -> occurs id t
  | TArr (t, _) -> occurs id t
  | TFun (args, ret) -> List.exists (occurs id) args || occurs id ret
  | TRecord (_, fields) -> List.exists (fun (_, t) -> occurs id t) fields
  | TVariant (_, constrs) ->
      List.exists
        (function _, None -> false | _, Some t -> occurs id t)
        constrs
  | TTuple ts -> List.exists (occurs id) ts

(** Unification error *)
type unify_error = Cannot_unify of typ * typ | Occurs_check of int * typ

(** Unify two types *)
let rec unify (t1 : typ) (t2 : typ) : (unit, unify_error) result =
  let t1 = repr t1 and t2 = repr t2 in
  if t1 == t2 then Ok ()
  else
    match (t1, t2) with
    | TVar {contents = Unbound (id1, _)}, TVar {contents = Unbound (id2, _)}
      when id1 = id2 ->
        Ok () (* Same variable *)
    | TVar ({contents = Unbound (id, level1)} as r), t
    | t, TVar ({contents = Unbound (id, level1)} as r) ->
        if occurs id t then Error (Occurs_check (id, t))
        else begin
          (* Update level for let-polymorphism *)
          (match t with
          | TVar {contents = Unbound (_, level2)} ->
              r := Unbound (id, min level1 level2)
          | _ -> ()) ;
          r := Link t ;
          Ok ()
        end
    | TPrim p1, TPrim p2 when p1 = p2 -> Ok ()
    | TVec t1, TVec t2 -> unify t1 t2
    | TArr (t1, m1), TArr (t2, m2) when m1 = m2 -> unify t1 t2
    | TFun (args1, ret1), TFun (args2, ret2) ->
        if List.length args1 <> List.length args2 then
          Error (Cannot_unify (t1, t2))
        else begin
          let rec unify_args = function
            | [], [] -> Ok ()
            | a1 :: rest1, a2 :: rest2 -> (
                match unify a1 a2 with
                | Ok () -> unify_args (rest1, rest2)
                | Error e -> Error e)
            | _ -> Error (Cannot_unify (t1, t2))
          in
          match unify_args (args1, args2) with
          | Ok () -> unify ret1 ret2
          | Error e -> Error e
        end
    | TRecord (n1, fields1), TRecord (n2, fields2)
      when n1 = n2 || n1 = "anon_record" || n2 = "anon_record" ->
        if List.length fields1 <> List.length fields2 then
          Error (Cannot_unify (t1, t2))
        else
          let rec unify_fields = function
            | [], [] -> Ok ()
            | (f1, t1) :: rest1, (f2, t2) :: rest2 when f1 = f2 -> (
                match unify t1 t2 with
                | Ok () -> unify_fields (rest1, rest2)
                | Error e -> Error e)
            | _ -> Error (Cannot_unify (t1, t2))
          in
          unify_fields (fields1, fields2)
    | TVariant (n1, _), TVariant (n2, _) when n1 = n2 ->
        Ok () (* Variants with same name are considered equal *)
    | TTuple ts1, TTuple ts2 ->
        if List.length ts1 <> List.length ts2 then Error (Cannot_unify (t1, t2))
        else
          let rec unify_elems = function
            | [], [] -> Ok ()
            | t1 :: rest1, t2 :: rest2 -> (
                match unify t1 t2 with
                | Ok () -> unify_elems (rest1, rest2)
                | Error e -> Error e)
            | _ -> Error (Cannot_unify (t1, t2))
          in
          unify_elems (ts1, ts2)
    | _, _ -> Error (Cannot_unify (t1, t2))

(** Pretty printing *)
let pp_prim fmt = function
  | TUnit -> Format.fprintf fmt "unit"
  | TBool -> Format.fprintf fmt "bool"
  | TInt32 -> Format.fprintf fmt "int32"
  | TInt64 -> Format.fprintf fmt "int64"
  | TFloat32 -> Format.fprintf fmt "float32"
  | TFloat64 -> Format.fprintf fmt "float64"

let pp_memspace fmt = function
  | Local -> Format.fprintf fmt "local"
  | Shared -> Format.fprintf fmt "shared"
  | Global -> Format.fprintf fmt "global"

let rec pp_typ fmt t =
  match repr t with
  | TPrim p -> pp_prim fmt p
  | TVar {contents = Unbound (id, level)} ->
      Format.fprintf fmt "'t%d[%d]" id level
  | TVar {contents = Link t} -> pp_typ fmt t
  | TVec t -> Format.fprintf fmt "%a vector" pp_typ t
  | TArr (t, m) -> Format.fprintf fmt "%a array[%a]" pp_typ t pp_memspace m
  | TFun (args, ret) ->
      Format.fprintf
        fmt
        "(%a) -> %a"
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt " * ")
           pp_typ)
        args
        pp_typ
        ret
  | TRecord (name, fields) ->
      Format.fprintf
        fmt
        "%s{%a}"
        name
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt "; ")
           (fun fmt (f, t) -> Format.fprintf fmt "%s: %a" f pp_typ t))
        fields
  | TVariant (name, _) -> Format.fprintf fmt "%s" name
  | TTuple ts ->
      Format.fprintf
        fmt
        "(%a)"
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt " * ")
           pp_typ)
        ts

let typ_to_string t = Format.asprintf "%a" pp_typ t

(** Helpers for common types *)
let t_unit = TPrim TUnit

let t_bool = TPrim TBool

let t_int32 = TPrim TInt32

let t_int64 = TPrim TInt64

let t_float32 = TPrim TFloat32

let t_float64 = TPrim TFloat64

let t_vec t = TVec t

let t_arr t m = TArr (t, m)

let t_fun args ret = TFun (args, ret)

(** Check if type is numeric *)
let is_numeric t =
  match repr t with
  | TPrim (TInt32 | TInt64 | TFloat32 | TFloat64) -> true
  | _ -> false

(** Check if type is integer *)
let is_integer t =
  match repr t with TPrim (TInt32 | TInt64) -> true | _ -> false

(** Check if type is floating point *)
let is_float t =
  match repr t with TPrim (TFloat32 | TFloat64) -> true | _ -> false

(** Convert AST type expression to type (with fresh type variables) *)
let rec type_of_type_expr (te : Sarek_ast.type_expr) : typ =
  match te with
  | Sarek_ast.TEVar _ -> fresh_tvar ()
  | Sarek_ast.TEConstr ("unit", []) -> t_unit
  | Sarek_ast.TEConstr ("bool", []) -> t_bool
  | Sarek_ast.TEConstr ("int", []) -> t_int32
  | Sarek_ast.TEConstr ("int32", []) -> t_int32
  | Sarek_ast.TEConstr ("int64", []) -> t_int64
  | Sarek_ast.TEConstr ("float", []) -> t_float32
  | Sarek_ast.TEConstr ("float32", []) -> t_float32
  | Sarek_ast.TEConstr ("float64", []) -> t_float64
  | Sarek_ast.TEConstr ("double", []) -> t_float64
  | Sarek_ast.TEConstr ("vector", [elem]) -> TVec (type_of_type_expr elem)
  | Sarek_ast.TEConstr (name, [elem])
    when String.ends_with ~suffix:"vector" name ->
      (* Handle "float32 vector" style *)
      TVec (type_of_type_expr elem)
  | Sarek_ast.TEConstr ("array", [elem]) -> TArr (type_of_type_expr elem, Local)
  | Sarek_ast.TEConstr (name, args) ->
      (* Custom type - we'll need to look it up in environment *)
      let _ = List.map type_of_type_expr args in
      TRecord (name, [])
      (* Placeholder - will be resolved by typer *)
  | Sarek_ast.TEArrow (a, b) -> TFun ([type_of_type_expr a], type_of_type_expr b)
  | Sarek_ast.TETuple ts -> TTuple (List.map type_of_type_expr ts)

(** Convert memspace from AST to types *)
let memspace_of_ast (m : Sarek_ast.memspace) : memspace =
  match m with
  | Sarek_ast.Local -> Local
  | Sarek_ast.Shared -> Shared
  | Sarek_ast.Global -> Global
