(** Sarek_ir_types - Pure type definitions for GPU kernel IR

    This module contains only type definitions with no external dependencies.
    Used by spoc_framework for typed generate_source signature. *)

(** Memory spaces *)
type memspace = Global | Shared | Local

(** Element types *)
type elttype =
  | TInt32
  | TInt64
  | TFloat32
  | TFloat64
  | TBool
  | TUnit
  | TRecord of string * (string * elttype) list
      (** Record type: name and field list *)
  | TVariant of string * (string * elttype list) list
      (** Variant type: name and constructor list with arg types *)
  | TArray of elttype * memspace
      (** Array type with element type and memory space *)
  | TVec of elttype  (** Vector (GPU array parameter) *)

(** Variables with type info *)
type var = {
  var_name : string;
  var_id : int;
  var_type : elttype;
  var_mutable : bool;
}

(** Constants *)
type const =
  | CInt32 of int32
  | CInt64 of int64
  | CFloat32 of float
  | CFloat64 of float
  | CBool of bool
  | CUnit

(** Binary operators *)
type binop =
  | Add
  | Sub
  | Mul
  | Div
  | Mod
  | Eq
  | Ne
  | Lt
  | Le
  | Gt
  | Ge
  | And
  | Or
  | Shl
  | Shr
  | BitAnd
  | BitOr
  | BitXor

(** Unary operators *)
type unop = Neg | Not | BitNot

(** Loop direction *)
type for_dir = Upto | Downto

(** Match pattern *)
type pattern =
  | PConstr of string * string list (* Constructor name, bound vars *)
  | PWild

(** Expressions (pure, no side effects) *)
type expr =
  | EConst of const
  | EVar of var
  | EBinop of binop * expr * expr
  | EUnop of unop * expr
  | EArrayRead of string * expr  (** arr[idx] *)
  | EArrayReadExpr of expr * expr  (** base_expr[idx] for complex bases *)
  | ERecordField of expr * string  (** r.field *)
  | EIntrinsic of string list * string * expr list
      (** module path, name, args *)
  | ECast of elttype * expr
  | ETuple of expr list
  | EApp of expr * expr list
  | ERecord of string * (string * expr) list
      (** Record construction: type name, field values *)
  | EVariant of string * string * expr list
      (** Variant construction: type name, constructor, args *)
  | EArrayLen of string  (** Array length intrinsic *)
  | EArrayCreate of elttype * expr * memspace  (** elem type, size, memspace *)
  | EIf of expr * expr * expr  (** condition, then, else - value-returning if *)
  | EMatch of expr * (pattern * expr) list
      (** scrutinee, cases - value-returning match *)

(** L-values (assignable locations) *)
type lvalue =
  | LVar of var
  | LArrayElem of string * expr (* arr[idx] *)
  | LArrayElemExpr of expr * expr (* base_expr[idx] for complex bases *)
  | LRecordField of lvalue * string (* r.field *)

(** Statements (imperative, side effects) *)
type stmt =
  | SAssign of lvalue * expr
  | SSeq of stmt list
  | SIf of expr * stmt * stmt option
  | SWhile of expr * stmt
  | SFor of var * expr * expr * for_dir * stmt
  | SMatch of expr * (pattern * stmt) list
  | SReturn of expr
  | SBarrier  (** Block-level barrier (__syncthreads) *)
  | SWarpBarrier  (** Warp-level sync (__syncwarp) *)
  | SExpr of expr  (** Side-effecting expression *)
  | SEmpty
  | SLet of var * expr * stmt  (** Let binding: let v = e in body *)
  | SLetMut of var * expr * stmt  (** Mutable let: let v = ref e in body *)
  | SPragma of string list * stmt  (** Pragma hints wrapping a statement *)
  | SMemFence  (** Memory fence (threadfence) *)
  | SBlock of stmt
      (** Scoped block - creates a C scope for variable isolation *)
  | SNative of {
      gpu : framework:string -> string;  (** Generate GPU code for framework *)
      ocaml : Obj.t;  (** OCaml fallback (polymorphic via Obj.t) *)
    }  (** Inline native GPU code with OCaml fallback *)

(** Declarations *)
type decl =
  | DParam of
      var * array_info option (* kernel parameter, optional array info *)
  | DLocal of var * expr option (* local variable, optional init *)
  | DShared of
      string * elttype * expr option (* shared array: name, elem type, size *)

and array_info = {arr_elttype : elttype; arr_memspace : memspace}

(** Helper function (device function called from kernel) *)
type helper_func = {
  hf_name : string;
  hf_params : var list;
  hf_ret_type : elttype;
  hf_body : stmt;
}

(** Native argument type for kernel execution. Typed arguments without Obj.t -
    used by PPX-generated native functions. *)
type native_arg =
  | NA_Int32 of int32
  | NA_Int64 of int64
  | NA_Float32 of float
  | NA_Float64 of float
  | NA_Vec of {
      length : int;
      elem_size : int;
      type_name : string;
      get_f32 : int -> float;
      set_f32 : int -> float -> unit;
      get_f64 : int -> float;
      set_f64 : int -> float -> unit;
      get_i32 : int -> int32;
      set_i32 : int -> int32 -> unit;
      get_i64 : int -> int64;
      set_i64 : int -> int64 -> unit;
      (* For custom types (records/variants): type-erased access.
         Internal use only - callers should use vec_get_custom/vec_set_custom. *)
      get_any : int -> Obj.t;
      set_any : int -> Obj.t -> unit;
      (* Get underlying Vector.t for passing to functions/intrinsics *)
      get_vec : unit -> Obj.t;
    }

(** {2 Typed Helpers for Custom Types}

    These functions encapsulate the Obj operations so that PPX-generated code
    doesn't need to use Obj directly. The type parameter is inferred from
    context. *)

(** Get element from NA_Vec as custom type. Type is inferred from usage. *)
let vec_get_custom : type a. native_arg -> int -> a =
 fun arg i ->
  match arg with
  | NA_Vec v -> Obj.magic (v.get_any i)
  | _ -> failwith "vec_get_custom: expected NA_Vec"

(** Set element in NA_Vec from custom type. Type is inferred from usage. *)
let vec_set_custom : type a. native_arg -> int -> a -> unit =
 fun arg i x ->
  match arg with
  | NA_Vec v -> v.set_any i (Obj.magic x)
  | _ -> failwith "vec_set_custom: expected NA_Vec"

(** Get length from NA_Vec *)
let vec_length : native_arg -> int =
 fun arg ->
  match arg with
  | NA_Vec v -> v.length
  | _ -> failwith "vec_length: expected NA_Vec"

(** Get underlying vector. Used when passing vectors to functions/intrinsics
    that need the actual Vector.t type. Returns type-erased value that the
    caller casts to the appropriate Vector.t type. *)
let vec_as_vector : type a. native_arg -> a =
 fun arg ->
  match arg with
  | NA_Vec v -> Obj.magic (v.get_vec ())
  | _ -> failwith "vec_as_vector: expected NA_Vec"

(** Native function type for V2 execution. Uses typed native_arg. *)
type native_fn_t =
  | NativeFn of
      (parallel:bool ->
      block:int * int * int ->
      grid:int * int * int ->
      native_arg array ->
      unit)

(** Kernel representation *)
type kernel = {
  kern_name : string;
  kern_params : decl list;
  kern_locals : decl list;
  kern_body : stmt;
  kern_types : (string * (string * elttype) list) list;
      (** Record type definitions: (type_name, [(field_name, field_type); ...])
      *)
  kern_variants : (string * (string * elttype list) list) list;
      (** Variant type definitions: (type_name,
          [(constructor_name, payload_types); ...]) *)
  kern_funcs : helper_func list;
      (** Helper functions defined in kernel scope *)
  kern_native_fn : native_fn_t option;
      (** Optional pre-compiled native function for CPU execution *)
}

(** {1 Pretty printing} *)

let rec string_of_elttype = function
  | TInt32 -> "int32"
  | TInt64 -> "int64"
  | TFloat32 -> "float32"
  | TFloat64 -> "float64"
  | TBool -> "bool"
  | TUnit -> "unit"
  | TRecord (name, _) -> name
  | TVariant (name, _) -> name
  | TArray (elt, ms) ->
      Printf.sprintf "%s %s[]" (string_of_memspace ms) (string_of_elttype elt)
  | TVec elt -> Printf.sprintf "%s vector" (string_of_elttype elt)

and string_of_memspace = function
  | Global -> "global"
  | Shared -> "shared"
  | Local -> "local"

let string_of_binop = function
  | Add -> "+"
  | Sub -> "-"
  | Mul -> "*"
  | Div -> "/"
  | Mod -> "%"
  | Eq -> "=="
  | Ne -> "!="
  | Lt -> "<"
  | Le -> "<="
  | Gt -> ">"
  | Ge -> ">="
  | And -> "&&"
  | Or -> "||"
  | Shl -> "<<"
  | Shr -> ">>"
  | BitAnd -> "&"
  | BitOr -> "|"
  | BitXor -> "^"

let string_of_unop = function Neg -> "-" | Not -> "!" | BitNot -> "~"

let pp_elttype fmt ty = Format.fprintf fmt "%s" (string_of_elttype ty)

let pp_memspace fmt ms = Format.fprintf fmt "%s" (string_of_memspace ms)

let pp_var fmt v = Format.fprintf fmt "%s" v.var_name

let rec pp_expr fmt = function
  | EConst (CInt32 n) -> Format.fprintf fmt "%ld" n
  | EConst (CInt64 n) -> Format.fprintf fmt "%LdL" n
  | EConst (CFloat32 f) -> Format.fprintf fmt "%gf" f
  | EConst (CFloat64 f) -> Format.fprintf fmt "%g" f
  | EConst (CBool b) -> Format.fprintf fmt "%b" b
  | EConst CUnit -> Format.fprintf fmt "()"
  | EVar v -> pp_var fmt v
  | EBinop (op, e1, e2) ->
      Format.fprintf fmt "(%a %s %a)" pp_expr e1 (string_of_binop op) pp_expr e2
  | EUnop (op, e) -> Format.fprintf fmt "(%s%a)" (string_of_unop op) pp_expr e
  | EArrayRead (arr, idx) -> Format.fprintf fmt "%s[%a]" arr pp_expr idx
  | ERecordField (e, field) -> Format.fprintf fmt "%a.%s" pp_expr e field
  | EIntrinsic (path, name, args) ->
      let full_name = String.concat "." (path @ [name]) in
      if args = [] then Format.fprintf fmt "%s" full_name
      else Format.fprintf fmt "%s(%a)" full_name pp_exprs args
  | ECast (ty, e) ->
      Format.fprintf fmt "(%s)%a" (string_of_elttype ty) pp_expr e
  | ETuple exprs -> Format.fprintf fmt "(%a)" pp_exprs exprs
  | EApp (fn, args) -> Format.fprintf fmt "%a(%a)" pp_expr fn pp_exprs args
  | ERecord (name, fields) ->
      Format.fprintf fmt "%s{" name ;
      List.iteri
        (fun i (f, e) ->
          if i > 0 then Format.fprintf fmt ", " ;
          Format.fprintf fmt "%s = %a" f pp_expr e)
        fields ;
      Format.fprintf fmt "}"
  | EVariant (_, constr, []) -> Format.fprintf fmt "%s" constr
  | EVariant (_, constr, args) ->
      Format.fprintf fmt "%s(%a)" constr pp_exprs args
  | EArrayLen arr -> Format.fprintf fmt "len(%s)" arr
  | EArrayCreate (ty, size, mem) ->
      Format.fprintf
        fmt
        "create_array<%a,%a>[%a]"
        pp_elttype
        ty
        pp_memspace
        mem
        pp_expr
        size
  | EArrayReadExpr (base, idx) ->
      Format.fprintf fmt "(%a)[%a]" pp_expr base pp_expr idx
  | EIf (cond, then_, else_) ->
      Format.fprintf
        fmt
        "(%a ? %a : %a)"
        pp_expr
        cond
        pp_expr
        then_
        pp_expr
        else_
  | EMatch (e, cases) ->
      Format.fprintf fmt "match %a { " pp_expr e ;
      List.iter
        (fun (_, body) -> Format.fprintf fmt "_ => %a; " pp_expr body)
        cases ;
      Format.fprintf fmt "}"

and pp_exprs fmt = function
  | [] -> ()
  | [e] -> pp_expr fmt e
  | e :: es -> Format.fprintf fmt "%a, %a" pp_expr e pp_exprs es

let rec pp_lvalue fmt = function
  | LVar v -> pp_var fmt v
  | LArrayElem (arr, idx) -> Format.fprintf fmt "%s[%a]" arr pp_expr idx
  | LArrayElemExpr (base, idx) ->
      Format.fprintf fmt "(%a)[%a]" pp_expr base pp_expr idx
  | LRecordField (lv, field) -> Format.fprintf fmt "%a.%s" pp_lvalue lv field

let rec pp_stmt fmt = function
  | SAssign (lv, e) -> Format.fprintf fmt "%a = %a;" pp_lvalue lv pp_expr e
  | SSeq stmts -> List.iter (fun s -> Format.fprintf fmt "%a@," pp_stmt s) stmts
  | SIf (cond, s1, s2) -> (
      Format.fprintf fmt "@[<v 2>if (%a) {@ %a@]@ }" pp_expr cond pp_stmt s1 ;
      match s2 with
      | None -> ()
      | Some s -> Format.fprintf fmt " else {@ %a@ }" pp_stmt s)
  | SWhile (cond, body) ->
      Format.fprintf
        fmt
        "@[<v 2>while (%a) {@ %a@]@ }"
        pp_expr
        cond
        pp_stmt
        body
  | SFor (v, start, stop, dir, body) ->
      let op = match dir with Upto -> "<" | Downto -> ">" in
      let inc = match dir with Upto -> "++" | Downto -> "--" in
      Format.fprintf
        fmt
        "@[<v 2>for (%s = %a; %s %s %a; %s%s) {@ %a@]@ }"
        v.var_name
        pp_expr
        start
        v.var_name
        op
        pp_expr
        stop
        v.var_name
        inc
        pp_stmt
        body
  | SMatch (e, cases) ->
      Format.fprintf fmt "@[<v 2>match %a with" pp_expr e ;
      List.iter
        (fun (p, s) -> Format.fprintf fmt "@ | %a -> %a" pp_pattern p pp_stmt s)
        cases ;
      Format.fprintf fmt "@]"
  | SReturn e -> Format.fprintf fmt "return %a;" pp_expr e
  | SBarrier -> Format.fprintf fmt "__syncthreads();"
  | SWarpBarrier -> Format.fprintf fmt "__syncwarp();"
  | SExpr e -> Format.fprintf fmt "%a;" pp_expr e
  | SEmpty -> ()
  | SLet (v, e, body) ->
      Format.fprintf
        fmt
        "@[<v 2>let %s = %a in@ %a@]"
        v.var_name
        pp_expr
        e
        pp_stmt
        body
  | SLetMut (v, e, body) ->
      Format.fprintf
        fmt
        "@[<v 2>let %s = ref %a in@ %a@]"
        v.var_name
        pp_expr
        e
        pp_stmt
        body
  | SPragma (hints, body) ->
      Format.fprintf fmt "#pragma %s@," (String.concat " " hints) ;
      pp_stmt fmt body
  | SMemFence -> Format.fprintf fmt "__threadfence();"
  | SBlock body -> Format.fprintf fmt "@[<v 2>{@ %a@]@ }" pp_stmt body
  | SNative _ -> Format.fprintf fmt "/* native code */"

and pp_pattern fmt = function
  | PConstr (name, vars) ->
      if vars = [] then Format.fprintf fmt "%s" name
      else Format.fprintf fmt "%s(%s)" name (String.concat ", " vars)
  | PWild -> Format.fprintf fmt "_"

let pp_decl fmt = function
  | DParam (v, None) ->
      Format.fprintf fmt "%s %s" (string_of_elttype v.var_type) v.var_name
  | DParam (v, Some arr) ->
      Format.fprintf
        fmt
        "%s %s* %s"
        (string_of_memspace arr.arr_memspace)
        (string_of_elttype arr.arr_elttype)
        v.var_name
  | DLocal (v, None) ->
      Format.fprintf fmt "%s %s;" (string_of_elttype v.var_type) v.var_name
  | DLocal (v, Some e) ->
      Format.fprintf
        fmt
        "%s %s = %a;"
        (string_of_elttype v.var_type)
        v.var_name
        pp_expr
        e
  | DShared (name, ty, None) ->
      Format.fprintf fmt "__shared__ %s %s[];" (string_of_elttype ty) name
  | DShared (name, ty, Some size) ->
      Format.fprintf
        fmt
        "__shared__ %s %s[%a];"
        (string_of_elttype ty)
        name
        pp_expr
        size

let pp_kernel fmt k =
  Format.fprintf fmt "@[<v>__kernel void %s(" k.kern_name ;
  (match k.kern_params with
  | [] -> ()
  | [p] -> pp_decl fmt p
  | p :: ps ->
      pp_decl fmt p ;
      List.iter (fun p -> Format.fprintf fmt ", %a" pp_decl p) ps) ;
  Format.fprintf fmt ") {@," ;
  List.iter (fun d -> Format.fprintf fmt "  %a@," pp_decl d) k.kern_locals ;
  Format.fprintf fmt "  %a@," pp_stmt k.kern_body ;
  Format.fprintf fmt "}@]"

let print_kernel k = Format.fprintf Format.std_formatter "%a@." pp_kernel k

(** {1 Float64 detection} *)

(** Check if an element type is or contains float64 *)
let rec elttype_uses_float64 = function
  | TFloat64 -> true
  | TRecord (_, fields) ->
      List.exists (fun (_, t) -> elttype_uses_float64 t) fields
  | TVariant (_, constrs) ->
      List.exists
        (fun (_, args) -> List.exists elttype_uses_float64 args)
        constrs
  | TArray (elt, _) | TVec elt -> elttype_uses_float64 elt
  | TInt32 | TInt64 | TFloat32 | TBool | TUnit -> false

(** Check if a constant is float64 *)
let const_uses_float64 = function CFloat64 _ -> true | _ -> false

(** Check if an expression uses float64 *)
let rec expr_uses_float64 = function
  | EConst c -> const_uses_float64 c
  | EVar v -> elttype_uses_float64 v.var_type
  | EBinop (_, e1, e2) -> expr_uses_float64 e1 || expr_uses_float64 e2
  | EUnop (_, e) -> expr_uses_float64 e
  | EArrayRead (_, idx) -> expr_uses_float64 idx
  | EArrayReadExpr (base, idx) ->
      expr_uses_float64 base || expr_uses_float64 idx
  | ERecordField (e, _) -> expr_uses_float64 e
  | EIntrinsic (_, _, args) -> List.exists expr_uses_float64 args
  | ECast (ty, e) -> elttype_uses_float64 ty || expr_uses_float64 e
  | ETuple exprs -> List.exists expr_uses_float64 exprs
  | EApp (fn, args) ->
      expr_uses_float64 fn || List.exists expr_uses_float64 args
  | ERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_float64 e) fields
  | EVariant (_, _, args) -> List.exists expr_uses_float64 args
  | EArrayLen _ -> false
  | EArrayCreate (ty, size, _) ->
      elttype_uses_float64 ty || expr_uses_float64 size
  | EIf (cond, then_, else_) ->
      expr_uses_float64 cond || expr_uses_float64 then_
      || expr_uses_float64 else_
  | EMatch (scrutinee, cases) ->
      expr_uses_float64 scrutinee
      || List.exists (fun (_, e) -> expr_uses_float64 e) cases

(** Check if a statement uses float64 *)
let rec stmt_uses_float64 = function
  | SAssign (_, e) -> expr_uses_float64 e
  | SSeq stmts -> List.exists stmt_uses_float64 stmts
  | SIf (cond, then_, else_) ->
      expr_uses_float64 cond || stmt_uses_float64 then_
      || Option.fold ~none:false ~some:stmt_uses_float64 else_
  | SWhile (cond, body) -> expr_uses_float64 cond || stmt_uses_float64 body
  | SFor (v, lo, hi, _, body) ->
      elttype_uses_float64 v.var_type
      || expr_uses_float64 lo || expr_uses_float64 hi || stmt_uses_float64 body
  | SMatch (scrutinee, cases) ->
      expr_uses_float64 scrutinee
      || List.exists (fun (_, s) -> stmt_uses_float64 s) cases
  | SReturn e | SExpr e -> expr_uses_float64 e
  | SBarrier | SWarpBarrier | SEmpty | SMemFence -> false
  | SLet (v, e, body) | SLetMut (v, e, body) ->
      elttype_uses_float64 v.var_type
      || expr_uses_float64 e || stmt_uses_float64 body
  | SPragma (_, body) | SBlock body -> stmt_uses_float64 body
  | SNative _ -> false

(** Check if a declaration uses float64 *)
let decl_uses_float64 = function
  | DParam (v, arr_info) ->
      elttype_uses_float64 v.var_type
      || Option.fold
           ~none:false
           ~some:(fun ai -> elttype_uses_float64 ai.arr_elttype)
           arr_info
  | DLocal (v, init) ->
      elttype_uses_float64 v.var_type
      || Option.fold ~none:false ~some:expr_uses_float64 init
  | DShared (_, ty, size) ->
      elttype_uses_float64 ty
      || Option.fold ~none:false ~some:expr_uses_float64 size

(** Check if a helper function uses float64 *)
let helper_uses_float64 hf =
  elttype_uses_float64 hf.hf_ret_type
  || List.exists (fun v -> elttype_uses_float64 v.var_type) hf.hf_params
  || stmt_uses_float64 hf.hf_body

(** Check if a kernel uses float64 anywhere *)
let kernel_uses_float64 k =
  List.exists decl_uses_float64 k.kern_params
  || List.exists decl_uses_float64 k.kern_locals
  || stmt_uses_float64 k.kern_body
  || List.exists helper_uses_float64 k.kern_funcs
  || List.exists
       (fun (_, fields) ->
         List.exists (fun (_, t) -> elttype_uses_float64 t) fields)
       k.kern_types
  || List.exists
       (fun (_, constrs) ->
         List.exists
           (fun (_, args) -> List.exists elttype_uses_float64 args)
           constrs)
       k.kern_variants
