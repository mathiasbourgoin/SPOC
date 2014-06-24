open Camlp4.PreCast
open Syntax
open Ast

let retype = ref false
let unknown = ref 0
let debug = false

let my_eprintf s = 
  if debug then
    (output_string stderr s;
     flush stderr;)
  else ()



type customtypes =
  | KRecord of ctyp list * ident list
  | KSum of int list

type memspace = 
  | Local
  | Global
  | Shared
  | Any

type ktyp =
  | TUnknown
  | TUnit
  | TInt32
  | TInt64
  | TFloat32
  | TFloat64
  | TBool
  | TVec of ktyp
  | TArr of (ktyp*memspace)
  | TApp of ktyp * ktyp


let rec ktyp_to_string = function
  | TUnit -> "unit"
  (* | TInt -> "int" *)
  | TInt32  -> "int32"
  | TInt64  -> "int64"
  (* | TFloat  ->  "float" *)
  | TFloat32 -> "float32"
  | TFloat64 -> "float64"
  | TUnknown  -> "unknown"
  | TBool -> "bool"
  | TVec k -> (ktyp_to_string k)^" vector" 
  | TArr (k,m) -> (ktyp_to_string k)^" array" 
  | TApp (k1,k2) -> (ktyp_to_string k1)^" -> "^(ktyp_to_string k2) 



let ex32 = 
  let _loc = Loc.ghost in
  <:expr< ExFloat32>>
let ex64 = 	let _loc = Loc.ghost in
  <:expr< ExFloat64>>

let extensions = 
  ref [ ex32 ] 

type var = {
  n : int;
  mutable var_type: ktyp;
  mutable is_mutable : bool;
  mutable read_only : bool;
  mutable write_only : bool;
  mutable is_global : bool;
}

let new_kernel = ref false

exception ArgumentError
exception Unbound_value of string * Loc.t
exception Unbound_module of string * Loc.t
exception Immutable of string * Loc.t
exception TypeError of ktyp * ktyp * Loc.t



type k_expr = 
  | Open of Loc.t * ident * kexpr
  | App of Loc.t * kexpr * kexpr list
  | Acc of Loc.t * kexpr * kexpr
  | VecSet of Loc.t*kexpr*kexpr
  | VecGet of Loc.t*kexpr*kexpr
  | ArrSet of Loc.t*kexpr*kexpr
  | ArrGet of Loc.t*kexpr*kexpr
  | Seq of Loc.t*kexpr*kexpr
  | Bind of Loc.t*kexpr*kexpr*kexpr*bool
(*  | Plus of Loc.t*kexpr*kexpr*)
  | Plus32 of Loc.t*kexpr*kexpr
  | Plus64 of Loc.t*kexpr*kexpr
  | PlusF of Loc.t*kexpr*kexpr
  | PlusF32 of Loc.t*kexpr*kexpr
  | PlusF64 of Loc.t*kexpr*kexpr

(*  | Min of Loc.t*kexpr*kexpr*)
  | Min32 of Loc.t*kexpr*kexpr
  | Min64 of Loc.t*kexpr*kexpr
  | MinF of Loc.t*kexpr*kexpr
  | MinF32 of Loc.t*kexpr*kexpr
  | MinF64 of Loc.t*kexpr*kexpr

(*  | Mul of Loc.t*kexpr*kexpr*)
  | Mul32 of Loc.t*kexpr*kexpr
  | Mul64 of Loc.t*kexpr*kexpr
  | MulF of Loc.t*kexpr*kexpr
  | MulF32 of Loc.t*kexpr*kexpr
  | MulF64 of Loc.t*kexpr*kexpr

  | Mod of Loc.t*kexpr*kexpr

(*  | Div of Loc.t*kexpr*kexpr*)
  | Div32 of Loc.t*kexpr*kexpr
  | Div64 of Loc.t*kexpr*kexpr
  | DivF of Loc.t*kexpr*kexpr
  | DivF32 of Loc.t*kexpr*kexpr
  | DivF64 of Loc.t*kexpr*kexpr

  | Id of Loc.t*ident
  | Int of Loc.t*string
  | Int32 of Loc.t*string
  | Int64 of Loc.t*string
  | Float of Loc.t*string
  | Float32 of Loc.t*string
  | Float64 of Loc.t*string
  | CastId of ktyp* k_expr
  | BoolAnd of Loc.t*kexpr*kexpr
  | BoolOr of Loc.t*kexpr*kexpr
  | BoolEq32 of Loc.t*kexpr*kexpr
  | BoolEq64 of Loc.t*kexpr*kexpr
  | BoolEqF of Loc.t*kexpr*kexpr
  | BoolEqF64 of Loc.t*kexpr*kexpr
  | BoolLt of Loc.t*kexpr*kexpr
  | BoolLt32 of Loc.t*kexpr*kexpr
  | BoolLt64 of Loc.t*kexpr*kexpr
  | BoolLtF of Loc.t*kexpr*kexpr
  | BoolLtF64 of Loc.t*kexpr*kexpr

  | BoolLtE of Loc.t*kexpr*kexpr
  | BoolLtE32 of Loc.t*kexpr*kexpr
  | BoolLtE64 of Loc.t*kexpr*kexpr
  | BoolLtEF of Loc.t*kexpr*kexpr
  | BoolLtEF64 of Loc.t*kexpr*kexpr

  | BoolGt of Loc.t*kexpr*kexpr
  | BoolGt32 of Loc.t*kexpr*kexpr
  | BoolGt64 of Loc.t*kexpr*kexpr
  | BoolGtF of Loc.t*kexpr*kexpr
  | BoolGtF64 of Loc.t*kexpr*kexpr

  | BoolGtE of Loc.t*kexpr*kexpr
  | BoolGtE32 of Loc.t*kexpr*kexpr
  | BoolGtE64 of Loc.t*kexpr*kexpr
  | BoolGtEF of Loc.t*kexpr*kexpr
  | BoolGtEF64 of Loc.t*kexpr*kexpr

  | Ife of Loc.t * kexpr * kexpr * kexpr
  | If of Loc.t * kexpr * kexpr 
  | DoLoop of Loc.t * kexpr * kexpr * kexpr * kexpr
  | While of Loc.t * kexpr * kexpr
  | End of Loc.t*kexpr
  | Ref of Loc.t*kexpr

  | ModuleAccess of Loc.t * string * kexpr

  | Noop

and kexpr = {
  mutable t : ktyp;
  mutable e: k_expr;
  loc: Loc.t}

let is_unknown t =
  match t with
  | TUnknown
  | TVec TUnknown
  | TArr (TUnknown, _) -> true
  | _ -> false


let update_type value t =
  match t,value.t with
  | TUnknown, t -> ()
  | _ , TUnknown -> value.t <- t
  | TVec TUnknown, TVec _ -> ()
  | TArr (TUnknown,_), TArr _ -> ()
  | TVec t, TVec TUnknown -> value.t <- t
  | TArr (t,_), TArr (TUnknown, _) -> value.t <- t
  | t1, t2 ->
    (*if t1 <> t2 then
      ( assert (not debug); raise (TypeError (t, value.t, value.loc)) *)
    if not (is_unknown t) then
      if not (is_unknown value.t)  && value.t <> t then
        ( assert (not debug); raise (TypeError (t, value.t, value.loc)) )
      else
        (
          if value.t <> t then
            (value.t <- t;
             retype := true)
        )
      
let rec string_of_ident i = 
  let aux = function
    | <:ident< $lid:s$ >> -> s
    | <:ident< $uid:s$ >> -> s
    | <:ident< $i1$.$i2$ >> -> "" ^ (string_of_ident i1) ^ "." ^ (string_of_ident i2)
    | <:ident< $i1$($i2$) >> -> "" ^ (string_of_ident i1) ^ " " ^ (string_of_ident i2)
    | _ -> assert false
  in aux i


let rec k_expr_to_string = function
  | App _ -> "App"
  | Acc _ -> "Acc"
  | VecSet _ -> "VecSet"
  | VecGet _ -> "VecGet"
  | ArrSet _ -> "ArrSet"
  | ArrGet _ -> "ArrGet"
  | Seq _ -> "Seq"
  | Bind _ -> "Bind"

(*  | Plus _ -> "Plus"*)
  | Plus32 _ -> "Plus32"
  | Plus64 _ -> "Plus64"
  | PlusF _ -> "PlusF"
  | PlusF32 _ -> "PlusF32"
  | PlusF64 _ -> "PlusF64"
(*  | Min _ -> "Min"*)
  | Min32 _ -> "Min32"
  | Min64 _ -> "Min64"
  | MinF _ -> "MinF"
  | MinF32 _ -> "MinF32"
  | MinF64 _ -> "MinF64"
(*  | Mul _ -> "Mul"*)
  | Mul32 _ -> "Mul32"
  | Mul64 _ -> "Mul64"
  | MulF _ -> "MulF"
  | MulF32 _ -> "MulF32"
  | MulF64 _ -> "MulF64"
(*  | Div _ -> "Div"*)
  | Div32 _ -> "Div32"
  | Div64 _ -> "Div64"
  | DivF _ -> "DivF"
  | DivF32 _ -> "DivF32"
  | DivF64 _ -> "DivF64"
  | Mod _ -> "Mod"
  | Id (l,s) -> ("Id "^(string_of_ident s))
  | Int _ -> "Int"
  | Int32 _ -> "Int32"
  | Int64 _ -> "Int64"
  | Float _ -> "Float"
  | Float32 _ -> "Float32"
  | Float64 _ -> "Float64"
  | CastId _ -> "CastId"
  | BoolEq32 _ -> "BoolEq32"
  | BoolEq64 _ -> "BoolEq64"
  | BoolEqF _ -> "BoolEqF"
  | BoolEqF64 _ -> "BoolEqF64"
  | BoolLt _ -> "BoolLt"
  | BoolLt32 _ -> "BoolLt32"
  | BoolLt64 _ -> "BoolLt64"
  | BoolLtF _ -> "BoolLtF"
  | BoolLtF64 _ -> "BoolEqF64"
  | BoolGt _ -> "BoolGt"
  | BoolGt32 _ -> "BoolGt32"
  | BoolGt64 _ -> "BoolGt64"
  | BoolGtF _ -> "BoolGtF"
  | BoolGtF64 _ -> "BoolGtF64"
  | BoolLtE _ -> "BoolLtE"
  | BoolLtE32 _ -> "BoolLtE32"
  | BoolLtE64 _ -> "BoolLtE64"
  | BoolLtEF _ -> "BoolLtEF"
  | BoolLtEF64 _ -> "BoolLtEF64"
  | BoolGtE _ -> "BoolGtE"
  | BoolGtE32 _ -> "BoolGtE32"
  | BoolGtE64 _ -> "BoolGtE64"
  | BoolGtEF _ -> "BoolGtEF"
  | BoolGtEF64 _ -> "BoolGtEF64"

  | BoolAnd _ -> "BoolAnd"
  | BoolOr _ -> "BoolOr"
  | Ife _ -> "Ife"
  | If _ -> "If"
  | DoLoop _ -> "DoLoop"
  | While _ -> "While"
  | End _ -> "End"
  | Open _ -> "Open"
  | Noop -> "Noop"
  | ModuleAccess _ -> "ModuleAccess"
  | Ref _ -> "Ref"



let expr_of_patt p =
  match p with
  | PaId (l,i) -> ExId (l,i)
  | _ -> raise ArgumentError

type k_patt 


type cfun = 
  { nb_args: int;
    cuda_val : string;
    opencl_val : string;
    typ : ktyp 
  }

type spoc_module = {
  mod_name : string;
  mod_constants : (ktyp * string * string * string) list;
  mod_functions : (ktyp * string * int * string * string) list;
  mod_modules : (string, spoc_module) Hashtbl.t
} 


let return_type = ref TUnknown

let arg_idx = ref 0

let args () =
  let (tbl : (string, var) Hashtbl.t)  = Hashtbl.create 10 in
  tbl 

let current_args = ref (args ()) 

let intrinsics_fun = ref ((Hashtbl.create 100):(string,cfun) Hashtbl.t)
let intrinsics_const = ref ((Hashtbl.create 100):(string,cfun) Hashtbl.t)

let (arg_list : Camlp4.PreCast.Syntax.Ast.expr list ref ) = ref []


let std = {
  mod_name = "Std";
  mod_constants = 
    [
      (TInt32, "thread_idx_x", "threadIdx.x", "(get_local_id (0))");
      (TInt32, "thread_idx_y", "threadIdx.y", "(get_local_id (1))");
      (TInt32, "thread_idx_z", "threadIdx.z", "(get_local_id (2))");

      (TInt32, "block_idx_x", "blockIdx.x", "(get_group_id (0))");
      (TInt32, "block_idx_y", "blockIdx.y", "(get_group_id (1))");
      (TInt32, "block_idx_z", "blockIdx.z", "(get_group_id (2))");

      (TInt32, "block_dim_x", "blockDim.x", "(get_local_size (0))");
      (TInt32, "block_dim_y", "blockDim.y", "(get_local_size (1))");
      (TInt32, "block_dim_z", "blockDim.z", "(get_local_size (2))");

      (TInt32, "grid_dim_x", "gridDim.x", "(get_num_groups (0))");
      (TInt32, "grid_dim_y", "gridDim.y", "(get_num_groups (1))");
      (TInt32, "grid_dim_z", "gridDim.z", "(get_num_groups (2))");

      (TInt32, "global_thread_id", "blockIdx.x*blockDim.x+threadIdx.x", 
       "get_global_id (0)");

      (TFloat64, "zero64", "0.", "0.");
      (TFloat64, "one64", "1.", "1.");
    ];
  mod_functions = [
    (TApp (TUnit, TUnit), "return", 1, "return", "return");

    (TApp (TInt32, TFloat32), "float_of_int", 1, "(float)", "(float)");
    (TApp (TInt32, TFloat32), "float", 1, "(float)", "(float)");

    (TApp (TInt32, TFloat64), "float64_of_int", 1, "(double)", "(double)");
    (TApp (TInt32, TFloat64), "float64", 1, "(double)", "(double)");
    (TApp (TFloat32, TFloat64), "float64_of_float", 1, "(double)", "(double)");

    (TApp (TFloat32, TInt32), "int_of_float", 1, "(int)", "(int)");
    (TApp (TFloat64, TInt32), "int_of_float64", 1, "(int)", "(int)");
    (TApp (TUnit, TUnit), "block_barrier", 1, "__syncthreads ", "");

    (TApp (TInt32, TArr (TInt32, Shared)), "make_shared", 1, "", "");
    (TApp (TInt32, TArr (TInt32, Local)), "make_local", 1, "", "");
  ];
  mod_modules = 
    let m = Hashtbl.create 2 in 
    m
}

let mathf32 = {
  mod_name = "Float32";
  mod_constants = 
    [
      (TFloat32, "zero", "0.f", "0.f");
      (TFloat32, "one", "1.f", "1.f");
    ];
  mod_functions = [
    (TApp ((TApp (TFloat32, TFloat32)), TFloat32), "add", 2, "spoc_fadd", "spoc_fadd");
    (TApp ((TApp (TFloat32, TFloat32)), TFloat32), "minus", 2, "spoc_fminus", "spoc_fminus");
    (TApp ((TApp (TFloat32, TFloat32)), TFloat32), "mul", 2, "spoc_fmul", "spoc_fmul");
    (TApp ((TApp (TFloat32, TFloat32)), TFloat32), "div", 2, "spoc_fdiv", "spoc_fdiv");

    (TApp ((TApp (TFloat32, TFloat32)), TFloat32), "pow", 2, "powf", "pow");
    (TApp (TFloat32, TFloat32), "sqrt", 1, "sqrtf", "sqrt");
    (TApp (TFloat32, TFloat32), "exp", 1, "expf", "exp");
    (TApp (TFloat32, TFloat32), "log", 1, "logf", "log");
    (TApp (TFloat32, TFloat32), "log10", 1, "log10f", "log10");
    (TApp (TFloat32, TFloat32), "expm1", 1, "expm1f", "expm1");
    (TApp (TFloat32, TFloat32), "log1p", 1, "log1pf", "log1p");


    (TApp (TFloat32, TFloat32), "acos", 1, "acosf", "acos");
    (TApp (TFloat32, TFloat32), "cos", 1, "cosf", "cos");
    (TApp (TFloat32, TFloat32), "cosh", 1, "coshf", "cosh");
    (TApp (TFloat32, TFloat32), "asin", 1, "asinf", "asin");
    (TApp (TFloat32, TFloat32), "sin", 1, "sinf", "sin");
    (TApp (TFloat32, TFloat32), "sinh", 1, "sinhf", "sinh");
    (TApp (TFloat32, TFloat32), "tan", 1, "tanf", "tan");
    (TApp (TFloat32, TFloat32), "tanh", 1, "tanhf", "tanh");
    (TApp (TFloat32, TFloat32), "atan", 1, "atanf", "atan");
    (TApp ((TApp (TFloat32, TFloat32)), TFloat32), "atan2", 2, "atan2f", "atan2");
    (TApp ((TApp (TFloat32, TFloat32)), TFloat32), "hypot", 2, "hypotf", "hypot");


    (TApp (TFloat32, TFloat32), "ceil", 1, "ceilf", "ceil");
    (TApp (TFloat32, TFloat32), "floor", 1, "floorf", "floor");

    (TApp (TFloat32, TFloat32), "abs_float", 1, "fabsf", "fabs");

    (TApp ((TApp (TFloat32, TFloat32)), TFloat32), "copysign", 2, "copysignf", "copysign");
    (TApp ((TApp (TFloat32, TFloat32)), TFloat32), "modf", 2, "fmodf", "fmod");

(*    (TApp (TFloat, TFloat32), "of_float", 1, "(float)", "(float)");
    (TApp (TFloat32, TFloat), "to_float", 1, "(float)", "(float)");
*)
    (TApp (TInt32, TArr (TFloat32, Shared)), "make_shared", 1, "", "");
    (TApp (TInt32, TArr (TFloat32, Local)), "make_local", 1, "", "");

  ];
  mod_modules = Hashtbl.create 0
}

let mathf64 = {
  mod_name = "Float64";
  mod_constants = 
    [
      (TFloat64, "zero", "0.", "0.");
      (TFloat64, "one", "1.", "1.");
    ];
  mod_functions = [
    (TApp ((TApp (TFloat64, TFloat64)), TFloat64), "add", 2, "spoc_dadd", "spoc_dadd");
    (TApp ((TApp (TFloat64, TFloat64)), TFloat64), "minus", 2, "spoc_dminus", "spoc_dminus");
    (TApp ((TApp (TFloat64, TFloat64)), TFloat64), "mul", 2, "spoc_dmul", "spoc_dmul");
    (TApp ((TApp (TFloat64, TFloat64)), TFloat64), "div", 2, "spoc_ddiv", "spoc_ddiv");

    (TApp ((TApp (TFloat64, TFloat64)), TFloat64), "pow", 2, "powf", "pow");
    (TApp (TFloat64, TFloat64), "sqrt", 1, "sqrt", "sqrt");
    (TApp (TFloat64, TFloat64), "exp", 1, "exp", "exp");
    (TApp (TFloat64, TFloat64), "log", 1, "log", "log");
    (TApp (TFloat64, TFloat64), "log10", 1, "log10", "log10");
    (TApp (TFloat64, TFloat64), "expm1", 1, "expm1", "expm1");
    (TApp (TFloat64, TFloat64), "log1p", 1, "log1p", "log1p");


    (TApp (TFloat64, TFloat64), "acos", 1, "acos", "acos");
    (TApp (TFloat64, TFloat64), "cos", 1, "cos", "cos");
    (TApp (TFloat64, TFloat64), "cosh", 1, "cosh", "cosh");
    (TApp (TFloat64, TFloat64), "asin", 1, "asin", "asin");
    (TApp (TFloat64, TFloat64), "sin", 1, "sin", "sin");
    (TApp (TFloat64, TFloat64), "sinh", 1, "sinh", "sinh");
    (TApp (TFloat64, TFloat64), "tan", 1, "tan", "tan");
    (TApp (TFloat64, TFloat64), "tanh", 1, "tanh", "tanh");
    (TApp (TFloat64, TFloat64), "atan", 1, "atan", "atan");
    (TApp ((TApp (TFloat64, TFloat64)), TFloat64), "atan2", 2, "atan2", "atan2");
    (TApp ((TApp (TFloat64, TFloat64)), TFloat64), "hypot", 2, "hypot", "hypot");


    (TApp (TFloat64, TFloat64), "ceil", 1, "ceil", "ceil");
    (TApp (TFloat64, TFloat64), "floor", 1, "floor", "floor");

    (TApp (TFloat64, TFloat64), "abs_float", 1, "fabs", "fabs");

    (TApp ((TApp (TFloat64, TFloat64)), TFloat64), "copysign", 2, "copysign", "copysign");
    (TApp ((TApp (TFloat64, TFloat64)), TFloat64), "modf", 2, "fmod", "fmod");

     (TApp (TFloat32, TFloat64), "of_float32", 1, "(double)", "(double)"); 
     (TApp (TFloat64, TFloat32), "to_float32", 1, "(double)", "(double)"); 

    (TApp (TInt32, TArr (TFloat64, Shared)), "make_shared", 1, "", "");
    (TApp (TInt32, TArr (TFloat64, Local)), "make_local", 1, "", "");


  ];
  mod_modules = Hashtbl.create 0
}


let math = {
  mod_name = "Math";
  mod_constants = 
    [
    ];
  mod_functions = [
    (TApp ((TApp (TInt32, TInt32)), TInt32), "pow", 2, "spoc_powint", "spoc_powint");
    (TApp ((TApp (TInt32, TInt32)), TInt32), "logical_and", 2, "logical_and", "logical_and");
    (TApp ((TApp (TInt32, TInt32)), TInt32), "xor", 2, "spoc_xor", "spoc_xor");
  ];
  mod_modules = 
    let m = Hashtbl.create 0 in
    Hashtbl.add m mathf32.mod_name mathf32; 
    Hashtbl.add m mathf64.mod_name mathf64;
    m
}


let modules = 
  let m = Hashtbl.create 10 in
  Hashtbl.add m std.mod_name std;
  Hashtbl.add m math.mod_name math; 
  m



let open_module  m_ident  _loc = 
  my_eprintf (Printf.sprintf "opening module %s\n%!" m_ident);

  (match m_ident with
   | "Float64" -> 
     if not (List.mem ex64 !extensions)  then 
       (
         extensions := ex64:: !extensions;
         Printf.eprintf "%s\n%!" ("\027[32m Warning \027[00m : kernel uses"^
                                  "\027[34m double precision floating point \027[00m"^
                                  "extension, make sure your device is compatible")
       )
   | _ -> ());
  let m =
    try Hashtbl.find modules (m_ident)
    with
    | _ -> assert (not debug); raise (Unbound_module (m_ident, _loc))
  in
  List.iter (fun (typ,s,nb_args,cuda_s, opencl_s) ->
      (Hashtbl.add !intrinsics_fun (s) {nb_args=2; cuda_val = cuda_s; opencl_val = opencl_s; typ=typ})) m.mod_functions;
  List.iter (fun (typ,s,cuda_s, opencl_s) ->
      (Hashtbl.add !intrinsics_const (s) {nb_args=0; cuda_val = cuda_s; opencl_val = opencl_s; typ=typ})) m.mod_constants ;
  Hashtbl.iter (fun name intern_m-> Hashtbl.add modules name intern_m) m.mod_modules 

and close_module m_ident = 
  my_eprintf (Printf.sprintf "closing module %s\n%!" m_ident);
  try
    let m =
      Hashtbl.find modules (m_ident)    
    in
    List.iter (fun (typ,s,nb_args,cuda_s, opencl_s) ->
        (Hashtbl.remove !intrinsics_fun (s))) m.mod_functions;
    List.iter (fun (typ,s,cuda_s, opencl_s) ->
        (Hashtbl.remove !intrinsics_const (s))) m.mod_constants;
    Hashtbl.iter (fun name intern_m-> Hashtbl.remove modules name) m.mod_modules 
  with
  | _ -> () 

let rec basic_check l expected_type current_type loc =
  if expected_type <> current_type && expected_type <> TUnknown then
    ( assert (not debug); raise (TypeError (expected_type, current_type, loc)) );
  List.iter (fun e -> typer e expected_type)

and elt_check body t l =
  if body.t <> t && body.t <> TUnknown then
    (assert (not debug); raise (TypeError (t, body.t, l)) );
  update_type body  t;

and equal_types t1 t2 =
  if t1 = t2 then
    true
  else
    match t1,t2 with
    | TArr (t1_, _), TArr (t2_, _) ->
      equal_types  t1_ t2_
    | _ -> false

and check t1 t2 l=
  if not (equal_types t1 t2) &&( not (is_unknown t1) || not (is_unknown t2)) then
    (assert (not debug); raise (TypeError (t1, t2, l)) )
    
and typer body t =
  my_eprintf (Printf.sprintf"(* %s >>>>>>>>>>>> typ %s *)\n%!" (k_expr_to_string body.e) (ktyp_to_string t)) ;  
  (match body.e with
  | Id (l, s) ->
    let tt = ref t in
    (try
       let var = Hashtbl.find !current_args (string_of_ident s) in
       my_eprintf ((string_of_ident s)^ " of type " ^(ktyp_to_string t)^"\n");
       if not (is_unknown t) then 
         if is_unknown var.var_type  then
           (var.var_type <- t;
            update_type body t)
         else
           check var.var_type t l;
       tt := var.var_type
     with Not_found ->
       try 
         let c_const = Hashtbl.find !intrinsics_const (string_of_ident s) in
         if t = TUnknown then
           ( update_type body c_const.typ;)
         else if t <> c_const.typ then
           (assert (not debug); raise (TypeError (c_const.typ, t, l)))
       with _ -> 
         try ignore(Hashtbl.find !intrinsics_fun (string_of_ident s) )
         with _ ->
           (Hashtbl.add !current_args (string_of_ident s) 
              {n = -1; var_type = t;
               is_mutable = false;
               read_only = true;
               write_only = false;
               is_global = true;};
            tt := t;)
    );
    update_type body !tt
  | ArrSet (l, e1, e2) ->
    check t TUnit l;
    typer e1 e2.t;
    typer e2 e1.t;
    update_type body TUnit
  | ArrGet (l, e1, e2) ->
    typer e1 (TArr (t, Any));
    typer e2 TInt32;
    update_type body (
      match e1.t with
      | TArr (tt,_) -> tt
      | _ -> my_eprintf (ktyp_to_string e1.t); 
        assert false;)
  | VecSet (l, e1, e2) ->
    (*check t TUnit l;*)
    typer e1 e2.t;
    typer e2 e1.t;
    update_type body TUnit
  | VecGet (l, e1, e2) ->
    typer e1 (TVec t);
    typer e2 TInt32;
    update_type body (
      match e1.t with
      | TVec tt -> tt
      | _ -> TVec TUnknown)
      (*my_eprintf ((k_expr_to_string e1.e) ^" ++ "^(ktyp_to_string e1.t)^"\n"); 
        assert false;)*)
  | Seq (l, e1, e2) ->
    typer e1 TUnit;
    typer e2 t;
    update_type body e2.t
  | Int32 (l,s) -> 
    elt_check body TInt32 l
  | Int64 (l,s) -> 
    elt_check body TInt64 l
  | Float32 (l,s) -> 
    elt_check body TFloat32 l
  | Float64 (l,s) -> 
    elt_check body TFloat64 l
  | Bind (_loc, var,y, z, is_mutable)  ->
    typer y TUnknown;
    (match var.e with
     | Id (_loc,s)  ->
       (incr arg_idx;
        Hashtbl.add !current_args (string_of_ident s) 
          {n = !arg_idx; var_type = y.t;
           is_mutable = is_mutable;
           read_only = false;
           write_only = false;
           is_global = false;}
       )
    );
    update_type var y.t;
(*    update_type body y.t;*)
    typer z t;
    update_type body z.t
  | Plus32 (l,e1,e2) | Min32 (l,e1,e2) 
  | Mul32 (l,e1,e2) | Div32 (l,e1,e2) 
  | Mod (l, e1, e2) -> 
    basic_check [e1;e2] t TInt32 l;
    update_type body TInt32;
  | PlusF32 (l,e1,e2) | MinF32 (l,e1,e2) | MulF32 (l,e1,e2) | DivF32 (l,e1,e2) ->
    basic_check [e1;e2] t TFloat32 l;
    update_type body TFloat32;
  | If (l, e1, e2) ->
    typer e1 TBool;
    basic_check [e1] e1.t TBool l;
    typer e2 TUnit;
    update_type body TUnit
  | Ife (l, e1, e2, e3) ->
    typer e1 TBool;
    basic_check [e1] e1.t TBool l;
    typer e2 e3.t;
    typer e3 e2.t;
    if e2.t <> e3.t then
      ( assert (not debug); raise (TypeError (e2.t, e3.t, l)) );
    update_type body e2.t
  | Noop ->
    if t <> TUnit && t <> TUnknown then
      assert false
    else
      update_type body TUnit
  | Open (l, m_ident, e2) ->
    let rec _open = function
      | IdAcc (l,a,b) -> _open a; _open b
      | IdUid (l,s) -> open_module s l
      | _ -> assert false
    and _close = function
      | IdAcc (l,a,b) -> _close a; _close b
      | IdUid (l,s) -> close_module s
      | _ -> assert false
    in
    _open m_ident;
    typer e2 t;
    _close m_ident;
    update_type body e2.t;
  | While (l, cond, loop_body) ->
    typer cond TBool;
    basic_check [cond] cond.t TBool;
    basic_check [loop_body] t TUnit;
    typer cond TBool;
    typer loop_body TUnit;
    body.t <- TUnit;
  | App (l, e1, e2) -> 
    let t = typer_app e1 e2 t in
    update_type body t
  | BoolEq32 (l, e1, e2) 
  | BoolLt32 (l, e1, e2) 
  | BoolLtE32 (l, e1, e2) 
  | BoolGt32 (l, e1, e2) 
  | BoolGtE32 (l, e1, e2) -> 
    typer e1 TInt32;
    typer e2 TInt32;
    update_type body TBool
  | BoolEqF (l, e1, e2) 
  | BoolLtF (l, e1, e2) 
  | BoolLtEF (l, e1, e2) 
  | BoolGtF (l, e1, e2) 
  | BoolGtEF (l, e1, e2) -> 
    typer e1 TFloat32;
    typer e2 TFloat32;
    update_type body TBool
  | BoolOr (l,e1,e2) 
  | BoolAnd (l,e1,e2) ->
    typer e1 TBool;
    typer e2 TBool;
    update_type body TBool
  | Ref (l, e) ->
    typer e t;
    update_type body e.t;
  | Acc (l, e1, e2) ->
    typer e2 e1.t;
    typer e1 e2.t;
    update_type body TUnit
  | _ -> my_eprintf  ((k_expr_to_string body.e)^"\n"); assert false);
  if is_unknown body.t then
    incr unknown
           
and typer_app e1 (e2 : kexpr list) t =
  let  typ, loc  = 
    let rec aux e1 =
      match e1.e with
      | Id (_l, s) -> (try (Hashtbl.find !intrinsics_fun (string_of_ident s)).typ , _l
                       with |_ -> 
                         typer e1 t; e1.t, _l); 
      | ModuleAccess (_l, s, e) ->
        open_module s _l;
        let typ, loc = aux e in
        close_module s;
        typ, loc
      | _ ->  typer e1 t; e1.t, e1.loc
      | _ ->  assert false 
    in 
    aux e1
  in
  let ret = ref TUnit in
  let rec aux2 typ expr =
    match typ, expr with
    | TApp (t1, t2), e::[] -> typer e t1; ret := t2
    | _ , [] -> assert false
  in		
  let rec aux typ1 e =
    match typ1,e with
    | (TApp (t1, (TApp (_,_) as t2)), 
       App ( l , e1, (t::(tt::qq as q) as e2))) ->
      aux2 t2 e2;
      typer e1 t1;
      if e1.t <> t1  then ( assert (not debug); raise (TypeError (t1, e1.t, l)) );
      update_type e1  t1;
    | ((TApp (TApp(t1, t3) as t, t2)), 
       (App (l, e1, e2::[]))) ->
      assert (t3 = t2);
      ret := t2;
      typer e2 t1;			
      if e2.t <> t1 && e2.t <> TUnknown then ( assert (not debug); raise (TypeError (t1, e2.t, l)) );
      update_type e2 t1;
      update_type e1 t;
    | (TApp(t1, t2), App(_, _, e2::[] ) )->   
      my_eprintf (Printf.sprintf"(* typ %s +++-> %s *)\n%!" (k_expr_to_string e2.e) (ktyp_to_string e2.t)) ; 
      ret := e2.t;
      typer e2 t1;
    | (t1 , App(_, _, e2::[] ) )->   
      my_eprintf (Printf.sprintf"(* typ %s +++-> %s *)\n%!" (k_expr_to_string e2.e) (ktyp_to_string e2.t)) ; 
      ret := t1;
      typer e2 t1;
    | _ -> assert (not debug);
  in aux typ (App (loc,e1,e2));
  let rec aux typ =
    match typ with
    | TApp(t1,t2) -> t2
    | t -> t
  in 
  my_eprintf (Printf.sprintf"(* >>>>>>>>>>>> typ %s *)\n%!" (ktyp_to_string typ)) ; 
  let t = aux typ  in
  my_eprintf (Printf.sprintf"(* <<<<<<<<<<<< typ %s *)\n%!" (ktyp_to_string t)) ;
  t
  
