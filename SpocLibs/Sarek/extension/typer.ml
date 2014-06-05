open Camlp4.PreCast
open Syntax
open Ast


let debug = false


let my_eprintf s = 
  if debug then
    output_string stderr s
  else ()



type customtypes =
  | KRecord of ctyp list * ident list
  | KSum of int list

type ktyp =
  | TUnknown
  | TUnit
  | TInt
  | TInt32
  | TInt64
  | TFloat
  | TFloat32
  | TFloat64
  | TBool
  | TVec of ktyp
  | TArr of ktyp
  | TApp of ktyp * ktyp


let rec ktyp_to_string = function
  | TUnit -> "unit"
  | TInt -> "int"
  | TInt32  -> "int32"
  | TInt64  -> "int64"
  | TFloat  ->  "float"
  | TFloat32 -> "float32"
  | TFloat64 -> "float64"
  | TUnknown  -> "unknown"
  | TBool -> "bool"
  | TVec k -> (ktyp_to_string k)^" vector" 
  | TArr k -> (ktyp_to_string k)^" array" 
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
  | Seqs of kexpr list
  | Bind of Loc.t*kexpr*kexpr*kexpr*bool
  | Plus of Loc.t*kexpr*kexpr
  | Plus32 of Loc.t*kexpr*kexpr
  | Plus64 of Loc.t*kexpr*kexpr
  | PlusF of Loc.t*kexpr*kexpr
  | PlusF32 of Loc.t*kexpr*kexpr
  | PlusF64 of Loc.t*kexpr*kexpr

  | Min of Loc.t*kexpr*kexpr
  | Min32 of Loc.t*kexpr*kexpr
  | Min64 of Loc.t*kexpr*kexpr
  | MinF of Loc.t*kexpr*kexpr
  | MinF32 of Loc.t*kexpr*kexpr
  | MinF64 of Loc.t*kexpr*kexpr

  | Mul of Loc.t*kexpr*kexpr
  | Mul32 of Loc.t*kexpr*kexpr
  | Mul64 of Loc.t*kexpr*kexpr
  | MulF of Loc.t*kexpr*kexpr
  | MulF32 of Loc.t*kexpr*kexpr
  | MulF64 of Loc.t*kexpr*kexpr

  | Mod of Loc.t*kexpr*kexpr

  | Div of Loc.t*kexpr*kexpr
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
  | BoolEq of Loc.t*kexpr*kexpr
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

let update_type value t =
  if t <> TUnknown && t <> TVec TUnknown && t <> TArr TUnknown then
    value.t <- t

let rec k_expr_to_string = function
  | App _ -> "App"
  | Acc _ -> "Acc"
  | VecSet _ -> "VecSet"
  | VecGet _ -> "VecGet"
  | ArrSet _ -> "ArrSet"
  | ArrGet _ -> "ArrGet"
  | Seq _ -> "Seq"
  | Bind _ -> "Bind"

  | Plus _ -> "Plus"
  | Plus32 _ -> "Plus32"
  | Plus64 _ -> "Plus64"
  | PlusF _ -> "PlusF"
  | PlusF32 _ -> "PlusF32"
  | PlusF64 _ -> "PlusF64"
  | Min _ -> "Min"
  | Min32 _ -> "Min32"
  | Min64 _ -> "Min64"
  | MinF _ -> "MinF"
  | MinF32 _ -> "MinF32"
  | MinF64 _ -> "MinF64"
  | Mul _ -> "Mul"
  | Mul32 _ -> "Mul32"
  | Mul64 _ -> "Mul64"
  | MulF _ -> "MulF"
  | MulF32 _ -> "MulF32"
  | MulF64 _ -> "MulF64"
  | Div _ -> "Div"
  | Div32 _ -> "Div32"
  | Div64 _ -> "Div64"
  | DivF _ -> "DivF"
  | DivF32 _ -> "DivF32"
  | DivF64 _ -> "DivF64"
  | Mod _ -> "Mod"
  | Id _ -> "Id"
  | Int _ -> "Int"
  | Int32 _ -> "Int32"
  | Int64 _ -> "Int64"
  | Float _ -> "Float"
  | Float32 _ -> "Float32"
  | Float64 _ -> "Float64"
  | CastId _ -> "CastId"
  | BoolEq _ -> "BoolEq"
  | BoolEq32 _ -> "BoolEq32"
  | BoolEq64 _ -> "BoolEq64"
  | BoolEqF _ -> "BoolEqF"
  | BoolEqF64 _ -> "BoolEqF64"
  | BoolLt _ -> "BoolEq"
  | BoolLt32 _ -> "BoolEq32"
  | BoolLt64 _ -> "BoolEq64"
  | BoolLtF _ -> "BoolEqF"
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
  | Seqs _ -> "Seqs"
  | Open _ -> "Open"
  | Noop -> "Noop"
  | ModuleAccess _ -> "ModuleAccess"
  | Ref _ -> "Ref"


let rec string_of_ident i = 
  let aux = function
    | <:ident< $lid:s$ >> -> s
    | <:ident< $uid:s$ >> -> s
    | <:ident< $i1$.$i2$ >> -> "" ^ (string_of_ident i1) ^ "." ^ (string_of_ident i2)
    | <:ident< $i1$($i2$) >> -> "" ^ (string_of_ident i1) ^ " " ^ (string_of_ident i2)
    | _ -> assert false
  in aux i

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
  let tbl = Hashtbl.create 10 in
  tbl
let current_args = ref (args ()) 

let intrinsics_fun = ref (Hashtbl.create 100)
let intrinsics_const = ref (Hashtbl.create 100)

let (arg_list : Camlp4.PreCast.Syntax.Ast.expr list ref ) = ref []


let std = {
  mod_name = "Std";
  mod_constants = 
    [
      (TInt, "thread_idx_x", "threadIdx.x", "(get_local_id (0))");
      (TInt, "thread_idx_y", "threadIdx.y", "(get_local_id (1))");
      (TInt, "thread_idx_z", "threadIdx.z", "(get_local_id (2))");

      (TInt, "block_idx_x", "blockIdx.x", "(get_group_id (0))");
      (TInt, "block_idx_y", "blockIdx.y", "(get_group_id (1))");
      (TInt, "block_idx_z", "blockIdx.z", "(get_group_id (2))");

      (TInt, "block_dim_x", "blockDim.x", "(get_local_size (0))");
      (TInt, "block_dim_y", "blockDim.y", "(get_local_size (1))");
      (TInt, "block_dim_z", "blockDim.z", "(get_local_size (2))");

      (TInt, "grid_dim_x", "gridDim.x", "(get_num_groups (0))");
      (TInt, "grid_dim_y", "gridDim.y", "(get_num_groups (1))");
      (TInt, "grid_dim_z", "gridDim.z", "(get_num_groups (2))");

      (TInt, "global_thread_id", "blockIdx.x*blockDim.x+threadIdx.x", 
       "get_global_id (0)");

      (TFloat64, "zero64", "0.", "0.");
      (TFloat64, "one64", "1.", "1.");
    ];
  mod_functions = [
    (TApp (TUnit, TUnit), "return", 1, "return", "return");

    (TApp (TInt, TFloat32), "float_of_int", 1, "(float)", "(float)");
    (TApp (TInt, TFloat32), "float", 1, "(float)", "(float)");

    (TApp (TInt, TFloat64), "float64_of_int", 1, "(double)", "(double)");
    (TApp (TInt, TFloat64), "float64", 1, "(double)", "(double)");
    (TApp (TFloat, TFloat64), "float64_of_float", 1, "(double)", "(double)");

    (TApp (TFloat32, TInt), "int_of_float", 1, "(int)", "(int)");
    (TApp (TFloat64, TInt), "int_of_float64", 1, "(int)", "(int)");
    (TApp (TUnit, TUnit), "block_barrier", 1, "__syncthreads ", "");

    (TApp (TInt32, TArr TInt32), "make_shared", 1, "", "");
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

    (TApp (TFloat, TFloat32), "of_float", 1, "(float)", "(float)");
    (TApp (TFloat32, TFloat), "to_float", 1, "(float)", "(float)");

    (TApp (TInt32, TArr TFloat32), "make_shared", 1, "", "");


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

    (TApp (TFloat, TFloat64), "of_float", 1, "(double)", "(double)");
    (TApp (TFloat64, TFloat), "to_float", 1, "(double)", "(double)");

    (TApp (TInt32, TArr TFloat64), "make_shared", 1, "", "");


  ];
  mod_modules = Hashtbl.create 0
}


let math = {
  mod_name = "Math";
  mod_constants = 
    [
    ];
  mod_functions = [
    (TApp ((TApp (TInt, TInt)), TInt), "pow", 2, "spoc_powint", "spoc_powint");
    (TApp ((TApp (TInt, TInt)), TInt), "logical_and", 2, "logical_and", "logical_and");
    (TApp ((TApp (TInt, TInt)), TInt), "xor", 2, "spoc_xor", "spoc_xor");
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



let rec  typer_app e1 (e2 : kexpr list) t =
  let  typ, loc  = 
    let rec aux e1 =
      match e1.e with
      | Id (_l, s) -> (try (Hashtbl.find !intrinsics_fun (string_of_ident s)).typ , _l
                       with |_ -> 
                         typer e1 t; e1.t, _l); 
      | CastId (typ, Id (_l, s)) -> 
        (	let var  = Hashtbl.find !intrinsics_fun (string_of_ident s)
          in
          var.typ , _l 
        )
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
  my_eprintf (Printf.sprintf"(* >>>>>>>>>>>> typ %s *)\n%!" (ktyp_to_string t)) ;
  t

and typer_ body t = 
    
  match body.e with
  | Bind (_loc, var,y, z, is_mutable)  ->
    (
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
       | CastId (tt, Id(_loc,s)) ->
         let ty = match y.t with 
           | TFloat -> TFloat64
           | _ -> y.t in
         (incr arg_idx;
          Hashtbl.add !current_args (string_of_ident s) 
            {n = !arg_idx; var_type = ty;
             is_mutable = is_mutable;
             read_only = false;
             write_only = false;
             is_global=false;})
       | _ -> (assert false)
      );
      update_type var y.t;
      update_type body y.t;
      typer z t;
    )   
  | Plus (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt -> ()
     | _ -> assert false);
    (typer a TInt; 
     typer b TInt;
     update_type body TInt;)
  | Plus32 (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt32 -> ()
     | _ -> assert false);
    (typer a TInt32; 
     typer b TInt32;
     update_type body TInt32;)
  | Plus64 (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt64 -> ()
     | _ -> assert false);    (typer a TInt64; 
                               typer b TInt64;
                               update_type body TInt64;)
  | PlusF (_loc, a,b) -> 
    (match t with
     | TUnknown | TFloat32 -> ()
     | _ -> assert false);    
    (typer a TFloat32; 
     typer b TFloat32;
     update_type body TFloat32;)
  | PlusF32 (_loc, a,b) -> 
    (match t with
     | TUnknown | TFloat32 -> ()
     | _ -> assert false);
    (typer a TFloat32; 
     typer b TFloat32;
     update_type body TFloat32;)
  | PlusF64 (_loc, a,b) ->
    (match t with
     | TUnknown | TFloat64 -> ()
     | _ -> assert false); 
    (typer a TFloat64; 
     typer b TFloat64;
     update_type body TFloat64;) 

  | Min (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt -> ()
     | _ -> assert false);
    (typer a TInt; 
     typer b TInt;
     update_type body TInt;)
  | Min32 (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt32 -> ()
     | _ -> assert false);
    (typer a TInt32; 
     typer b TInt32;
     update_type body TInt32;)
  | Min64 (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt64 -> ()
     | _ -> assert false);    (typer a TInt64; 
                               typer b TInt64;
                               update_type body TInt64;)
  | MinF (_loc, a,b) -> 
    (match t with
     | TUnknown | TFloat32 -> ()
     | _ -> assert false);    
    (typer a TFloat32; 
     typer b TFloat32;
     update_type body TFloat32;)
  | MinF32 (_loc, a,b) -> 
    (match t with
     | TUnknown | TFloat32 -> ()
     | _ -> assert false);
    (typer a TFloat32; 
     typer b TFloat32;
     update_type body TFloat32;)
  | MinF64 (_loc, a,b) ->
    (match t with
     | TUnknown | TFloat64 -> ()
     | _ -> assert false); 
    (typer a TFloat64; 
     typer b TFloat64;
     update_type body TFloat64;) 

  | Mul (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt -> ()
     | _ -> assert false);
    (typer a TInt; 
     typer b TInt;
     update_type body TInt;)
  | Mul32 (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt32 -> ()
     | _ -> assert false);
    (typer a TInt32; 
     typer b TInt32;
     update_type body TInt32;)
  | Mul64 (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt64 -> ()
     | _ -> assert false);    (typer a TInt64; 
                               typer b TInt64;
                               update_type body TInt64;)
  | MulF (_loc, a,b) -> 
    (match t with
     | TUnknown | TFloat32 -> ()
     | _ -> assert false);    
    (typer a TFloat32; 
     typer b TFloat32;
     update_type body TFloat32;)
  | MulF32 (_loc, a,b) -> 
    (match t with
     | TUnknown | TFloat32 -> ()
     | _ -> assert false);
    (typer a TFloat32; 
     typer b TFloat32;
     update_type body TFloat32;)
  | MulF64 (_loc, a,b) ->
    (match t with
     | TUnknown | TFloat64 -> ()
     | _ -> assert false); 
    (typer a TFloat64; 
     typer b TFloat64;
     update_type body TFloat64;) 

  | Div (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt -> ()
     | _ -> assert false);
    (typer a TInt; 
     typer b TInt;
     update_type body TInt;)
  | Div32 (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt32 -> ()
     | _ -> assert false);
    (typer a TInt32; 
     typer b TInt32;
     update_type body TInt32;)
  | Div64 (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt64 -> ()
     | _ -> assert false);    (typer a TInt64; 
                               typer b TInt64;
                               update_type body TInt64;)
  | DivF (_loc, a,b) -> 
    (match t with
     | TUnknown | TFloat32 -> ()
     | _ -> assert false);    
    (typer a TFloat32; 
     typer b TFloat32;
     update_type body TFloat32;)
  | DivF32 (_loc, a,b) -> 
    (match t with
     | TUnknown | TFloat32 -> ()
     | _ -> assert false);
    (typer a TFloat32; 
     typer b TFloat32;
     update_type body TFloat32;)
  | DivF64 (_loc, a,b) ->
    (match t with
     | TUnknown | TFloat64 -> ()
     | _ -> assert false); 
    (typer a TFloat64; 
     typer b TFloat64;
     update_type body TFloat64;) 
  | Mod (_loc, a,b) -> 
    (match t with
     | TUnknown | TInt -> ()
     | _ -> assert false);
    (typer a TInt; 
     typer b TInt;
     update_type body TInt;)
  | Id (_loc,s) ->  ( 
      try
        let var = Hashtbl.find !current_args (string_of_ident s) in
        if t <> TUnknown && t <> TVec TUnknown && t <> TArr TUnknown then
            if var.var_type = TUnknown || var.var_type = TVec TUnknown || var.var_type = TArr TUnknown then
              ( var.var_type <- t;
                update_type body t)
            else
            if t <> var.var_type then
              match t, var.var_type with
              | TInt, TInt32 
              | TInt32, TInt 
              | TVec TInt, TVec TInt32
              | TVec TInt32, TVec TInt
              | TArr TInt, TArr TInt32
              | TArr TInt32, TArr TInt                -> 
                ( update_type body TInt32 )
              | _ ->
              ( assert (not debug); raise (TypeError (t, var.var_type, _loc)))
            else 
              ( update_type body t )
      with Not_found ->
        try 
          let c_const = Hashtbl.find !intrinsics_const (string_of_ident s) in
          if t = TUnknown then
           ( update_type body c_const.typ;)
          else if t <> c_const.typ then
            (assert (not debug); raise (TypeError (c_const.typ, t, _loc)))
        with _ -> 
          try ignore(Hashtbl.find !intrinsics_fun (string_of_ident s) )
          with _ ->
            Hashtbl.add !current_args (string_of_ident s) 
              {n = -1; var_type = t;
               is_mutable = false;
               read_only = true;
               write_only = false;
               is_global = true;}	    
    )
  | CastId (tt, Id(_loc,s)) -> 
    ( let var = 
      ( try Hashtbl.find !current_args (string_of_ident s) 
        with _  -> assert (not debug); raise (Unbound_value ((string_of_ident s),_loc))) in
      var.var_type <- t;
      update_type body t)
  | Int (_loc, i)  -> body.t <- TInt 
  | Int32 (_loc, i)  -> body.t <- TInt32 
  | Int64 (_loc, i)  -> body.t <- TInt64 
  | Float (_loc, f)  -> body.t <- TFloat32
  | Float32 (_loc, f) -> body.t <- TFloat32
  | Float64 (_loc, f) -> body.t <- TFloat64
  | Seq (_loc, x, y) -> 
     (typer x TUnit;
     typer y t; 
      update_type body y.t)
  | Seqs exprs ->
    let rec aux = function
      | [] -> ()
      | e::[] ->
        typer e TUnknown
      | e::q -> typer e TUnit; aux q
    in 
    aux exprs
  | End(_loc, x)  -> ()

  | Acc (_loc, var, value) ->
    (match var.e with
     | Id (_loc,s) -> (
         let var =
           ( try Hashtbl.find !current_args (string_of_ident s)
             with
             | _ -> assert false)
         in
         if not var.is_mutable then 
           raise (Immutable (string_of_ident s,_loc)))
     | _ -> assert false );
    typer var TUnknown;
    (match var.t with
     | TVec _ -> 
       typer {
         t = body.t; 
         e = (VecSet (_loc, var, value));
         loc = body.loc} 
         t;
       update_type body TUnit
     | _ -> 
       typer var TUnknown; 
       typer value var.t;
       update_type body TUnit
    )

  | VecSet (_loc, vector, value)  -> 
    ((match value.e with
        | Id (_loc, s) ->
          let var = 
            ( try Hashtbl.find !current_args (string_of_ident s)
              with _ -> assert (not debug); raise (Unbound_value ((string_of_ident s),_loc))) in
          typer value var.var_type;
          typer vector (value.t)
        | _ -> () ); (*typer value t);*)
     typer vector (value.t);
     (match vector.e with
      | VecGet(_,v,_) -> 
        (match v.e with 
         | Id (_loc,s)  -> 
           ( let var = 
             ( try Hashtbl.find !current_args (string_of_ident s)
               with _ -> assert (not debug); raise (Unbound_value ((string_of_ident s), _loc))) in
             var.var_type <- TVec value.t)
         | _  -> () )
      | _  -> () );
     update_type vector value.t;
     update_type body TUnit;)       
  | VecGet(_loc, vector, index)  -> 
    (typer vector (TVec t);
     typer index TInt;
     (match vector.e with
      | Id (_loc,s)  -> 
        ( let var = 
          ( try Hashtbl.find !current_args (string_of_ident s)
            with 
            | _ -> assert (not debug); raise (Unbound_value ((string_of_ident s), _loc))) in
          var.var_type <- TVec t)
      | _  ->  () );
     update_type vector (TVec t);
     update_type body t) 

  | ArrSet (_loc, array, value)  -> 
(*    ((match value.e with
        | Id (_loc, s) ->
          let var = 
            ( try Hashtbl.find !current_args (string_of_ident s)
              with _ -> assert (not debug); raise (Unbound_value ((string_of_ident s),_loc))) in
          (*          typer array (value.t);*)
          ( match  array.t with
          | TArr t -> typer value t);
        | _ -> ()); (*typer value t);*)
(*     typer array ( value.t);*)
     (match array.e with
      | ArrGet(_,a,_) -> 
        (match a.e with 
         | Id (_loc,s)  -> 
           ( let var = 
             ( try Hashtbl.find !current_args (string_of_ident s)
               with _ -> assert (not debug); raise (Unbound_value ((string_of_ident s), _loc))) in
             var.var_type <- TArr value.t)
         | _  -> () )
      | _  -> () );
      update_type array value.t;
      update_type body TUnit;)  
*)
    ((match value.e with
        | Id (_loc, s) ->
          let var = 
            ( try Hashtbl.find !current_args (string_of_ident s)
              with _ -> assert (not debug); raise (Unbound_value ((string_of_ident s),_loc))) in
          typer value var.var_type;
          typer array (value.t)
        | _ -> () ); (*typer value t);*)
     typer array (value.t);
     typer value array.t;
     (match array.e with
      | ArrGet(_,v,_) -> 
        (match v.e with 
         | Id (_loc,s)  -> 
           ( let var = 
             ( try Hashtbl.find !current_args (string_of_ident s)
               with _ -> assert (not debug); raise (Unbound_value ((string_of_ident s), _loc))) in
             var.var_type <- TArr value.t)
         | _  -> () )
      | _  -> () );
     update_type array value.t;
     update_type body TUnit;)        
   
  | ArrGet(_loc, array, index)  -> 
    (typer array (TArr t);
     typer index TInt;
     (match array.e with
      | Id (_loc,s)  -> 
        ( let var = 
          ( try Hashtbl.find !current_args (string_of_ident s)
            with 
            | _ -> assert (not debug); raise (Unbound_value ((string_of_ident s), _loc))) in
          var.var_type <- TArr t)
      | _  ->  () );
     update_type array (TArr t);
     update_type body t) 

(*    (typer array (TArr t);
     typer index TInt;
     (match array.e with
      | Id (_loc,s)  -> 
        ( let var = 
          ( try Hashtbl.find !current_args (string_of_ident s)
            with 
            | _ -> assert (not debug); raise (Unbound_value ((string_of_ident s), _loc))) in
          var.var_type <- TArr t)
      | _  ->  () );
      update_type array TArr t;
      update_type body t) *)
  (* | VecSet (_loc, vector, value)  ->  *)
  (*   begin  *)
  (*     begin *)
  (*       match value.e with *)
  (*       | Id (_loc, s) -> *)
  (*         let var =  *)
  (*           ( try Hashtbl.find !current_args (string_of_ident s) *)
  (*             with _ -> assert (not debug); raise (Unbound_value ((string_of_ident s),_loc))) in *)
  (*         typer value var.var_type; *)
  (*         typer vector (value.t) *)
  (*       | _ -> () (\*typer value t *\) *)
  (*     end; *)
  (*     typer vector (value.t); *)
  (*     (match vector.e with *)
  (*      | VecGet(_,v,_) ->  *)
  (*        (match v.e with  *)
  (*         | Id (_loc,s)  ->  *)
  (*           ( let var =  *)
  (*             ( try Hashtbl.find !current_args (string_of_ident s) *)
  (*               with _ -> assert (not debug); raise (Unbound_value ((string_of_ident s), _loc))) in *)
  (*            var.var_type <- TVec value.t) *)
  (*         | _  -> () ) *)
  (*      | _  -> () ); *)
  (*    update_type      vector value.t; *)
  (*    update_type      body TUnit;  *)
  (*   end         *)
  (* | VecGet(_loc, vector, index)  ->  *)
  (*   (typer vector (TVec t); *)
  (*    typer index TInt; *)
  (*    (match vector.e with *)
  (*     | Id (_loc,s)  ->  *)
  (*       ( let var =  *)
  (*         ( try Hashtbl.find !current_args (string_of_ident s) *)
  (*           with  *)
  (*           | _ -> assert (not debug); raise (Unbound_value ((string_of_ident s), _loc))) in *)
  (*         var.var_type <- TVec t) *)
  (*     | _  ->  () ); *)
  (*    vector.t <- TVec t; *)
  (*    body.t <- t)  *)
  | BoolEq(_loc, a, b) ->
    typer a TInt;
    typer b TInt;
    update_type body TBool;
  | BoolEq32(_loc, a, b) ->
    typer a TInt32;
    typer b TInt32;
    update_type body TBool;
  | BoolEq64(_loc, a, b) ->
    typer a TInt64;
    typer b TInt64;
    update_type body TBool;
  | BoolEqF(_loc, a, b) ->
    typer a TFloat32;
    typer b TFloat32;
    update_type body TBool;
  | BoolEqF64(_loc, a, b) ->
    typer a TFloat64;
    typer b TFloat64;
    update_type body TBool;
  | BoolLt(_loc, a, b) ->
    typer a TInt;
    typer b TInt;
    update_type body TBool;
  | BoolLt32(_loc, a, b) ->
    typer a TInt32;
    typer b TInt32;
    update_type body TBool;
  | BoolLt64(_loc, a, b) ->
    typer a TInt64;
    typer b TInt64;
    update_type body TBool;
  | BoolLtF(_loc, a, b) ->
    typer a TFloat32;
    typer b TFloat32;
    update_type body TBool;
  | BoolLtF64(_loc, a, b) ->
    typer a TFloat64;
    typer b TFloat64;
    update_type body TBool;  

  | BoolGt(_loc, a, b) ->
    typer a TInt;
    typer b TInt;
    update_type body TBool;
  | BoolGt32(_loc, a, b) ->
    typer a TInt32;
    typer b TInt32;
    update_type body TBool;
  | BoolGt64(_loc, a, b) ->
    typer a TInt64;
    typer b TInt64;
    update_type body TBool;
  | BoolGtF(_loc, a, b) ->
    typer a TFloat32;
    typer b TFloat32;
    update_type body TBool;
  | BoolGtF64(_loc, a, b) ->
    typer a TFloat64;
    typer b TFloat64;
    update_type body TBool;  

  | BoolLtE(_loc, a, b) ->
    typer a TInt;
    typer b TInt;
    update_type body TBool;
  | BoolLtE32(_loc, a, b) ->
    typer a TInt32;
    typer b TInt32;
    update_type body TBool;
  | BoolLtE64(_loc, a, b) ->
    typer a TInt64;
    typer b TInt64;
    update_type body TBool;
  | BoolLtEF(_loc, a, b) ->
    typer a TFloat32;
    typer b TFloat32;
    update_type body TBool;
  | BoolLtEF64(_loc, a, b) ->
    typer a TFloat64;
    typer b TFloat64;
    update_type body TBool;  


  | BoolGtE(_loc, a, b) ->
    typer a TInt;
    typer b TInt;
    update_type body TBool;
  | BoolGtE32(_loc, a, b) ->
    typer a TInt32;
    typer b TInt32;
    update_type body TBool;
  | BoolGtE64(_loc, a, b) ->
    typer a TInt64;
    typer b TInt64;
    update_type body TBool;
  | BoolGtEF(_loc, a, b) ->
    typer a TFloat32;
    typer b TFloat32;
    update_type body TBool;
  | BoolGtEF64(_loc, a, b) ->
    typer a TFloat64;
    typer b TFloat64;
    update_type body TBool;  

  | Ife (_loc, cond, cons1, cons2) ->
    (typer cond  TBool;
     typer cons1 t;	
     typer cons2 t;
     if cons1.t = TUnknown || cons1.t = TVec TUnknown then
       typer cons1 cons2.t;
     if cons2.t = TUnknown || cons2.t = TVec TUnknown then
       typer cons2 cons1.t;
     (match cons1.t,cons2.t with
      | TUnknown, TUnknown -> ()
      | TUnknown, _ -> update_type cons1  cons2.t
      | _, TUnknown -> update_type cons2 cons1.t
      | tt1, tt2 -> 
        if tt1 != tt2 then 
          ( assert (not debug); raise (TypeError (tt1, tt2, cons2.loc)))
        else ());
     update_type body cons1.t;
    )
  | If (_loc, cond, cons1) ->
    (typer cond  TBool;
     typer cons1 TUnknown;	
     update_type body cons1.t;
    )
  | DoLoop (l, id, min, max, body) ->
    (match id.e with
     | Id (_loc,s)  ->
       (incr arg_idx;
        Hashtbl.add !current_args (string_of_ident s) 
          {n = !arg_idx; var_type = TInt;
           is_mutable = false;
           read_only = false;
           write_only = false;
           is_global = false;
          }) 
     | CastId (tt, Id(_loc,s)) ->
       (incr arg_idx;
        Hashtbl.add !current_args (string_of_ident s) 
          {n = !arg_idx; var_type = TInt;
           is_mutable = false;
           read_only = false;
           write_only = false;
           is_global = false;})
     | _  ->  ()
    );
    typer id TInt;
    typer min TInt;
    typer max TInt;
    typer body t;
    update_type body TUnit;
  | While (l, cond, body) ->
    typer cond TBool;
    typer body TUnit;
    update_type body TUnit;
  | App (l, e1, e2) -> 
    let t = typer_app e1 e2 t in
    update_type body t
  | Open (l, id, e) ->
    let rec aux = function
      | IdAcc (l,a,b) -> aux a; aux b
      | IdUid (l,s) -> open_module s l
      | _ -> assert false
    in
    aux id;
    typer e TUnknown;
    let rec aux = function
      | IdAcc (l,a,b) -> aux a; aux b
      | IdUid (l,s) -> close_module s
      | _ -> assert false
    in
    aux id;
    update_type body e.t
  | Noop -> ()
  | BoolAnd (l, e1, e2)  | BoolOr (l, e1, e2) ->
    typer e1 TBool;
    typer e2 TBool;
    update_type body TBool
  | Ref (l,i) ->
    typer i t;
    update_type body i.t
  | ModuleAccess(l,s,e) ->
    open_module s l;
    typer e TUnknown;
    close_module (s);
    update_type body e.t;
  | _ -> assert false
and   typer body t = 
  my_eprintf (Printf.sprintf"(* typ %s -> expected  %s  ??? current %s *)\n%!" 
                (k_expr_to_string body.e) (ktyp_to_string t)  
                (ktyp_to_string body.t));
  let b = typer_ body t in
  my_eprintf (Printf.sprintf"(* +++ typ %s -> expected %s  ??? current %s *)\n%!" 
                (k_expr_to_string body.e) (ktyp_to_string t)  
                (ktyp_to_string body.t));
  b
  
and open_module m_ident  _loc = 
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
