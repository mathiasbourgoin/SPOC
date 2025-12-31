(******************************************************************************
 * Sarek_ir_cuda - CUDA Code Generation from Sarek IR
 *
 * Generates CUDA C source code from Sarek_ir.kernel.
 * This is the Phase 4 replacement for the Kirc_Ast-based Gen.ml generator.
 *
 * Features:
 * - Direct generation from clean IR (not legacy Kirc_Ast)
 * - Intrinsic registry support for extensible builtins
 * - Record/variant type support with C struct generation
 * - Pragma support for optimization hints
 ******************************************************************************)

open Sarek_ir

(** {1 Type Mapping} *)

(** Map Sarek IR element type to CUDA C type string *)
let rec cuda_type_of_elttype = function
  | TInt32 -> "int"
  | TInt64 -> "long long"
  | TFloat32 -> "float"
  | TFloat64 -> "double"
  | TBool -> "int"
  | TUnit -> "void"
  | TRecord (name, _) -> name
  | TVariant (name, _) -> name
  | TArray (elt, _) -> cuda_type_of_elttype elt ^ "*"
  | TVec elt -> cuda_type_of_elttype elt ^ "*"

(** Map Sarek IR element type to CUDA C type for kernel parameters *)
let cuda_param_type = function
  | TVec elt -> cuda_type_of_elttype elt ^ "* __restrict__"
  | TArray (elt, _) -> cuda_type_of_elttype elt ^ "*"
  | t -> cuda_type_of_elttype t

(** {1 Thread Intrinsics} *)

let cuda_thread_intrinsic = function
  | "thread_id_x" -> "threadIdx.x"
  | "thread_id_y" -> "threadIdx.y"
  | "thread_id_z" -> "threadIdx.z"
  | "block_id_x" -> "blockIdx.x"
  | "block_id_y" -> "blockIdx.y"
  | "block_id_z" -> "blockIdx.z"
  | "block_dim_x" -> "blockDim.x"
  | "block_dim_y" -> "blockDim.y"
  | "block_dim_z" -> "blockDim.z"
  | "grid_dim_x" -> "gridDim.x"
  | "grid_dim_y" -> "gridDim.y"
  | "grid_dim_z" -> "gridDim.z"
  | "global_thread_id" -> "(threadIdx.x + blockIdx.x * blockDim.x)"
  | "global_size" -> "(blockDim.x * gridDim.x)"
  | name -> failwith ("Unknown thread intrinsic: " ^ name)

(** {1 Expression Generation} *)

let rec gen_expr buf = function
  | EConst (CInt32 n) -> Buffer.add_string buf (Int32.to_string n)
  | EConst (CInt64 n) -> Buffer.add_string buf (Int64.to_string n ^ "LL")
  | EConst (CFloat32 f) -> Buffer.add_string buf (Printf.sprintf "%.17gf" f)
  | EConst (CFloat64 f) -> Buffer.add_string buf (Printf.sprintf "%.17g" f)
  | EConst (CBool true) -> Buffer.add_string buf "1"
  | EConst (CBool false) -> Buffer.add_string buf "0"
  | EConst CUnit -> Buffer.add_string buf "(void)0"
  | EVar v -> Buffer.add_string buf v.var_name
  | EBinop (op, e1, e2) ->
      Buffer.add_char buf '(' ;
      gen_expr buf e1 ;
      Buffer.add_string buf (gen_binop op) ;
      gen_expr buf e2 ;
      Buffer.add_char buf ')'
  | EUnop (op, e) ->
      Buffer.add_char buf '(' ;
      Buffer.add_string buf (gen_unop op) ;
      gen_expr buf e ;
      Buffer.add_char buf ')'
  | EArrayRead (arr, idx) ->
      Buffer.add_string buf arr ;
      Buffer.add_char buf '[' ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | ERecordField (e, field) ->
      gen_expr buf e ;
      Buffer.add_char buf '.' ;
      Buffer.add_string buf field
  | EIntrinsic (path, name, args) -> gen_intrinsic buf path name args
  | ECast (ty, e) ->
      Buffer.add_char buf '(' ;
      Buffer.add_string buf (cuda_type_of_elttype ty) ;
      Buffer.add_char buf ')' ;
      gen_expr buf e
  | ETuple exprs ->
      (* Tuples become struct literals in CUDA *)
      Buffer.add_string buf "{" ;
      List.iteri
        (fun i e ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        exprs ;
      Buffer.add_string buf "}"
  | EApp (fn, args) ->
      gen_expr buf fn ;
      Buffer.add_char buf '(' ;
      List.iteri
        (fun i e ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        args ;
      Buffer.add_char buf ')'
  | ERecord (name, fields) ->
      Buffer.add_string buf ("(" ^ name ^ "){") ;
      List.iteri
        (fun i (f, e) ->
          if i > 0 then Buffer.add_string buf ", " ;
          Buffer.add_string buf ("." ^ f ^ " = ") ;
          gen_expr buf e)
        fields ;
      Buffer.add_string buf "}"
  | EVariant (_, constr, []) -> Buffer.add_string buf constr
  | EVariant (type_name, constr, args) ->
      Buffer.add_string buf ("make_" ^ type_name ^ "_" ^ constr ^ "(") ;
      List.iteri
        (fun i e ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        args ;
      Buffer.add_char buf ')'
  | EArrayLen arr -> Buffer.add_string buf (arr ^ "_len")

and gen_binop = function
  | Add -> " + "
  | Sub -> " - "
  | Mul -> " * "
  | Div -> " / "
  | Mod -> " % "
  | Eq -> " == "
  | Ne -> " != "
  | Lt -> " < "
  | Le -> " <= "
  | Gt -> " > "
  | Ge -> " >= "
  | And -> " && "
  | Or -> " || "

and gen_unop = function Neg -> "-" | Not -> "!"

and gen_intrinsic buf path name args =
  (* Check for thread intrinsics first *)
  let full_name =
    match path with [] -> name | _ -> String.concat "." path ^ "." ^ name
  in
  (* Try thread intrinsics *)
  if
    List.mem
      name
      [
        "thread_id_x";
        "thread_id_y";
        "thread_id_z";
        "block_id_x";
        "block_id_y";
        "block_id_z";
        "block_dim_x";
        "block_dim_y";
        "block_dim_z";
        "grid_dim_x";
        "grid_dim_y";
        "grid_dim_z";
        "global_thread_id";
        "global_size";
      ]
  then Buffer.add_string buf (cuda_thread_intrinsic name)
  else
    (* Standard math intrinsics *)
    match name with
    | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh"
    | "tanh" | "exp" | "exp2" | "log" | "log2" | "log10" | "sqrt" | "rsqrt"
    | "cbrt" | "floor" | "ceil" | "round" | "trunc" | "fabs" ->
        Buffer.add_string buf name ;
        Buffer.add_char buf '(' ;
        List.iteri
          (fun i e ->
            if i > 0 then Buffer.add_string buf ", " ;
            gen_expr buf e)
          args ;
        Buffer.add_char buf ')'
    | "atan2" | "pow" | "fma" | "min" | "max" ->
        Buffer.add_string buf name ;
        Buffer.add_char buf '(' ;
        List.iteri
          (fun i e ->
            if i > 0 then Buffer.add_string buf ", " ;
            gen_expr buf e)
          args ;
        Buffer.add_char buf ')'
    | "atomic_add" ->
        Buffer.add_string buf "atomicAdd(" ;
        (match args with
        | [addr; value] ->
            Buffer.add_char buf '&' ;
            gen_expr buf addr ;
            Buffer.add_string buf ", " ;
            gen_expr buf value
        | _ -> failwith "atomic_add requires 2 arguments") ;
        Buffer.add_char buf ')'
    | "atomic_sub" ->
        Buffer.add_string buf "atomicSub(" ;
        (match args with
        | [addr; value] ->
            Buffer.add_char buf '&' ;
            gen_expr buf addr ;
            Buffer.add_string buf ", " ;
            gen_expr buf value
        | _ -> failwith "atomic_sub requires 2 arguments") ;
        Buffer.add_char buf ')'
    | "atomic_min" ->
        Buffer.add_string buf "atomicMin(" ;
        (match args with
        | [addr; value] ->
            Buffer.add_char buf '&' ;
            gen_expr buf addr ;
            Buffer.add_string buf ", " ;
            gen_expr buf value
        | _ -> failwith "atomic_min requires 2 arguments") ;
        Buffer.add_char buf ')'
    | "atomic_max" ->
        Buffer.add_string buf "atomicMax(" ;
        (match args with
        | [addr; value] ->
            Buffer.add_char buf '&' ;
            gen_expr buf addr ;
            Buffer.add_string buf ", " ;
            gen_expr buf value
        | _ -> failwith "atomic_max requires 2 arguments") ;
        Buffer.add_char buf ')'
    | _ ->
        (* Unknown intrinsic - emit as function call *)
        Buffer.add_string buf full_name ;
        Buffer.add_char buf '(' ;
        List.iteri
          (fun i e ->
            if i > 0 then Buffer.add_string buf ", " ;
            gen_expr buf e)
          args ;
        Buffer.add_char buf ')'

(** {1 L-value Generation} *)

let rec gen_lvalue buf = function
  | LVar v -> Buffer.add_string buf v.var_name
  | LArrayElem (arr, idx) ->
      Buffer.add_string buf arr ;
      Buffer.add_char buf '[' ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | LRecordField (lv, field) ->
      gen_lvalue buf lv ;
      Buffer.add_char buf '.' ;
      Buffer.add_string buf field

(** {1 Statement Generation} *)

let rec gen_stmt buf indent = function
  | SEmpty -> ()
  | SSeq stmts -> List.iter (gen_stmt buf indent) stmts
  | SAssign (lv, e) ->
      Buffer.add_string buf indent ;
      gen_lvalue buf lv ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | SIf (cond, then_, else_opt) -> (
      Buffer.add_string buf indent ;
      Buffer.add_string buf "if (" ;
      gen_expr buf cond ;
      Buffer.add_string buf ") {\n" ;
      gen_stmt buf (indent ^ "  ") then_ ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}" ;
      match else_opt with
      | None -> Buffer.add_char buf '\n'
      | Some else_ ->
          Buffer.add_string buf " else {\n" ;
          gen_stmt buf (indent ^ "  ") else_ ;
          Buffer.add_string buf indent ;
          Buffer.add_string buf "}\n")
  | SWhile (cond, body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "while (" ;
      gen_expr buf cond ;
      Buffer.add_string buf ") {\n" ;
      gen_stmt buf (indent ^ "  ") body ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"
  | SFor (v, start, stop, dir, body) ->
      let op, incr =
        match dir with Upto -> ("<", "++") | Downto -> (">", "--")
      in
      Buffer.add_string buf indent ;
      Buffer.add_string buf "for (" ;
      Buffer.add_string buf (cuda_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf start ;
      Buffer.add_string buf "; " ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf (" " ^ op ^ " ") ;
      gen_expr buf stop ;
      Buffer.add_string buf "; " ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf incr ;
      Buffer.add_string buf ") {\n" ;
      gen_stmt buf (indent ^ "  ") body ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"
  | SMatch (e, cases) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "switch (" ;
      gen_expr buf e ;
      Buffer.add_string buf ".tag) {\n" ;
      List.iter
        (fun (pattern, body) ->
          Buffer.add_string buf indent ;
          (match pattern with
          | PConstr (name, _) -> Buffer.add_string buf ("  case " ^ name ^ ":\n")
          | PWild -> Buffer.add_string buf "  default:\n") ;
          gen_stmt buf (indent ^ "    ") body ;
          Buffer.add_string buf (indent ^ "    break;\n"))
        cases ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"
  | SReturn e ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "return " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | SBarrier ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "__syncthreads();\n"
  | SWarpBarrier ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "__syncwarp();\n"
  | SMemFence ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "__threadfence();\n"
  | SExpr e ->
      Buffer.add_string buf indent ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | SLet (v, e, body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (cuda_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n" ;
      gen_stmt buf indent body
  | SLetMut (v, e, body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (cuda_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n" ;
      gen_stmt buf indent body
  | SPragma (hints, body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "#pragma " ;
      Buffer.add_string buf (String.concat " " hints) ;
      Buffer.add_char buf '\n' ;
      gen_stmt buf indent body

(** {1 Declaration Generation} *)

let gen_param buf = function
  | DParam (v, None) ->
      Buffer.add_string buf (cuda_param_type v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name
  | DParam (v, Some arr) ->
      Buffer.add_string buf (cuda_type_of_elttype arr.arr_elttype) ;
      Buffer.add_string buf "* __restrict__ " ;
      Buffer.add_string buf v.var_name
  | DLocal _ | DShared _ -> failwith "gen_param: expected DParam"

let gen_local buf indent = function
  | DLocal (v, None) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (cuda_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf ";\n"
  | DLocal (v, Some e) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (cuda_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | DShared (name, elt, None) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "__shared__ " ;
      Buffer.add_string buf (cuda_type_of_elttype elt) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf name ;
      Buffer.add_string buf "[];\n"
  | DShared (name, elt, Some size) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "__shared__ " ;
      Buffer.add_string buf (cuda_type_of_elttype elt) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf name ;
      Buffer.add_char buf '[' ;
      gen_expr buf size ;
      Buffer.add_string buf "];\n"
  | DParam _ -> failwith "gen_local: expected DLocal or DShared"

(** {1 Kernel Generation} *)

(** Generate the CUDA kernel header *)
let cuda_header = {|
extern "C" {
|}

(** Generate complete CUDA source for a kernel *)
let generate (k : kernel) : string =
  let buf = Buffer.create 4096 in

  (* Header *)
  Buffer.add_string buf cuda_header ;

  (* Kernel signature *)
  Buffer.add_string buf "__global__ void " ;
  Buffer.add_string buf k.kern_name ;
  Buffer.add_char buf '(' ;

  (* Parameters *)
  List.iteri
    (fun i p ->
      if i > 0 then Buffer.add_string buf ", " ;
      gen_param buf p)
    k.kern_params ;

  Buffer.add_string buf ") {\n" ;

  (* Local declarations *)
  List.iter (gen_local buf "  ") k.kern_locals ;

  (* Body *)
  gen_stmt buf "  " k.kern_body ;

  (* Close kernel *)
  Buffer.add_string buf "}\n" ;

  (* Close extern "C" *)
  Buffer.add_string buf "}\n" ;

  Buffer.contents buf

(** Generate CUDA source with custom type definitions *)
let generate_with_types ~(types : (string * (string * elttype) list) list)
    (k : kernel) : string =
  let buf = Buffer.create 4096 in

  (* Header *)
  Buffer.add_string buf cuda_header ;

  (* Type definitions *)
  List.iter
    (fun (name, fields) ->
      Buffer.add_string buf "typedef struct {\n" ;
      List.iter
        (fun (fname, ftype) ->
          Buffer.add_string buf "  " ;
          Buffer.add_string buf (cuda_type_of_elttype ftype) ;
          Buffer.add_char buf ' ' ;
          Buffer.add_string buf fname ;
          Buffer.add_string buf ";\n")
        fields ;
      Buffer.add_string buf "} " ;
      Buffer.add_string buf name ;
      Buffer.add_string buf ";\n\n")
    types ;

  (* Kernel signature *)
  Buffer.add_string buf "__global__ void " ;
  Buffer.add_string buf k.kern_name ;
  Buffer.add_char buf '(' ;

  (* Parameters *)
  List.iteri
    (fun i p ->
      if i > 0 then Buffer.add_string buf ", " ;
      gen_param buf p)
    k.kern_params ;

  Buffer.add_string buf ") {\n" ;

  (* Local declarations *)
  List.iter (gen_local buf "  ") k.kern_locals ;

  (* Body *)
  gen_stmt buf "  " k.kern_body ;

  (* Close kernel *)
  Buffer.add_string buf "}\n" ;

  (* Close extern "C" *)
  Buffer.add_string buf "}\n" ;

  Buffer.contents buf
