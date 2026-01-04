(******************************************************************************
 * Sarek_ir_glsl - GLSL Compute Shader Generation from Sarek IR
 *
 * Generates GLSL compute shader source code from Sarek_ir.kernel.
 * The output is compatible with Vulkan compute shaders and can be compiled
 * to SPIR-V using glslangValidator.
 *
 * Features:
 * - Direct generation from clean IR
 * - Storage buffer bindings for arrays
 * - Record/variant type support with GLSL structs
 * - Workgroup size configuration
 ******************************************************************************)

open Sarek.Sarek_ir

(** Current kernel's variant definitions (set during generate) *)
let current_variants : (string * (string * elttype list) list) list ref = ref []

(** {1 Type Mapping} *)

(** Mangle OCaml type name to valid GLSL identifier *)
let mangle_name name = String.map (fun c -> if c = '.' then '_' else c) name

(** Map Sarek IR element type to GLSL type string *)
let rec glsl_type_of_elttype = function
  | TInt32 -> "int"
  | TInt64 -> "int64_t" (* Requires GL_ARB_gpu_shader_int64 *)
  | TFloat32 -> "float"
  | TFloat64 -> "double" (* Requires GL_ARB_gpu_shader_fp64 *)
  | TBool -> "bool"
  | TUnit -> "void"
  | TRecord (name, _) -> mangle_name name
  | TVariant (name, _) -> mangle_name name
  | TArray (elt, _) -> glsl_type_of_elttype elt (* Arrays are special in GLSL *)
  | TVec elt -> glsl_type_of_elttype elt

(** {1 Thread Intrinsics} *)

let glsl_thread_intrinsic = function
  | "thread_id_x" | "thread_idx_x" -> "int(gl_LocalInvocationID.x)"
  | "thread_id_y" | "thread_idx_y" -> "int(gl_LocalInvocationID.y)"
  | "thread_id_z" | "thread_idx_z" -> "int(gl_LocalInvocationID.z)"
  | "block_id_x" | "block_idx_x" -> "int(gl_WorkGroupID.x)"
  | "block_id_y" | "block_idx_y" -> "int(gl_WorkGroupID.y)"
  | "block_id_z" | "block_idx_z" -> "int(gl_WorkGroupID.z)"
  | "block_dim_x" -> "int(gl_WorkGroupSize.x)"
  | "block_dim_y" -> "int(gl_WorkGroupSize.y)"
  | "block_dim_z" -> "int(gl_WorkGroupSize.z)"
  | "grid_dim_x" -> "int(gl_NumWorkGroups.x)"
  | "grid_dim_y" -> "int(gl_NumWorkGroups.y)"
  | "grid_dim_z" -> "int(gl_NumWorkGroups.z)"
  | "global_thread_id" | "global_idx" | "global_idx_x" ->
      "int(gl_GlobalInvocationID.x)"
  | "global_idx_y" -> "int(gl_GlobalInvocationID.y)"
  | "global_idx_z" -> "int(gl_GlobalInvocationID.z)"
  | "global_size" -> "int(gl_WorkGroupSize.x * gl_NumWorkGroups.x)"
  | name -> failwith ("Unknown thread intrinsic: " ^ name)

(** {1 Expression Generation} *)

let rec gen_expr buf = function
  | EConst (CInt32 n) -> Buffer.add_string buf (Int32.to_string n)
  | EConst (CInt64 n) -> Buffer.add_string buf (Int64.to_string n ^ "L")
  | EConst (CFloat32 f) ->
      let s = Printf.sprintf "%.17g" f in
      let s =
        if String.contains s '.' || String.contains s 'e' then s else s ^ ".0"
      in
      Buffer.add_string buf s
  | EConst (CFloat64 f) ->
      let s = Printf.sprintf "%.17g" f in
      let s =
        if String.contains s '.' || String.contains s 'e' then s else s ^ ".0"
      in
      Buffer.add_string buf (s ^ "lf")
  | EConst (CBool true) -> Buffer.add_string buf "true"
  | EConst (CBool false) -> Buffer.add_string buf "false"
  | EConst CUnit -> Buffer.add_string buf "/* unit */"
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
      Buffer.add_string buf ".data[" ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | EArrayReadExpr (base, idx) ->
      Buffer.add_char buf '(' ;
      gen_expr buf base ;
      Buffer.add_string buf ")[" ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | ERecordField (e, field) ->
      gen_expr buf e ;
      Buffer.add_char buf '.' ;
      Buffer.add_string buf field
  | EIntrinsic (path, name, args) -> gen_intrinsic buf path name args
  | ECast (ty, e) ->
      Buffer.add_string buf (glsl_type_of_elttype ty) ;
      Buffer.add_char buf '(' ;
      gen_expr buf e ;
      Buffer.add_char buf ')'
  | ETuple exprs ->
      (* Tuples become struct literals in GLSL *)
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
      Buffer.add_string buf (mangle_name name ^ "(") ;
      List.iteri
        (fun i (_, e) ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        fields ;
      Buffer.add_char buf ')'
  | EVariant (type_name, constr, args) ->
      Buffer.add_string
        buf
        ("make_" ^ mangle_name type_name ^ "_" ^ constr ^ "(") ;
      List.iteri
        (fun i e ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        args ;
      Buffer.add_char buf ')'
  | EArrayLen arr -> Buffer.add_string buf ("sarek_" ^ arr ^ "_length")
  | EArrayCreate _ ->
      failwith "gen_expr: EArrayCreate should be handled in gen_stmt SLet"
  | EIf (cond, then_, else_) ->
      Buffer.add_char buf '(' ;
      gen_expr buf cond ;
      Buffer.add_string buf " ? " ;
      gen_expr buf then_ ;
      Buffer.add_string buf " : " ;
      gen_expr buf else_ ;
      Buffer.add_char buf ')'
  | EMatch (_, []) -> failwith "gen_expr: empty match"
  | EMatch (_, [(_, body)]) -> gen_expr buf body
  | EMatch (e, cases) ->
      let rec gen_cases = function
        | [] -> failwith "gen_expr: empty match cases"
        | [(_, body)] -> gen_expr buf body
        | (pat, body) :: rest ->
            Buffer.add_char buf '(' ;
            (match pat with
            | PConstr (name, _) ->
                Buffer.add_char buf '(' ;
                gen_expr buf e ;
                Buffer.add_string buf (".tag == " ^ name ^ ")")
            | PWild -> Buffer.add_string buf "true") ;
            Buffer.add_string buf " ? " ;
            gen_expr buf body ;
            Buffer.add_string buf " : " ;
            gen_cases rest ;
            Buffer.add_char buf ')'
      in
      gen_cases cases

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
  | Shl -> " << "
  | Shr -> " >> "
  | BitAnd -> " & "
  | BitOr -> " | "
  | BitXor -> " ^ "

and gen_unop = function Neg -> "-" | Not -> "!" | BitNot -> "~"

and gen_intrinsic buf path name args =
  let full_name =
    match path with [] -> name | _ -> String.concat "." path ^ "." ^ name
  in
  (* Try thread intrinsics *)
  if
    List.mem
      name
      [
        "thread_id_x";
        "thread_idx_x";
        "thread_id_y";
        "thread_idx_y";
        "thread_id_z";
        "thread_idx_z";
        "block_id_x";
        "block_idx_x";
        "block_id_y";
        "block_idx_y";
        "block_id_z";
        "block_idx_z";
        "block_dim_x";
        "block_dim_y";
        "block_dim_z";
        "grid_dim_x";
        "grid_dim_y";
        "grid_dim_z";
        "global_thread_id";
        "global_idx";
        "global_idx_x";
        "global_idx_y";
        "global_idx_z";
        "global_size";
      ]
  then Buffer.add_string buf (glsl_thread_intrinsic name)
  else
    (* Standard math intrinsics - GLSL versions *)
    match name with
    | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh"
    | "tanh" | "exp" | "exp2" | "log" | "log2" | "sqrt" | "floor" | "ceil"
    | "round" | "trunc" | "abs" ->
        Buffer.add_string buf name ;
        Buffer.add_char buf '(' ;
        List.iteri
          (fun i e ->
            if i > 0 then Buffer.add_string buf ", " ;
            gen_expr buf e)
          args ;
        Buffer.add_char buf ')'
    | "fabs" ->
        Buffer.add_string buf "abs" ;
        Buffer.add_char buf '(' ;
        List.iteri
          (fun i e ->
            if i > 0 then Buffer.add_string buf ", " ;
            gen_expr buf e)
          args ;
        Buffer.add_char buf ')'
    | "rsqrt" ->
        Buffer.add_string buf "inversesqrt" ;
        Buffer.add_char buf '(' ;
        List.iteri
          (fun i e ->
            if i > 0 then Buffer.add_string buf ", " ;
            gen_expr buf e)
          args ;
        Buffer.add_char buf ')'
    | "atan2" | "pow" | "min" | "max" ->
        Buffer.add_string buf name ;
        Buffer.add_char buf '(' ;
        List.iteri
          (fun i e ->
            if i > 0 then Buffer.add_string buf ", " ;
            gen_expr buf e)
          args ;
        Buffer.add_char buf ')'
    | "fma" ->
        Buffer.add_string buf "fma" ;
        Buffer.add_char buf '(' ;
        List.iteri
          (fun i e ->
            if i > 0 then Buffer.add_string buf ", " ;
            gen_expr buf e)
          args ;
        Buffer.add_char buf ')'
    (* Barrier synchronization *)
    | "block_barrier" -> Buffer.add_string buf "barrier()"
    (* Atomic operations - GLSL uses atomicAdd etc. *)
    | "atomic_add" | "atomic_add_int32" | "atomic_add_global_int32" ->
        Buffer.add_string buf "atomicAdd(" ;
        (match args with
        | [addr; value] ->
            gen_expr buf addr ;
            Buffer.add_string buf ", " ;
            gen_expr buf value
        | [arr; idx; value] ->
            gen_expr buf arr ;
            Buffer.add_string buf ".data[" ;
            gen_expr buf idx ;
            Buffer.add_string buf "], " ;
            gen_expr buf value
        | _ -> failwith "atomic_add requires 2 or 3 arguments") ;
        Buffer.add_char buf ')'
    | "atomic_min" ->
        Buffer.add_string buf "atomicMin(" ;
        (match args with
        | [addr; value] ->
            gen_expr buf addr ;
            Buffer.add_string buf ", " ;
            gen_expr buf value
        | _ -> failwith "atomic_min requires 2 arguments") ;
        Buffer.add_char buf ')'
    | "atomic_max" ->
        Buffer.add_string buf "atomicMax(" ;
        (match args with
        | [addr; value] ->
            gen_expr buf addr ;
            Buffer.add_string buf ", " ;
            gen_expr buf value
        | _ -> failwith "atomic_max requires 2 arguments") ;
        Buffer.add_char buf ')'
    | "float" ->
        Buffer.add_string buf "float(" ;
        (match args with [e] -> gen_expr buf e | _ -> ()) ;
        Buffer.add_char buf ')'
    | "int_of_float" ->
        Buffer.add_string buf "int(" ;
        (match args with [e] -> gen_expr buf e | _ -> ()) ;
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
      Buffer.add_string buf ".data[" ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | LArrayElemExpr (base, idx) ->
      Buffer.add_char buf '(' ;
      gen_expr buf base ;
      Buffer.add_string buf ")[" ;
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
        match dir with Upto -> ("<=", "++") | Downto -> (">=", "--")
      in
      Buffer.add_string buf indent ;
      Buffer.add_string buf "for (" ;
      Buffer.add_string buf (glsl_type_of_elttype v.var_type) ;
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
      let scrutinee_buf = Buffer.create 64 in
      gen_expr scrutinee_buf e ;
      let scrutinee = Buffer.contents scrutinee_buf in
      let find_constr_types cname =
        List.find_map
          (fun (_vname, constrs) ->
            List.find_map
              (fun (cn, args) -> if cn = cname then Some args else None)
              constrs)
          !current_variants
      in
      Buffer.add_string buf indent ;
      Buffer.add_string buf "switch (" ;
      Buffer.add_string buf scrutinee ;
      Buffer.add_string buf ".tag) {\n" ;
      List.iter
        (fun (pattern, body) ->
          Buffer.add_string buf indent ;
          (match pattern with
          | PConstr (cname, bindings) -> (
              Buffer.add_string buf ("  case " ^ cname ^ ": {\n") ;
              match (bindings, find_constr_types cname) with
              | [var_name], Some [ty] ->
                  Buffer.add_string buf (indent ^ "    ") ;
                  Buffer.add_string buf (glsl_type_of_elttype ty) ;
                  Buffer.add_string buf " " ;
                  Buffer.add_string buf var_name ;
                  Buffer.add_string buf " = " ;
                  Buffer.add_string buf scrutinee ;
                  Buffer.add_string buf ".data." ;
                  Buffer.add_string buf cname ;
                  Buffer.add_string buf "_v;\n"
              | vars, Some types when List.length vars = List.length types ->
                  List.iteri
                    (fun i (var_name, ty) ->
                      Buffer.add_string buf (indent ^ "    ") ;
                      Buffer.add_string buf (glsl_type_of_elttype ty) ;
                      Buffer.add_string buf " " ;
                      Buffer.add_string buf var_name ;
                      Buffer.add_string buf " = " ;
                      Buffer.add_string buf scrutinee ;
                      Buffer.add_string buf ".data." ;
                      Buffer.add_string buf cname ;
                      Buffer.add_string buf (Printf.sprintf "_v._%d;\n" i))
                    (List.combine vars types)
              | [], _ | _, None | _, Some [] -> ()
              | _ ->
                  failwith
                    "Mismatch between pattern bindings and constructor args")
          | PWild -> Buffer.add_string buf "  default: {\n") ;
          gen_stmt buf (indent ^ "    ") body ;
          Buffer.add_string buf (indent ^ "    break;\n") ;
          Buffer.add_string buf (indent ^ "  }\n"))
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
      Buffer.add_string buf "barrier();\n"
  | SWarpBarrier ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "subgroupBarrier();\n"
  | SMemFence ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "memoryBarrier();\n"
  | SNative _ ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "/* native code not supported in GLSL */\n"
  | SExpr e ->
      Buffer.add_string buf indent ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | SLet (v, EArrayCreate (elem_ty, size, Shared), body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "shared " ;
      Buffer.add_string buf (glsl_type_of_elttype elem_ty) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_char buf '[' ;
      gen_expr buf size ;
      Buffer.add_string buf "];\n" ;
      gen_stmt buf indent body
  | SLet (v, EArrayCreate (elem_ty, size, _), body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (glsl_type_of_elttype elem_ty) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_char buf '[' ;
      gen_expr buf size ;
      Buffer.add_string buf "];\n" ;
      gen_stmt buf indent body
  | SLet (v, e, body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (glsl_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n" ;
      gen_stmt buf indent body
  | SLetMut (v, e, body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (glsl_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n" ;
      gen_stmt buf indent body
  | SPragma (_hints, body) ->
      (* GLSL doesn't have #pragma in the same way *)
      gen_stmt buf indent body
  | SBlock body ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "{\n" ;
      gen_stmt buf (indent ^ "  ") body ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"

(** {1 Helper Function Generation} *)

let gen_helper_func buf (hf : helper_func) =
  Buffer.add_string buf (glsl_type_of_elttype hf.hf_ret_type) ;
  Buffer.add_char buf ' ' ;
  Buffer.add_string buf hf.hf_name ;
  Buffer.add_char buf '(' ;
  List.iteri
    (fun i (v : var) ->
      if i > 0 then Buffer.add_string buf ", " ;
      Buffer.add_string buf (glsl_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name)
    hf.hf_params ;
  Buffer.add_string buf ") {\n" ;
  gen_stmt buf "  " hf.hf_body ;
  Buffer.add_string buf "}\n\n"

(** {1 Kernel Generation} *)

(** Count vector parameters for binding assignment *)
let count_vec_params params =
  List.fold_left
    (fun acc decl ->
      match decl with
      | DParam (v, _) -> ( match v.var_type with TVec _ -> acc + 1 | _ -> acc)
      | _ -> acc)
    0
    params

(** Generate GLSL compute shader header *)
let glsl_header ~kernel_name =
  Printf.sprintf
    {|#version 450

// Sarek-generated compute shader: %s
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

|}
    kernel_name

(** Generate buffer binding for a vector parameter *)
let gen_buffer_binding buf binding_idx v elem_type =
  Buffer.add_string
    buf
    (Printf.sprintf
       "layout(std430, binding = %d) buffer Buffer_%s {\n"
       binding_idx
       v.var_name) ;
  Buffer.add_string
    buf
    (Printf.sprintf "  %s data[];\n" (glsl_type_of_elttype elem_type)) ;
  Buffer.add_string buf (Printf.sprintf "} %s;\n" v.var_name)

let gen_push_constants buf params =
  let vectors = ref [] in
  let scalars = ref [] in
  List.iter
    (fun decl ->
      match decl with
      | DParam (v, _) -> (
          match v.var_type with
          | TVec _ -> vectors := v :: !vectors
          | _ -> scalars := v :: !scalars)
      | _ -> ())
    params ;
  let vectors = List.rev !vectors in
  let scalars = List.rev !scalars in
  (* Generate push constants if we have vectors (for lengths) or scalars *)
  if vectors <> [] || scalars <> [] then begin
    Buffer.add_string buf "layout(push_constant) uniform PushConstants {\n" ;
    (* Add length parameter for each vector *)
    List.iter
      (fun v ->
        Buffer.add_string buf (Printf.sprintf "  int %s_len;\n" v.var_name))
      vectors ;
    (* Add user-defined scalar parameters *)
    List.iter
      (fun v ->
        Buffer.add_string
          buf
          (Printf.sprintf
             "  %s %s;\n"
             (glsl_type_of_elttype v.var_type)
             v.var_name))
      scalars ;
    Buffer.add_string buf "} pc;\n\n" ;
    (* Define convenience aliases *)
    List.iter
      (fun v ->
        Buffer.add_string
          buf
          (Printf.sprintf "#define %s_len pc.%s_len\n" v.var_name v.var_name))
      vectors ;
    List.iter
      (fun v ->
        Buffer.add_string
          buf
          (Printf.sprintf "#define %s pc.%s\n" v.var_name v.var_name))
      scalars ;
    Buffer.add_string buf "\n"
  end

(** Generate complete GLSL source for a kernel *)
let generate (k : kernel) : string =
  let buf = Buffer.create 4096 in

  (* Header *)
  Buffer.add_string buf (glsl_header ~kernel_name:k.kern_name) ;

  (* Generate buffer bindings for vector parameters *)
  let binding_idx = ref 0 in
  List.iter
    (fun decl ->
      match decl with
      | DParam (v, _) -> (
          match v.var_type with
          | TVec elem_type ->
              gen_buffer_binding buf !binding_idx v elem_type ;
              incr binding_idx
          | _ -> ())
      | _ -> ())
    k.kern_params ;

  Buffer.add_char buf '\n' ;

  (* Generate push constants for scalar parameters *)
  gen_push_constants buf k.kern_params ;

  Buffer.add_char buf '\n' ;

  (* Generate helper functions before main *)
  List.iter (gen_helper_func buf) k.kern_funcs ;

  (* Main function *)
  Buffer.add_string buf "void main() {\n" ;

  (* Body *)
  gen_stmt buf "  " k.kern_body ;

  (* Close main *)
  Buffer.add_string buf "}\n" ;

  let result = Buffer.contents buf in
  Printf.eprintf "[GLSL] Generated shader:\n%s\n%!" result ;
  result

(** Generate GLSL variant type definition *)
let gen_variant_def buf (name, constrs) =
  let mangled = mangle_name name in
  (* Enum constants *)
  List.iteri
    (fun i (cname, _) ->
      Buffer.add_string buf (Printf.sprintf "const int %s = %d;\n" cname i))
    constrs ;
  Buffer.add_char buf '\n' ;
  (* Struct with tag and union-like data *)
  Buffer.add_string buf (Printf.sprintf "struct %s {\n  int tag;\n" mangled) ;
  let has_payload = List.exists (fun (_, args) -> args <> []) constrs in
  if has_payload then begin
    (* GLSL doesn't have unions, so we use the largest payload type *)
    List.iter
      (fun (cname, args) ->
        match args with
        | [] -> ()
        | [ty] ->
            Buffer.add_string
              buf
              (Printf.sprintf "  %s %s_v;\n" (glsl_type_of_elttype ty) cname)
        | _ ->
            (* Multiple args - would need struct, simplify for now *)
            Buffer.add_string
              buf
              (Printf.sprintf
                 "  // %s has multiple args - not supported\n"
                 cname))
      constrs
  end ;
  Buffer.add_string buf "};\n\n" ;
  (* Constructor functions *)
  List.iteri
    (fun _i (cname, args) ->
      Buffer.add_string
        buf
        (Printf.sprintf "%s make_%s_%s(" mangled mangled cname) ;
      (match args with
      | [] -> ()
      | [ty] ->
          Buffer.add_string buf (glsl_type_of_elttype ty) ;
          Buffer.add_string buf " v"
      | _ -> ()) ;
      Buffer.add_string buf ") {\n" ;
      Buffer.add_string buf (Printf.sprintf "  %s r;\n" mangled) ;
      Buffer.add_string buf (Printf.sprintf "  r.tag = %s;\n" cname) ;
      (match args with
      | [] -> ()
      | [_] -> Buffer.add_string buf (Printf.sprintf "  r.%s_v = v;\n" cname)
      | _ -> ()) ;
      Buffer.add_string buf "  return r;\n}\n\n")
    constrs

(** Generate GLSL source with custom type definitions *)
let generate_with_types ~(types : (string * (string * elttype) list) list)
    (k : kernel) : string =
  current_variants := k.kern_variants ;
  let buf = Buffer.create 4096 in

  (* Header *)
  Buffer.add_string buf (glsl_header ~kernel_name:k.kern_name) ;

  (* Variant type definitions first *)
  List.iter (gen_variant_def buf) k.kern_variants ;

  (* Record type definitions *)
  List.iter
    (fun (name, fields) ->
      Buffer.add_string buf (Printf.sprintf "struct %s {\n" (mangle_name name)) ;
      List.iter
        (fun (fname, ftype) ->
          Buffer.add_string
            buf
            (Printf.sprintf "  %s %s;\n" (glsl_type_of_elttype ftype) fname))
        fields ;
      Buffer.add_string buf "};\n\n")
    types ;

  (* Generate buffer bindings for vector parameters *)
  let binding_idx = ref 0 in
  List.iter
    (fun decl ->
      match decl with
      | DParam (v, _) -> (
          match v.var_type with
          | TVec elem_type ->
              gen_buffer_binding buf !binding_idx v elem_type ;
              incr binding_idx
          | _ -> ())
      | _ -> ())
    k.kern_params ;

  Buffer.add_char buf '\n' ;

  (* Generate push constants for scalar parameters *)
  gen_push_constants buf k.kern_params ;

  Buffer.add_char buf '\n' ;

  (* Generate helper functions before main *)
  List.iter (gen_helper_func buf) k.kern_funcs ;

  (* Main function *)
  Buffer.add_string buf "void main() {\n" ;

  (* Body *)
  gen_stmt buf "  " k.kern_body ;

  (* Close main *)
  Buffer.add_string buf "}\n" ;

  let result = Buffer.contents buf in
  Printf.eprintf "[GLSL] Generated shader:\n%s\n%!" result ;
  result
