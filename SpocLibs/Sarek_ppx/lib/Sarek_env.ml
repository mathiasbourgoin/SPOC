(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module defines the immutable typing environment for name resolution
 * and type checking. Replaces the global mutable hashtables from the old
 * implementation.
 ******************************************************************************)

open Sarek_types
module StringMap = Map.Make (String)

(** Information about a variable *)
type var_info = {
  vi_type : typ;
  vi_mutable : bool;
  vi_is_param : bool;  (** Kernel parameter? *)
  vi_index : int;  (** For parameter ordering *)
  vi_is_vec : bool;  (** Is this a vector parameter? *)
}

(** Reference to an intrinsic function in a stdlib module. The module path
    allows the PPX to generate correct references for compile-time type
    checking. For example:
    - IntrinsicRef (["Sarek_stdlib"; "Float32"], "sin") ->
      Sarek_stdlib.Float32.sin
    - IntrinsicRef (["Sarek_stdlib"; "Int32"], "add_int32") ->
      Sarek_stdlib.Int32.add_int32

    This enables extensibility: user libraries can define their own intrinsics
    via %sarek_intrinsic and the PPX will reference them correctly. *)
type intrinsic_ref =
  | IntrinsicRef of string list * string  (** module_path, function_name *)

(** Information about an intrinsic function. Note: intr_cuda and intr_opencl are
    kept for the lowering phase. The JIT uses Sarek_registry for device code. *)
type intrinsic_fun_info = {
  intr_type : typ;
  intr_cuda : string;  (** TODO: Remove once lowering uses registry directly *)
  intr_opencl : string;
      (** TODO: Remove once lowering uses registry directly *)
  intr_ocaml : intrinsic_ref;  (** Reference to stdlib module function *)
}

(** Information about an intrinsic constant *)
type intrinsic_const_info = {
  const_type : typ;
  const_cuda : string;
  const_opencl : string;
}

(** Information about a custom type *)
type type_info =
  | TIRecord of {
      ti_name : string;
      ti_fields : (string * typ * bool) list;  (** name, type, mutable *)
    }
  | TIVariant of {
      ti_name : string;
      ti_constrs : (string * typ option) list;
          (** constructor name, optional arg type *)
    }

(** The typing environment - immutable *)
type t = {
  vars : var_info StringMap.t;
  types : type_info StringMap.t;
  intrinsic_funs : intrinsic_fun_info StringMap.t;
  intrinsic_consts : intrinsic_const_info StringMap.t;
  constructors : (string * typ option) StringMap.t;
      (** constr -> (type_name, arg_type) *)
  fields : (string * int * typ * bool) StringMap.t;
      (** field -> (type_name, index, type, mutable) *)
  current_level : int;  (** For let-polymorphism *)
  local_funs : (string * typ) StringMap.t;  (** Local function definitions *)
}

(** Empty environment *)
let empty =
  {
    vars = StringMap.empty;
    types = StringMap.empty;
    intrinsic_funs = StringMap.empty;
    intrinsic_consts = StringMap.empty;
    constructors = StringMap.empty;
    fields = StringMap.empty;
    current_level = 0;
    local_funs = StringMap.empty;
  }

(** Environment operations - all return new environments *)

let add_var name info env = {env with vars = StringMap.add name info env.vars}

let add_type name info env =
  match info with
  | TIRecord {ti_fields; ti_name; _} ->
      let env = {env with types = StringMap.add name info env.types} in
      (* Also add field mappings *)
      let _, fields =
        List.fold_left
          (fun (idx, fields) (fname, ftyp, fmut) ->
            (idx + 1, StringMap.add fname (ti_name, idx, ftyp, fmut) fields))
          (0, env.fields)
          ti_fields
      in
      {env with fields}
  | TIVariant {ti_constrs; ti_name; _} ->
      let env = {env with types = StringMap.add name info env.types} in
      (* Also add constructor mappings *)
      let constrs =
        List.fold_left
          (fun constrs (cname, carg) ->
            StringMap.add cname (ti_name, carg) constrs)
          env.constructors
          ti_constrs
      in
      {env with constructors = constrs}

let add_intrinsic_fun name info env =
  {env with intrinsic_funs = StringMap.add name info env.intrinsic_funs}

let add_intrinsic_const name info env =
  {env with intrinsic_consts = StringMap.add name info env.intrinsic_consts}

let add_local_fun name typ env =
  {env with local_funs = StringMap.add name (name, typ) env.local_funs}

let find_var name env = StringMap.find_opt name env.vars

let find_type name env = StringMap.find_opt name env.types

let find_intrinsic_fun name env = StringMap.find_opt name env.intrinsic_funs

let find_intrinsic_const name env = StringMap.find_opt name env.intrinsic_consts

let find_constructor name env = StringMap.find_opt name env.constructors

let find_field name env = StringMap.find_opt name env.fields

let find_local_fun name env = StringMap.find_opt name env.local_funs

(** Scope management *)
let enter_level env = {env with current_level = env.current_level + 1}

let exit_level env = {env with current_level = max 0 (env.current_level - 1)}

(** Create standard library environment with GPU intrinsics *)
let with_stdlib env =
  (* Helper functions to create intrinsic refs for stdlib modules.
     Sarek_stdlib is a wrapped library, so modules are accessed as Sarek_stdlib.Float32, etc. *)
  let float32_ref name = IntrinsicRef (["Sarek_stdlib"; "Float32"], name) in
  let float64_ref name = IntrinsicRef (["Sarek_stdlib"; "Float64"], name) in
  let int32_ref name = IntrinsicRef (["Sarek_stdlib"; "Int32"], name) in
  let _int64_ref name = IntrinsicRef (["Sarek_stdlib"; "Int64"], name) in
  (* GPU intrinsics that don't belong to a type-specific module.
     These reference Sarek.Sarek_prim for now - will be moved to Gpu.ml later *)
  let gpu_ref name = IntrinsicRef (["Sarek"; "Sarek_prim"], name) in

  (* Std module constants *)
  let env =
    add_intrinsic_const
      "thread_idx_x"
      {
        const_type = t_int32;
        const_cuda = "threadIdx.x";
        const_opencl = "get_local_id(0)";
      }
      env
  in
  let env =
    add_intrinsic_const
      "thread_idx_y"
      {
        const_type = t_int32;
        const_cuda = "threadIdx.y";
        const_opencl = "get_local_id(1)";
      }
      env
  in
  let env =
    add_intrinsic_const
      "thread_idx_z"
      {
        const_type = t_int32;
        const_cuda = "threadIdx.z";
        const_opencl = "get_local_id(2)";
      }
      env
  in

  let env =
    add_intrinsic_const
      "block_idx_x"
      {
        const_type = t_int32;
        const_cuda = "blockIdx.x";
        const_opencl = "get_group_id(0)";
      }
      env
  in
  let env =
    add_intrinsic_const
      "block_idx_y"
      {
        const_type = t_int32;
        const_cuda = "blockIdx.y";
        const_opencl = "get_group_id(1)";
      }
      env
  in
  let env =
    add_intrinsic_const
      "block_idx_z"
      {
        const_type = t_int32;
        const_cuda = "blockIdx.z";
        const_opencl = "get_group_id(2)";
      }
      env
  in

  let env =
    add_intrinsic_const
      "block_dim_x"
      {
        const_type = t_int32;
        const_cuda = "blockDim.x";
        const_opencl = "get_local_size(0)";
      }
      env
  in
  let env =
    add_intrinsic_const
      "block_dim_y"
      {
        const_type = t_int32;
        const_cuda = "blockDim.y";
        const_opencl = "get_local_size(1)";
      }
      env
  in
  let env =
    add_intrinsic_const
      "block_dim_z"
      {
        const_type = t_int32;
        const_cuda = "blockDim.z";
        const_opencl = "get_local_size(2)";
      }
      env
  in

  let env =
    add_intrinsic_const
      "grid_dim_x"
      {
        const_type = t_int32;
        const_cuda = "gridDim.x";
        const_opencl = "get_num_groups(0)";
      }
      env
  in
  let env =
    add_intrinsic_const
      "grid_dim_y"
      {
        const_type = t_int32;
        const_cuda = "gridDim.y";
        const_opencl = "get_num_groups(1)";
      }
      env
  in
  let env =
    add_intrinsic_const
      "grid_dim_z"
      {
        const_type = t_int32;
        const_cuda = "gridDim.z";
        const_opencl = "get_num_groups(2)";
      }
      env
  in

  let env =
    add_intrinsic_const
      "global_thread_id"
      {
        const_type = t_int32;
        const_cuda = "(blockIdx.x*blockDim.x+threadIdx.x)";
        const_opencl = "get_global_id(0)";
      }
      env
  in

  (* GPU synchronization functions *)
  let env =
    add_intrinsic_fun
      "block_barrier"
      {
        intr_type = t_fun [t_unit] t_unit;
        intr_cuda = "__syncthreads()";
        intr_opencl = "barrier(CLK_LOCAL_MEM_FENCE)";
        intr_ocaml = gpu_ref "block_barrier";
      }
      env
  in

  let env =
    add_intrinsic_fun
      "return"
      {
        intr_type = t_fun [t_unit] t_unit;
        intr_cuda = "return";
        intr_opencl = "return";
        intr_ocaml = gpu_ref "return_unit";
      }
      env
  in

  (* Type conversion functions *)
  let env =
    add_intrinsic_fun
      "float"
      {
        intr_type = t_fun [t_int32] t_float32;
        intr_cuda = "(float)";
        intr_opencl = "(float)";
        intr_ocaml = gpu_ref "float_of_int";
      }
      env
  in

  let env =
    add_intrinsic_fun
      "float64"
      {
        intr_type = t_fun [t_int32] t_float64;
        intr_cuda = "(double)";
        intr_opencl = "(double)";
        intr_ocaml = gpu_ref "float64_of_int";
      }
      env
  in

  let env =
    add_intrinsic_fun
      "int_of_float"
      {
        intr_type = t_fun [t_float32] t_int32;
        intr_cuda = "(int)";
        intr_opencl = "(int)";
        intr_ocaml = gpu_ref "int_of_float";
      }
      env
  in

  let env =
    add_intrinsic_fun
      "int_of_float64"
      {
        intr_type = t_fun [t_float64] t_int32;
        intr_cuda = "(int)";
        intr_opencl = "(int)";
        intr_ocaml = gpu_ref "int_of_float64";
      }
      env
  in

  (* Math.Float32 unary functions *)
  let float32_unary_funs =
    [
      ("sin", "sinf", "sin");
      ("cos", "cosf", "cos");
      ("tan", "tanf", "tan");
      ("asin", "asinf", "asin");
      ("acos", "acosf", "acos");
      ("atan", "atanf", "atan");
      ("sinh", "sinhf", "sinh");
      ("cosh", "coshf", "cosh");
      ("tanh", "tanhf", "tanh");
      ("exp", "expf", "exp");
      ("log", "logf", "log");
      ("log10", "log10f", "log10");
      ("sqrt", "sqrtf", "sqrt");
      ("ceil", "ceilf", "ceil");
      ("floor", "floorf", "floor");
      ("expm1", "expm1f", "expm1");
      ("log1p", "log1pf", "log1p");
      ("abs_float", "fabsf", "fabs");
      ("rsqrt", "rsqrtf", "rsqrt");
    ]
  in
  let env =
    List.fold_left
      (fun env (name, cuda, opencl) ->
        add_intrinsic_fun
          name
          {
            intr_type = t_fun [t_float32] t_float32;
            intr_cuda = cuda;
            intr_opencl = opencl;
            intr_ocaml = float32_ref name;
          }
          env)
      env
      float32_unary_funs
  in

  (* Math.Float32 binary functions *)
  let float32_bin_funs =
    [
      ("pow", "powf", "pow");
      ("atan2", "atan2f", "atan2");
      ("hypot", "hypotf", "hypot");
      ("copysign", "copysignf", "copysign");
    ]
  in
  let env =
    List.fold_left
      (fun env (name, cuda, opencl) ->
        add_intrinsic_fun
          name
          {
            intr_type = t_fun [t_float32; t_float32] t_float32;
            intr_cuda = cuda;
            intr_opencl = opencl;
            intr_ocaml = float32_ref name;
          }
          env)
      env
      float32_bin_funs
  in

  (* Math.Float32 binary functions - GPU specific (spoc helpers) *)
  let float32_bin_funs_gpu =
    [
      ("add", "spoc_fadd", "spoc_fadd", "add_float32");
      ("minus", "spoc_fminus", "spoc_fminus", "sub_float32");
      ("mul", "spoc_fmul", "spoc_fmul", "mul_float32");
      ("div", "spoc_fdiv", "spoc_fdiv", "div_float32");
    ]
  in
  let env =
    List.fold_left
      (fun env (name, cuda, opencl, stdlib_name) ->
        add_intrinsic_fun
          name
          {
            intr_type = t_fun [t_float32; t_float32] t_float32;
            intr_cuda = cuda;
            intr_opencl = opencl;
            intr_ocaml = float32_ref stdlib_name;
          }
          env)
      env
      float32_bin_funs_gpu
  in

  (* Math.Float64 unary functions - mapped to Float64.stdlib_name *)
  let float64_unary_funs =
    [
      ("sin64", "sin", "sin", "sin");
      ("cos64", "cos", "cos", "cos");
      ("tan64", "tan", "tan", "tan");
      ("asin64", "asin", "asin", "asin");
      ("acos64", "acos", "acos", "acos");
      ("atan64", "atan", "atan", "atan");
      ("sinh64", "sinh", "sinh", "sinh");
      ("cosh64", "cosh", "cosh", "cosh");
      ("tanh64", "tanh", "tanh", "tanh");
      ("exp64", "exp", "exp", "exp");
      ("log64", "log", "log", "log");
      ("log1064", "log10", "log10", "log10");
      ("sqrt64", "sqrt", "sqrt", "sqrt");
      ("ceil64", "ceil", "ceil", "ceil");
      ("floor64", "floor", "floor", "floor");
      ("abs_float64", "fabs", "fabs", "abs_float");
      ("rsqrt64", "rsqrt", "rsqrt", "rsqrt");
    ]
  in
  let env =
    List.fold_left
      (fun env (name, cuda, opencl, stdlib_name) ->
        add_intrinsic_fun
          name
          {
            intr_type = t_fun [t_float64] t_float64;
            intr_cuda = cuda;
            intr_opencl = opencl;
            intr_ocaml = float64_ref stdlib_name;
          }
          env)
      env
      float64_unary_funs
  in

  (* Math.Float64 binary functions *)
  let float64_bin_funs =
    [
      ("pow64", "pow", "pow", "pow");
      ("atan264", "atan2", "atan2", "atan2");
      ("hypot64", "hypot", "hypot", "hypot");
      ("copysign64", "copysign", "copysign", "copysign");
    ]
  in
  let env =
    List.fold_left
      (fun env (name, cuda, opencl, stdlib_name) ->
        add_intrinsic_fun
          name
          {
            intr_type = t_fun [t_float64; t_float64] t_float64;
            intr_cuda = cuda;
            intr_opencl = opencl;
            intr_ocaml = float64_ref stdlib_name;
          }
          env)
      env
      float64_bin_funs
  in

  (* Math.Float64 binary functions - GPU specific (spoc helpers) *)
  let float64_bin_funs_gpu =
    [
      ("add64", "spoc_dadd", "spoc_dadd", "add_float64");
      ("minus64", "spoc_dminus", "spoc_dminus", "sub_float64");
      ("mul64", "spoc_dmul", "spoc_dmul", "mul_float64");
      ("div64", "spoc_ddiv", "spoc_ddiv", "div_float64");
    ]
  in
  let env =
    List.fold_left
      (fun env (name, cuda, opencl, stdlib_name) ->
        add_intrinsic_fun
          name
          {
            intr_type = t_fun [t_float64; t_float64] t_float64;
            intr_cuda = cuda;
            intr_opencl = opencl;
            intr_ocaml = float64_ref stdlib_name;
          }
          env)
      env
      float64_bin_funs_gpu
  in

  (* Integer math functions *)
  let env =
    add_intrinsic_fun
      "logical_and"
      {
        intr_type = t_fun [t_int32; t_int32] t_int32;
        intr_cuda = "logical_and";
        intr_opencl = "logical_and";
        intr_ocaml = int32_ref "logand";
      }
      env
  in

  let env =
    add_intrinsic_fun
      "xor"
      {
        intr_type = t_fun [t_int32; t_int32] t_int32;
        intr_cuda = "spoc_xor";
        intr_opencl = "spoc_xor";
        intr_ocaml = int32_ref "logxor";
      }
      env
  in

  let env =
    add_intrinsic_fun
      "spoc_powint"
      {
        intr_type = t_fun [t_int32; t_int32] t_int32;
        intr_cuda = "spoc_powint";
        intr_opencl = "spoc_powint";
        (* No stdlib equivalent yet - keep in gpu_ref for now *)
        intr_ocaml = gpu_ref "spoc_powint";
      }
      env
  in

  env

(** Lookup that checks all namespaces for an identifier *)
type lookup_result =
  | LVar of var_info
  | LIntrinsicConst of intrinsic_const_info
  | LIntrinsicFun of intrinsic_fun_info
  | LConstructor of string * typ option  (** type_name, arg_type *)
  | LLocalFun of string * typ
  | LNotFound

let lookup name env =
  match find_var name env with
  | Some info -> LVar info
  | None -> (
      match find_intrinsic_const name env with
      | Some info -> LIntrinsicConst info
      | None -> (
          match find_intrinsic_fun name env with
          | Some info -> LIntrinsicFun info
          | None -> (
              match find_constructor name env with
              | Some (type_name, arg_type) -> LConstructor (type_name, arg_type)
              | None -> (
                  match find_local_fun name env with
                  | Some (name, typ) -> LLocalFun (name, typ)
                  | None -> LNotFound))))

(** Debug: print environment contents *)
let pp_env fmt env =
  Format.fprintf fmt "Variables:@." ;
  StringMap.iter
    (fun name info ->
      Format.fprintf
        fmt
        "  %s : %a (param=%b, idx=%d)@."
        name
        pp_typ
        info.vi_type
        info.vi_is_param
        info.vi_index)
    env.vars ;
  Format.fprintf fmt "Intrinsic constants:@." ;
  StringMap.iter
    (fun name info ->
      Format.fprintf
        fmt
        "  %s : %a (cuda=%s, opencl=%s)@."
        name
        pp_typ
        info.const_type
        info.const_cuda
        info.const_opencl)
    env.intrinsic_consts ;
  Format.fprintf fmt "Intrinsic functions:@." ;
  StringMap.iter
    (fun name info ->
      Format.fprintf
        fmt
        "  %s : %a (cuda=%s, opencl=%s)@."
        name
        pp_typ
        info.intr_type
        info.intr_cuda
        info.intr_opencl)
    env.intrinsic_funs
