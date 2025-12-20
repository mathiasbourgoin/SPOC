(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module defines the GPU intrinsics (thread indices, math functions, etc.)
 * and their mappings to CUDA and OpenCL code.
 ******************************************************************************)

open Sarek_types

(** Module paths for intrinsics lookup *)
type module_path = string list

(** Intrinsic definition *)
type intrinsic = {
  i_name: string;
  i_type: typ;
  i_cuda: string;
  i_opencl: string;
  i_module: module_path;  (** Which module this belongs to, e.g., ["Std"] or ["Math"; "Float32"] *)
}

(** All standard intrinsics *)
let std_consts : intrinsic list = [
  (* Thread indices *)
  { i_name = "thread_idx_x"; i_type = t_int32;
    i_cuda = "threadIdx.x"; i_opencl = "get_local_id(0)"; i_module = ["Std"] };
  { i_name = "thread_idx_y"; i_type = t_int32;
    i_cuda = "threadIdx.y"; i_opencl = "get_local_id(1)"; i_module = ["Std"] };
  { i_name = "thread_idx_z"; i_type = t_int32;
    i_cuda = "threadIdx.z"; i_opencl = "get_local_id(2)"; i_module = ["Std"] };

  (* Block indices *)
  { i_name = "block_idx_x"; i_type = t_int32;
    i_cuda = "blockIdx.x"; i_opencl = "get_group_id(0)"; i_module = ["Std"] };
  { i_name = "block_idx_y"; i_type = t_int32;
    i_cuda = "blockIdx.y"; i_opencl = "get_group_id(1)"; i_module = ["Std"] };
  { i_name = "block_idx_z"; i_type = t_int32;
    i_cuda = "blockIdx.z"; i_opencl = "get_group_id(2)"; i_module = ["Std"] };

  (* Block dimensions *)
  { i_name = "block_dim_x"; i_type = t_int32;
    i_cuda = "blockDim.x"; i_opencl = "get_local_size(0)"; i_module = ["Std"] };
  { i_name = "block_dim_y"; i_type = t_int32;
    i_cuda = "blockDim.y"; i_opencl = "get_local_size(1)"; i_module = ["Std"] };
  { i_name = "block_dim_z"; i_type = t_int32;
    i_cuda = "blockDim.z"; i_opencl = "get_local_size(2)"; i_module = ["Std"] };

  (* Grid dimensions *)
  { i_name = "grid_dim_x"; i_type = t_int32;
    i_cuda = "gridDim.x"; i_opencl = "get_num_groups(0)"; i_module = ["Std"] };
  { i_name = "grid_dim_y"; i_type = t_int32;
    i_cuda = "gridDim.y"; i_opencl = "get_num_groups(1)"; i_module = ["Std"] };
  { i_name = "grid_dim_z"; i_type = t_int32;
    i_cuda = "gridDim.z"; i_opencl = "get_num_groups(2)"; i_module = ["Std"] };

  (* Convenience *)
  { i_name = "global_thread_id"; i_type = t_int32;
    i_cuda = "(blockIdx.x*blockDim.x+threadIdx.x)";
    i_opencl = "get_global_id(0)"; i_module = ["Std"] };
]

let std_funs : intrinsic list = [
  (* Synchronization *)
  { i_name = "block_barrier"; i_type = t_fun [t_unit] t_unit;
    i_cuda = "__syncthreads()"; i_opencl = "barrier(CLK_LOCAL_MEM_FENCE)";
    i_module = ["Std"] };

  (* Type conversions *)
  { i_name = "float"; i_type = t_fun [t_int32] t_float32;
    i_cuda = "(float)"; i_opencl = "(float)"; i_module = ["Std"] };
  { i_name = "float64"; i_type = t_fun [t_int32] t_float64;
    i_cuda = "(double)"; i_opencl = "(double)"; i_module = ["Std"] };
  { i_name = "int_of_float"; i_type = t_fun [t_float32] t_int32;
    i_cuda = "(int)"; i_opencl = "(int)"; i_module = ["Std"] };
  { i_name = "int_of_float64"; i_type = t_fun [t_float64] t_int32;
    i_cuda = "(int)"; i_opencl = "(int)"; i_module = ["Std"] };
]

(** Float32 math functions *)
let math_float32_funs : intrinsic list =
  let unary name cuda opencl = {
    i_name = name;
    i_type = t_fun [t_float32] t_float32;
    i_cuda = cuda;
    i_opencl = opencl;
    i_module = ["Math"; "Float32"];
  } in
  let binary name cuda opencl = {
    i_name = name;
    i_type = t_fun [t_float32; t_float32] t_float32;
    i_cuda = cuda;
    i_opencl = opencl;
    i_module = ["Math"; "Float32"];
  } in
  [
    (* Unary functions *)
    unary "sin" "sinf" "sin";
    unary "cos" "cosf" "cos";
    unary "tan" "tanf" "tan";
    unary "asin" "asinf" "asin";
    unary "acos" "acosf" "acos";
    unary "atan" "atanf" "atan";
    unary "sinh" "sinhf" "sinh";
    unary "cosh" "coshf" "cosh";
    unary "tanh" "tanhf" "tanh";
    unary "exp" "expf" "exp";
    unary "log" "logf" "log";
    unary "log10" "log10f" "log10";
    unary "expm1" "expm1f" "expm1";
    unary "log1p" "log1pf" "log1p";
    unary "sqrt" "sqrtf" "sqrt";
    unary "rsqrt" "rsqrtf" "rsqrt";
    unary "ceil" "ceilf" "ceil";
    unary "floor" "floorf" "floor";
    unary "abs_float" "fabsf" "fabs";

    (* Binary functions *)
    binary "add" "spoc_fadd" "spoc_fadd";
    binary "minus" "spoc_fminus" "spoc_fminus";
    binary "mul" "spoc_fmul" "spoc_fmul";
    binary "div" "spoc_fdiv" "spoc_fdiv";
    binary "pow" "powf" "pow";
    binary "atan2" "atan2f" "atan2";
    binary "hypot" "hypotf" "hypot";
    binary "copysign" "copysignf" "copysign";
  ]

(** Float64 math functions *)
let math_float64_funs : intrinsic list =
  let unary name cuda opencl = {
    i_name = name;
    i_type = t_fun [t_float64] t_float64;
    i_cuda = cuda;
    i_opencl = opencl;
    i_module = ["Math"; "Float64"];
  } in
  let binary name cuda opencl = {
    i_name = name;
    i_type = t_fun [t_float64; t_float64] t_float64;
    i_cuda = cuda;
    i_opencl = opencl;
    i_module = ["Math"; "Float64"];
  } in
  [
    (* Unary functions *)
    unary "sin" "sin" "sin";
    unary "cos" "cos" "cos";
    unary "tan" "tan" "tan";
    unary "asin" "asin" "asin";
    unary "acos" "acos" "acos";
    unary "atan" "atan" "atan";
    unary "sinh" "sinh" "sinh";
    unary "cosh" "cosh" "cosh";
    unary "tanh" "tanh" "tanh";
    unary "exp" "exp" "exp";
    unary "log" "log" "log";
    unary "log10" "log10" "log10";
    unary "expm1" "expm1" "expm1";
    unary "log1p" "log1p" "log1p";
    unary "sqrt" "sqrt" "sqrt";
    unary "rsqrt" "rsqrt" "rsqrt";
    unary "ceil" "ceil" "ceil";
    unary "floor" "floor" "floor";
    unary "abs_float" "fabs" "fabs";
    unary "of_float32" "" "";  (* Identity cast *)
    unary "of_float" "" "";    (* Identity cast *)
    unary "to_float32" "(float)" "(float)";

    (* Binary functions *)
    binary "add" "spoc_dadd" "spoc_dadd";
    binary "minus" "spoc_dminus" "spoc_dminus";
    binary "mul" "spoc_dmul" "spoc_dmul";
    binary "div" "spoc_ddiv" "spoc_ddiv";
    binary "pow" "pow" "pow";
    binary "atan2" "atan2" "atan2";
    binary "hypot" "hypot" "hypot";
    binary "copysign" "copysign" "copysign";
  ]

(** Integer math functions *)
let math_int_funs : intrinsic list = [
  { i_name = "pow";
    i_type = t_fun [t_int32; t_int32] t_int32;
    i_cuda = "spoc_powint"; i_opencl = "spoc_powint";
    i_module = ["Math"] };
  { i_name = "logical_and";
    i_type = t_fun [t_int32; t_int32] t_int32;
    i_cuda = "logical_and"; i_opencl = "logical_and";
    i_module = ["Math"] };
  { i_name = "xor";
    i_type = t_fun [t_int32; t_int32] t_int32;
    i_cuda = "spoc_xor"; i_opencl = "spoc_xor";
    i_module = ["Math"] };
]

(** Sarek_vector functions *)
let vector_funs : intrinsic list = [
  (* Vector.length - handled specially at runtime *)
]

(** All intrinsics combined *)
let all_intrinsics =
  std_consts @ std_funs @ math_float32_funs @ math_float64_funs @ math_int_funs @ vector_funs

(** Find an intrinsic by name in a specific module path *)
let find_in_module (path : module_path) (name : string) : intrinsic option =
  List.find_opt (fun i -> i.i_name = name && i.i_module = path) all_intrinsics

(** Find an intrinsic by name, searching all modules *)
let find_any (name : string) : intrinsic option =
  List.find_opt (fun i -> i.i_name = name) all_intrinsics

(** Check if a name is a known intrinsic constant *)
let is_intrinsic_const (name : string) : bool =
  List.exists (fun i -> i.i_name = name) std_consts

(** Check if a name is a known intrinsic function *)
let is_intrinsic_fun (name : string) : bool =
  let all_funs = std_funs @ math_float32_funs @ math_float64_funs @ math_int_funs in
  List.exists (fun i -> i.i_name = name) all_funs

(** Module path mappings for open statements *)
let module_aliases : (string list * module_path) list = [
  (["Std"], ["Std"]);
  (["Math"], ["Math"]);
  (["Math"; "Float32"], ["Math"; "Float32"]);
  (["Math"; "Float64"], ["Math"; "Float64"]);
  (["Float32"], ["Math"; "Float32"]);  (* Alias *)
  (["Float64"], ["Math"; "Float64"]);  (* Alias *)
  (["Sarek_vector"], ["Sarek_vector"]);
]

(** Resolve a module path, following aliases *)
let resolve_module_path (path : string list) : module_path option =
  List.assoc_opt path module_aliases
