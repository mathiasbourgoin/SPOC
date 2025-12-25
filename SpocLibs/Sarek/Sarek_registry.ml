(******************************************************************************
 * Sarek Intrinsics Registry
 *
 * Runtime registry for intrinsic types and functions. The PPX registers
 * intrinsics here, and the code generator looks them up at JIT time.
 ******************************************************************************)

(** Information about an intrinsic type *)
type type_info = {
  ti_name : string;
  ti_device : Spoc.Devices.device -> string;
  ti_size : int; (* bytes *)
}

(** Information about an intrinsic function *)
type fun_info = {
  fi_name : string;
  fi_arity : int;
  fi_device : Spoc.Devices.device -> string;
  fi_arg_types : string list;
  fi_ret_type : string;
}

(** Type registry - maps type names to their info *)
let type_registry : (string, type_info) Hashtbl.t = Hashtbl.create 32

(** Function registry - maps (module_path, name) to their info *)
let fun_registry : (string list * string, fun_info) Hashtbl.t =
  Hashtbl.create 64

(** Register a primitive type *)
let register_type name ~device ~size =
  Hashtbl.replace
    type_registry
    name
    {ti_name = name; ti_device = device; ti_size = size}

(** Register an intrinsic function *)
let register_fun ?(module_path = []) name ~arity ~device ~arg_types ~ret_type =
  Hashtbl.replace
    fun_registry
    (module_path, name)
    {
      fi_name = name;
      fi_arity = arity;
      fi_device = device;
      fi_arg_types = arg_types;
      fi_ret_type = ret_type;
    }

(** Find a type by name *)
let find_type name = Hashtbl.find_opt type_registry name

(** Find a function by name, optionally in a module *)
let find_fun ?(module_path = []) name =
  Hashtbl.find_opt fun_registry (module_path, name)

(** Check if a name is a registered type *)
let is_type name = Hashtbl.mem type_registry name

(** Check if a name is a registered function *)
let is_fun ?(module_path = []) name =
  Hashtbl.mem fun_registry (module_path, name)

(** Get device code for a type *)
let type_device_code name dev =
  match find_type name with
  | Some ti -> ti.ti_device dev
  | None -> failwith ("Unknown intrinsic type: " ^ name)

(** Get device code for a function *)
let fun_device_code ?(module_path = []) name dev =
  match find_fun ~module_path name with
  | Some fi -> fi.fi_device dev
  | None ->
      let path = String.concat "." (module_path @ [name]) in
      failwith ("Unknown intrinsic function: " ^ path)

(******************************************************************************
 * Register standard types
 ******************************************************************************)

let () =
  register_type "float32" ~device:(fun _ -> "float") ~size:4 ;
  register_type "float64" ~device:(fun _ -> "double") ~size:8 ;
  register_type "int32" ~device:(fun _ -> "int") ~size:4 ;
  register_type "int64" ~device:(fun _ -> "long") ~size:8 ;
  register_type "bool" ~device:(fun _ -> "int") ~size:4 ;
  register_type "unit" ~device:(fun _ -> "void") ~size:0

(******************************************************************************
 * Register standard functions
 ******************************************************************************)

let cuda_or_opencl dev cuda_code opencl_code =
  match dev.Spoc.Devices.specific_info with
  | Spoc.Devices.CudaInfo _ -> cuda_code
  | Spoc.Devices.OpenCLInfo _ -> opencl_code

let () =
  (* Thread indices *)
  register_fun
    "thread_idx_x"
    ~arity:0
    ~arg_types:[]
    ~ret_type:"int32"
    ~device:(fun dev -> cuda_or_opencl dev "threadIdx.x" "get_local_id(0)") ;
  register_fun
    "thread_idx_y"
    ~arity:0
    ~arg_types:[]
    ~ret_type:"int32"
    ~device:(fun dev -> cuda_or_opencl dev "threadIdx.y" "get_local_id(1)") ;
  register_fun
    "thread_idx_z"
    ~arity:0
    ~arg_types:[]
    ~ret_type:"int32"
    ~device:(fun dev -> cuda_or_opencl dev "threadIdx.z" "get_local_id(2)") ;

  (* Block indices *)
  register_fun
    "block_idx_x"
    ~arity:0
    ~arg_types:[]
    ~ret_type:"int32"
    ~device:(fun dev -> cuda_or_opencl dev "blockIdx.x" "get_group_id(0)") ;
  register_fun
    "block_idx_y"
    ~arity:0
    ~arg_types:[]
    ~ret_type:"int32"
    ~device:(fun dev -> cuda_or_opencl dev "blockIdx.y" "get_group_id(1)") ;
  register_fun
    "block_idx_z"
    ~arity:0
    ~arg_types:[]
    ~ret_type:"int32"
    ~device:(fun dev -> cuda_or_opencl dev "blockIdx.z" "get_group_id(2)") ;

  (* Block dimensions *)
  register_fun
    "block_dim_x"
    ~arity:0
    ~arg_types:[]
    ~ret_type:"int32"
    ~device:(fun dev -> cuda_or_opencl dev "blockDim.x" "get_local_size(0)") ;
  register_fun
    "block_dim_y"
    ~arity:0
    ~arg_types:[]
    ~ret_type:"int32"
    ~device:(fun dev -> cuda_or_opencl dev "blockDim.y" "get_local_size(1)") ;
  register_fun
    "block_dim_z"
    ~arity:0
    ~arg_types:[]
    ~ret_type:"int32"
    ~device:(fun dev -> cuda_or_opencl dev "blockDim.z" "get_local_size(2)") ;

  (* Grid dimensions *)
  register_fun
    "grid_dim_x"
    ~arity:0
    ~arg_types:[]
    ~ret_type:"int32"
    ~device:(fun dev -> cuda_or_opencl dev "gridDim.x" "get_num_groups(0)") ;
  register_fun
    "grid_dim_y"
    ~arity:0
    ~arg_types:[]
    ~ret_type:"int32"
    ~device:(fun dev -> cuda_or_opencl dev "gridDim.y" "get_num_groups(1)") ;
  register_fun
    "grid_dim_z"
    ~arity:0
    ~arg_types:[]
    ~ret_type:"int32"
    ~device:(fun dev -> cuda_or_opencl dev "gridDim.z" "get_num_groups(2)") ;

  (* Synchronization *)
  register_fun
    "block_barrier"
    ~arity:1
    ~arg_types:["unit"]
    ~ret_type:"unit"
    ~device:(fun dev ->
      cuda_or_opencl dev "__syncthreads()" "barrier(CLK_LOCAL_MEM_FENCE)") ;

  (* Float32 arithmetic operators *)
  register_fun
    "add_float32"
    ~arity:2
    ~arg_types:["float32"; "float32"]
    ~ret_type:"float32"
    ~device:(fun _dev -> "(%s + %s)") ;
  register_fun
    "sub_float32"
    ~arity:2
    ~arg_types:["float32"; "float32"]
    ~ret_type:"float32"
    ~device:(fun _dev -> "(%s - %s)") ;
  register_fun
    "mul_float32"
    ~arity:2
    ~arg_types:["float32"; "float32"]
    ~ret_type:"float32"
    ~device:(fun _dev -> "(%s * %s)") ;
  register_fun
    "div_float32"
    ~arity:2
    ~arg_types:["float32"; "float32"]
    ~ret_type:"float32"
    ~device:(fun _dev -> "(%s / %s)") ;

  (* Int32 arithmetic operators *)
  register_fun
    "add_int32"
    ~arity:2
    ~arg_types:["int32"; "int32"]
    ~ret_type:"int32"
    ~device:(fun _dev -> "(%s + %s)") ;
  register_fun
    "sub_int32"
    ~arity:2
    ~arg_types:["int32"; "int32"]
    ~ret_type:"int32"
    ~device:(fun _dev -> "(%s - %s)") ;
  register_fun
    "mul_int32"
    ~arity:2
    ~arg_types:["int32"; "int32"]
    ~ret_type:"int32"
    ~device:(fun _dev -> "(%s * %s)") ;
  register_fun
    "div_int32"
    ~arity:2
    ~arg_types:["int32"; "int32"]
    ~ret_type:"int32"
    ~device:(fun _dev -> "(%s / %s)") ;
  register_fun
    "mod_int32"
    ~arity:2
    ~arg_types:["int32"; "int32"]
    ~ret_type:"int32"
    ~device:(fun _dev -> "(%s %% %s)")

(* Note: Float32/Int32 math functions (sqrt, sin, abs, etc.) are defined in
     Sarek_stdlib and auto-registered via %sarek_intrinsic when that library
     is loaded. Only core arithmetic operators are defined here. *)
