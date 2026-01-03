type float64 = float

type float32 = float

type extension = ExFloat32 | ExFloat64

type ('a, 'b, 'c) kirc_kernel = {
  ml_kern : 'a;
  body : Sarek.Kirc_Ast.k_ext;
  body_v2 : Sarek.Sarek_ir.kernel option;
  ret_val : Sarek.Kirc_Ast.k_ext * ('b, 'c) Spoc.Vector.kind;
  extensions : extension array;
  cpu_kern :
    (mode:Sarek.Sarek_cpu_runtime.exec_mode ->
    block:int * int * int ->
    grid:int * int * int ->
    Obj.t array ->
    unit)
    option;
}

type ('a, 'b, 'c, 'd) kirc_function = {
  fun_name : string;
  ml_fun : 'a;
  funbody : Sarek.Kirc_Ast.k_ext;
  fun_ret : Sarek.Kirc_Ast.k_ext * ('b, 'c) Spoc.Vector.kind;
  fastflow_acc : 'd;
  fun_extensions : extension array;
}

type ('a, 'b, 'c, 'd, 'e) sarek_kernel =
  ('a, 'b) Spoc.Kernel.spoc_kernel * ('c, 'd, 'e) kirc_kernel

val constructors : string list ref

val eint32 : Sarek.Kirc_Ast.elttype

val eint64 : Sarek.Kirc_Ast.elttype

val efloat32 : Sarek.Kirc_Ast.elttype

val efloat64 : Sarek.Kirc_Ast.elttype

val global : Sarek.Kirc_Ast.memspace

val local : Sarek.Kirc_Ast.memspace

val shared : Sarek.Kirc_Ast.memspace

val print_ast : Sarek.Kirc_Ast.k_ext -> unit

val opencl_head : string

val opencl_float64 : string

val cuda_float64 : string

val cuda_head : string

val register_constructor_string : string -> unit

val new_var : int -> Sarek.Kirc_Ast.k_ext

val global_fun : ('a, 'b, 'c, 'd) kirc_function -> Sarek.Kirc_Ast.k_ext

val new_array :
  string ->
  Sarek.Kirc_Ast.k_ext ->
  Sarek.Kirc_Ast.elttype ->
  Sarek.Kirc_Ast.memspace ->
  Sarek.Kirc_Ast.k_ext

val var : int -> string -> Sarek.Kirc_Ast.k_ext

val spoc_gen_kernel : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_fun_kernel : 'a -> 'b -> unit

val seq : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val app : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext array -> Sarek.Kirc_Ast.k_ext

val spoc_unit : unit -> Sarek.Kirc_Ast.k_ext

val spoc_int : int -> Sarek.Kirc_Ast.k_ext

val global_int_var : (unit -> int32) -> Sarek.Kirc_Ast.k_ext

val global_float_var : (unit -> float) -> Sarek.Kirc_Ast.k_ext

val global_float64_var : (unit -> float) -> Sarek.Kirc_Ast.k_ext

val spoc_int32 : int32 -> Sarek.Kirc_Ast.k_ext

val spoc_float : float -> Sarek.Kirc_Ast.k_ext

val spoc_double : float -> Sarek.Kirc_Ast.k_ext

val spoc_int_id : int -> Sarek.Kirc_Ast.k_ext

val spoc_float_id : float -> Sarek.Kirc_Ast.k_ext

val spoc_plus : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_plus_float : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_min : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_min_float : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_mul : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_mul_float : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_div : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_div_float : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_mod : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_ife :
  Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_if : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_match :
  string -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.case array -> Sarek.Kirc_Ast.k_ext

val spoc_case :
  int ->
  (string * string * int * string) option ->
  Sarek.Kirc_Ast.k_ext ->
  Sarek.Kirc_Ast.case

val spoc_do :
  Sarek.Kirc_Ast.k_ext ->
  Sarek.Kirc_Ast.k_ext ->
  Sarek.Kirc_Ast.k_ext ->
  Sarek.Kirc_Ast.k_ext ->
  Sarek.Kirc_Ast.k_ext

val spoc_while : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val params : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_id : 'a -> Sarek.Kirc_Ast.k_ext

val spoc_constr : string -> string -> Sarek.Kirc_Ast.k_ext list -> Sarek.Kirc_Ast.k_ext

val spoc_record : string -> Sarek.Kirc_Ast.k_ext list -> Sarek.Kirc_Ast.k_ext

val spoc_rec_get : Sarek.Kirc_Ast.k_ext -> string -> Sarek.Kirc_Ast.k_ext

val spoc_rec_set : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_return : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val concat : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val empty_arg : unit -> Sarek.Kirc_Ast.k_ext

val new_int_var : ?mutable_:bool -> int -> string -> Sarek.Kirc_Ast.k_ext

val new_float_var : ?mutable_:bool -> int -> string -> Sarek.Kirc_Ast.k_ext

val new_float64_var : ?mutable_:bool -> int -> string -> Sarek.Kirc_Ast.k_ext

val new_double_var : ?mutable_:bool -> int -> string -> Sarek.Kirc_Ast.k_ext

val new_unit_var : ?mutable_:bool -> int -> string -> Sarek.Kirc_Ast.k_ext

val new_custom_var : string -> int -> string -> Sarek.Kirc_Ast.k_ext

val new_int_vec_var : int -> string -> Sarek.Kirc_Ast.k_ext

val new_float_vec_var : int -> string -> Sarek.Kirc_Ast.k_ext

val new_double_vec_var : int -> string -> Sarek.Kirc_Ast.k_ext

val new_custom_vec_var : string -> int -> string -> Sarek.Kirc_Ast.k_ext

val int_vect : int -> Sarek.Kirc_Ast.kvect

val set_vect_var : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val set_arr_var : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val intrinsics : string -> string -> Sarek.Kirc_Ast.k_ext

val spoc_local_env : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_set : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_declare : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_local_var : 'a -> 'a

val spoc_acc : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val int_var : 'a -> 'a

val int32_var : 'a -> 'a

val float_var : 'a -> 'a

val double_var : int -> Sarek.Kirc_Ast.k_ext

val equals : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val equals_custom : string -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val equals32 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val equals64 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val equalsF : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val equalsF64 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val b_or : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val b_and : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val b_not : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val lt : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val lt32 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val lt64 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val ltF : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val ltF64 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val gt : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val gt32 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val gt64 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val gtF : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val gtF64 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val lte : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val lte32 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val lte64 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val lteF : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val lteF64 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val gte : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val gte32 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val gte64 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val gteF : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val gteF64 : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val get_vec : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val get_arr : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val return_unit : unit -> Sarek.Kirc_Ast.k_ext

val return_int : int -> string -> Sarek.Kirc_Ast.k_ext

val return_float : int -> string -> Sarek.Kirc_Ast.k_ext

val return_double : int -> string -> Sarek.Kirc_Ast.k_ext

val return_bool : int -> string -> Sarek.Kirc_Ast.k_ext

val return_custom : string -> string -> string -> Sarek.Kirc_Ast.k_ext

val rewrite : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val spoc_native : (Spoc.Devices.device -> string) -> Sarek.Kirc_Ast.k_ext

val pragma : string list -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val return_v : (string * string) ref

val save : string -> string -> unit

val load_file : string -> bytes

val map : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext

val gen_profile :
  ('a, 'b, 'c, 'd, 'e) sarek_kernel -> Spoc.Devices.device -> unit

val opencl_source :
  ?profile:bool ->
  ?return:bool ->
  ('a, 'b, 'c, 'd, 'e) sarek_kernel ->
  Spoc.Devices.device ->
  string

val gen :
  ?keep_temp:bool ->
  ?profile:bool ->
  ?return:bool ->
  ?only:Spoc.Devices.specificLibrary ->
  ?nvrtc_options:string array ->
  ('a, 'b, 'c, 'd, 'e) sarek_kernel ->
  Spoc.Devices.device ->
  ('a, 'b, 'c, 'd, 'e) sarek_kernel

val run :
  ?recompile:bool ->
  ('a, ('b, 'f) Spoc.Kernel.kernelArgs array, 'c, 'd, 'e) sarek_kernel ->
  'a ->
  Spoc.Kernel.block * Spoc.Kernel.grid ->
  int ->
  Spoc.Devices.device ->
  unit

(** Flush device, handling fission queue for native fission devices. This wraps
    Devices.flush and adds fission queue synchronization. Use this instead of
    Devices.flush when using the fission device. *)
val flush : Spoc.Devices.device -> ?queue_id:int -> unit -> unit

val profile_run :
  ?recompile:bool ->
  ('a, ('b, 'f) Spoc.Kernel.kernelArgs array, 'c, 'd, 'e) sarek_kernel ->
  'a ->
  Spoc.Kernel.block * Spoc.Kernel.grid ->
  int ->
  Spoc.Devices.device ->
  unit

val compile_kernel_to_files :
  string -> ('a, 'b, 'c, 'd, 'e) sarek_kernel -> Spoc.Devices.device -> unit

module Std : sig
  val thread_idx_x : Int32.t

  val thread_idx_y : Int32.t

  val thread_idx_z : Int32.t

  val block_idx_x : Int32.t

  val block_idx_y : Int32.t

  val block_idx_z : Int32.t

  val block_dim_x : Int32.t

  val block_dim_y : Int32.t

  val block_dim_z : Int32.t

  val grid_dim_x : Int32.t

  val grid_dim_y : Int32.t

  val grid_dim_z : Int32.t

  val global_thread_id : Int32.t

  val return : unit -> unit

  val float64 : Int32.t -> float

  val int_of_float64 : float -> Int32.t

  val float : Int32.t -> float

  val int_of_float : float -> Int32.t

  val block_barrier : unit -> unit

  val make_shared : Int32.t -> Int32.t array

  val make_local : Int32.t -> Int32.t array

  val map :
    ('a -> 'b) ->
    (*, 'c, 'd, 'e) kirc_function -> *)
    ('a, 'f) Spoc.Vector.vector ->
    ('b, 'g) Spoc.Vector.vector ->
    unit

  val reduce :
    ('a -> 'a -> 'a) ->
    (*, 'c, 'd, 'e) kirc_function -> *)
    ('a, 'f) Spoc.Vector.vector ->
    ('a, 'g) Spoc.Vector.vector ->
    unit
end

module Sarek_vector : sig
  val length : ('a, 'b) Spoc.Vector.vector -> int32
end

module Math : sig
  val pow : Int32.t -> Int32.t -> Int32.t

  val logical_and : Int32.t -> Int32.t -> Int32.t

  val xor : Int32.t -> Int32.t -> Int32.t

  module Float32 : sig
    val add : float -> float -> float

    val minus : float -> float -> float

    val mul : float -> float -> float

    val div : float -> float -> float

    val pow : float -> float -> float

    val sqrt : float -> float

    val rsqrt : float -> float

    val exp : float -> float

    val log : float -> float

    val log10 : float -> float

    val expm1 : float -> float

    val log1p : float -> float

    val acos : float -> float

    val cos : float -> float

    val cosh : float -> float

    val asin : float -> float

    val sin : float -> float

    val sinh : float -> float

    val tan : float -> float

    val tanh : float -> float

    val atan : float -> float

    val atan2 : float -> float -> float

    val hypot : float -> float -> float

    val ceil : float -> float

    val floor : float -> float

    val abs_float : float -> float

    val copysign : float -> float -> float

    val modf : float -> float * float

    val zero : float

    val one : float

    val make_shared : Int32.t -> float array

    val make_local : Int32.t -> float array
  end

  module Float64 : sig
    val add : float -> float -> float

    val minus : float -> float -> float

    val mul : float -> float -> float

    val div : float -> float -> float

    val pow : float -> float -> float

    val sqrt : float -> float

    val rsqrt : float -> float

    val exp : float -> float

    val log : float -> float

    val log10 : float -> float

    val expm1 : float -> float

    val log1p : float -> float

    val acos : float -> float

    val cos : float -> float

    val cosh : float -> float

    val asin : float -> float

    val sin : float -> float

    val sinh : float -> float

    val tan : float -> float

    val tanh : float -> float

    val atan : float -> float

    val atan2 : float -> float -> float

    val hypot : float -> float -> float

    val ceil : float -> float

    val floor : float -> float

    val abs_float : float -> float

    val copysign : float -> float -> float

    val modf : float -> float * float

    val zero : float

    val one : float

    val of_float32 : float -> float

    val of_float : float -> float

    val to_float32 : float -> float

    val make_shared : Int32.t -> float array

    val make_local : Int32.t -> float array
  end
end

(** Kernel fusion module - fuse producer/consumer kernels *)
module Fusion : sig
  (** Check if two kernel bodies can be fused via an intermediate array *)
  val can_fuse_bodies :
    Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> intermediate:string -> bool

  (** Fuse two kernel bodies, eliminating the intermediate array *)
  val fuse_bodies :
    Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> intermediate:string -> Sarek.Kirc_Ast.k_ext

  (** Fuse two kirc_kernel records *)
  val fuse_kernels :
    ('a, 'b, 'c) kirc_kernel ->
    ('d, 'e, 'f) kirc_kernel ->
    intermediate:string ->
    ('d, 'e, 'f) kirc_kernel

  (** Fuse a pipeline of kernel bodies, returning fused body and eliminated
      arrays *)
  val fuse_pipeline_bodies : Sarek.Kirc_Ast.k_ext list -> Sarek.Kirc_Ast.k_ext * string list
end

(*val a_to_vect : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext
  val a_to_return_vect :
  Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext
  val param_list : int list ref
  val add_to_param_list : int -> unit
  val check_and_transform_to_map : Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext
  val arg_of_vec :
  ('a, 'b) Spoc.Vector.vector -> ('a, 'b) Spoc.Kernel.kernelArgs
  val propagate :
  (Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext) -> Sarek.Kirc_Ast.k_ext -> Sarek.Kirc_Ast.k_ext
  val map :
  ('a, 'b, 'c -> 'i, 'i, 'j) sarek_kernel ->
  ?dev:Spoc.Devices.device ->
  ('g, 'h) Spoc.Vector.vector -> ('i, 'j) Spoc.Vector.vector
  val map2 :
  ('a, 'b, 'c -> 'd -> 'l, 'l, 'm) sarek_kernel ->
  ?dev:Spoc.Devices.device ->
  ('h, 'i) Spoc.Vector.vector ->
  ('j, 'k) Spoc.Vector.vector -> ('l, 'm) Spoc.Vector.vector
*)

(** New runtime integration module.

    Provides integration with the ctypes-based plugin runtime (sarek_core). This
    allows Sarek kernels to run on devices discovered via the new plugin
    architecture, with source code generation from the IR at runtime. *)
module NewRuntime : sig
  (** Generate CUDA source code from a kernel IR *)
  val generate_cuda_source :
    ('a, 'b, 'c, 'd, 'e) sarek_kernel -> Spoc.Devices.device -> string

  (** Generate OpenCL source code from a kernel IR *)
  val generate_opencl_source :
    ('a, 'b, 'c, 'd, 'e) sarek_kernel -> Spoc.Devices.device -> string

  (** Generate source code for the appropriate framework.
      @param framework "CUDA" or "OpenCL" *)
  val generate_source :
    ('a, 'b, 'c, 'd, 'e) sarek_kernel ->
    framework:string ->
    Spoc.Devices.device ->
    string

  (** Check if a device framework is available via the new runtime *)
  val is_new_runtime_device : string -> bool

  (** Run a kernel using the new plugin-based runtime. For vector arguments,
      buffers must be pre-allocated and passed as ArgBuffer. *)
  val run_with_buffers :
    device:Sarek_core.Device.t ->
    name:string ->
    source:string ->
    args:Sarek_core.Runtime.arg list ->
    grid:Sarek_core.Runtime.dims ->
    block:Sarek_core.Runtime.dims ->
    ?shared_mem:int ->
    unit ->
    unit

  (** Get kernel name from a sarek_kernel *)
  val kernel_name : ('a, 'b, 'c, 'd, 'e) sarek_kernel -> string
end
