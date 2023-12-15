open Devices

module F = struct
  exception No_Cuda_Device
  exception ERROR_DEINITIALIZED
  exception ERROR_NOT_INITIALIZED
  exception ERROR_INVALID_CONTEXT
  exception ERROR_INVALID_VALUE
  exception ERROR_OUT_OF_MEMORY
  exception ERROR_INVALID_DEVICE
  exception ERROR_NOT_FOUND
  exception ERROR_FILE_NOT_FOUND
  exception ERROR_UNKNOWN
  exception ERROR_LAUNCH_FAILED
  exception ERROR_LAUNCH_OUT_OF_RESOURCES
  exception ERROR_LAUNCH_TIMEOUT
  exception ERROR_LAUNCH_INCOMPATIBLE_TEXTURING

  let _ =
    Callback.register_exception "no_cuda_device" No_Cuda_Device;
    Callback.register_exception "cuda_error_deinitialized" ERROR_DEINITIALIZED;
    Callback.register_exception "cuda_error_not_initialized"
      ERROR_NOT_INITIALIZED;
    Callback.register_exception "cuda_error_invalid_context"
      ERROR_INVALID_CONTEXT;
    Callback.register_exception "cuda_error_invalid_value" ERROR_INVALID_VALUE;
    Callback.register_exception "cuda_error_out_of_memory" ERROR_OUT_OF_MEMORY;
    Callback.register_exception "cuda_error_invalid_device" ERROR_INVALID_DEVICE;
    Callback.register_exception "cuda_error_not_found" ERROR_NOT_FOUND;
    Callback.register_exception "cuda_error_file_not_found" ERROR_FILE_NOT_FOUND;
    Callback.register_exception "cuda_error_launch_failed" ERROR_LAUNCH_FAILED;
    Callback.register_exception "cuda_error_launch_out_of_resources"
      ERROR_LAUNCH_OUT_OF_RESOURCES;
    Callback.register_exception "cuda_error_launch_timeout" ERROR_LAUNCH_TIMEOUT;
    Callback.register_exception "cuda_error_launch_incompatible_texturing"
      ERROR_LAUNCH_INCOMPATIBLE_TEXTURING;
    Callback.register_exception "cuda_error_unknown" ERROR_UNKNOWN

  let name = "Cuda"
  let version = 1

  module Devices = struct
    external init : unit -> unit = "spoc_cuInit"
    external get_compatible_devices : unit -> int = "spoc_getCudaDevicesCount"
    external get_device : int -> device = "spoc_getCudaDevice"
    external flush : generalInfo -> device -> int -> unit = "spoc_cuda_flush"
    external flush_all : generalInfo -> device -> unit = "spoc_cuda_flush_all"
  end

  module Kernel = struct
    open Kernel

    type extra

    external compile : string -> string -> generalInfo -> kernel
      = "spoc_cuda_compile"

    external debug_compile : string -> string -> generalInfo -> kernel
      = "spoc_cuda_debug_compile"

    external cuda_load_param_vec :
      int ref ->
      extra ->
      Vector.device_vec ->
      ('a, 'b) Vector.vector ->
      device ->
      unit = "spoc_cuda_load_param_vec_b" "spoc_cuda_load_param_vec_n"

    external cuda_custom_load_param_vec :
      int ref -> extra -> Vector.device_vec -> ('a, 'b) Vector.vector -> unit
      = "spoc_cuda_custom_load_param_vec_b" "spoc_cuda_custom_load_param_vec_n"

    external cuda_load_param_int : int ref -> extra -> int -> unit
      = "spoc_cuda_load_param_int_b" "spoc_cuda_load_param_int_n"

    external cuda_load_param_int64 : int ref -> extra -> int -> unit
      = "spoc_cuda_load_param_int64_b" "spoc_cuda_load_param_int64_n"

    external cuda_load_param_float : int ref -> extra -> float -> unit
      = "spoc_cuda_load_param_float_b" "spoc_cuda_load_param_float_n"

    external cuda_load_param_float64 : int ref -> extra -> float -> unit
      = "spoc_cuda_load_param_float64_b" "spoc_cuda_load_param_float64_n"

    let load_arg offset extra dev _cuFun _idx (arg : ('a, 'b) kernelArgs) =
      let load_non_vect = function
        | Int32 i -> cuda_load_param_int offset extra i
        | Int64 i -> cuda_load_param_int64 offset extra i
        | Float32 f -> cuda_load_param_float offset extra f
        | Float64 f -> cuda_load_param_float64 offset extra f
        | _ -> failwith "CU LOAD ARG Type Not Implemented\n"
      and check_vect v =
        (if !Mem.auto then
           try
             Mem.to_device v dev;
             flush dev ()
           with ERROR_OUT_OF_MEMORY -> raise ERROR_OUT_OF_MEMORY);
        match arg with
        | VCustom v2 ->
            cuda_custom_load_param_vec offset extra
              (Vector.device_vec v2 `Cuda dev.general_info.id)
              v
        | _ ->
            cuda_load_param_vec offset extra
              (Vector.device_vec v `Cuda dev.general_info.id)
              v dev
      in
      match arg with
      | VChar v | VFloat32 v | VComplex32 v | VInt32 v | VInt64 v | VFloat64 v
        ->
          check_vect v
      | VCustom (v : ('a, 'b) Vector.vector) -> check_vect v
      | _ -> load_non_vect arg

    external launch_grid :
      int ref -> kernel -> grid -> block -> extra -> generalInfo -> int -> unit
      = "spoc_cuda_launch_grid_b" "spoc_cuda_launch_grid_n"
  end

  module Mem = struct
    exception No_Cuda_Device
    exception ERROR_DEINITIALIZED
    exception ERROR_NOT_INITIALIZED
    exception ERROR_INVALID_CONTEXT
    exception ERROR_INVALID_VALUE
    exception ERROR_OUT_OF_MEMORY
    exception ERROR_INVALID_DEVICE
    exception ERROR_NOT_FOUND
    exception ERROR_FILE_NOT_FOUND
    exception ERROR_UNKNOWN
    exception ERROR_LAUNCH_FAILED
    exception ERROR_LAUNCH_OUT_OF_RESOURCES
    exception ERROR_LAUNCH_TIMEOUT
    exception ERROR_LAUNCH_INCOMPATIBLE_TEXTURING

    let _ =
      Callback.register_exception "no_cuda_device" No_Cuda_Device;
      Callback.register_exception "cuda_error_deinitialized" ERROR_DEINITIALIZED;
      Callback.register_exception "cuda_error_not_initialized"
        ERROR_NOT_INITIALIZED;
      Callback.register_exception "cuda_error_invalid_context"
        ERROR_INVALID_CONTEXT;
      Callback.register_exception "cuda_error_invalid_value" ERROR_INVALID_VALUE;
      Callback.register_exception "cuda_error_out_of_memory" ERROR_OUT_OF_MEMORY;
      Callback.register_exception "cuda_error_invalid_device"
        ERROR_INVALID_DEVICE;
      Callback.register_exception "cuda_error_not_found" ERROR_NOT_FOUND;
      Callback.register_exception "cuda_error_file_not_found"
        ERROR_FILE_NOT_FOUND;
      Callback.register_exception "cuda_error_launch_failed" ERROR_LAUNCH_FAILED;
      Callback.register_exception "cuda_error_launch_out_of_resources"
        ERROR_LAUNCH_OUT_OF_RESOURCES;
      Callback.register_exception "cuda_error_launch_timeout"
        ERROR_LAUNCH_TIMEOUT;
      Callback.register_exception "cuda_error_launch_incompatible_texturing"
        ERROR_LAUNCH_INCOMPATIBLE_TEXTURING;
      Callback.register_exception "cuda_error_unknown" ERROR_UNKNOWN

    external alloc_custom : ('a, 'b) Vector.vector -> int -> generalInfo -> unit
      = "spoc_cuda_custom_alloc_vect"

    external alloc : ('a, 'b) Vector.vector -> int -> generalInfo -> unit
      = "spoc_cuda_alloc_vect"

    external free : ('a, 'b) Vector.vector -> int -> unit
      = "spoc_cuda_free_vect"

    let free v i (_ : generalInfo) = free v i

    external part_host_to_device :
      ('a, 'b) Vector.vector ->
      ('a, 'b) Vector.vector ->
      int ->
      generalInfo ->
      gcInfo ->
      int ->
      int ->
      int ->
      int ->
      int ->
      unit = "spoc_cuda_part_cpu_to_device_b" "spoc_cuda_part_cpu_to_device_n"

    external host_to_device :
      ('a, 'b) Vector.vector -> int -> generalInfo -> gcInfo -> int -> unit
      = "spoc_cuda_cpu_to_device"

    external device_to_device : ('a, 'b) Vector.vector -> int -> device -> unit
      = "spoc_cuda_device_to_device"

    external device_to_host :
      ('a, 'b) Vector.vector -> int -> generalInfo -> device -> int -> unit
      = "spoc_cuda_device_to_cpu"

    external custom_part_host_to_device :
      ('a, 'b) Vector.vector ->
      ('a, 'b) Vector.vector ->
      int ->
      generalInfo ->
      gcInfo ->
      int ->
      int ->
      int ->
      int ->
      int ->
      unit
      = "spoc_cuda_custom_part_cpu_to_device_b"
        "spoc_cuda_custom_part_cpu_to_device_n"

    external custom_host_to_device :
      ('a, 'b) Vector.vector -> int -> generalInfo -> int -> unit
      = "spoc_cuda_custom_cpu_to_device"

    external custom_device_to_host :
      ('a, 'b) Vector.vector -> int -> generalInfo -> int -> unit
      = "spoc_cuda_custom_device_to_cpu"

    let custom_device_to_host v d gI (_ : specificInfo) i =
      custom_device_to_host v d gI i

    external part_device_to_host :
      ('a, 'b) Vector.vector ->
      ('a, 'b) Vector.vector ->
      int ->
      generalInfo ->
      gcInfo ->
      int ->
      int ->
      int ->
      int ->
      int ->
      unit = "spoc_cuda_part_device_to_cpu_b" "spoc_cuda_part_device_to_cpu_n"

    external custom_part_device_to_host :
      ('a, 'b) Vector.vector ->
      ('a, 'b) Vector.vector ->
      int ->
      generalInfo ->
      gcInfo ->
      int ->
      int ->
      int ->
      int ->
      int ->
      unit
      = "spoc_cuda_custom_part_device_to_cpu_b"
        "spoc_cuda_custom_part_device_to_cpu_n"

    external vector_copy :
      ('a, 'b) Vector.vector ->
      int ->
      ('a, 'b) Vector.vector ->
      int ->
      int ->
      generalInfo ->
      int ->
      unit = "spoc_cuda_vector_copy_b" "spoc_cuda_vector_copy_n"

    external custom_vector_copy :
      ('a, 'b) Vector.vector ->
      int ->
      ('a, 'b) Vector.vector ->
      int ->
      int ->
      generalInfo ->
      int ->
      unit = "spoc_cuda_custom_vector_copy_b" "spoc_cuda_custom_vector_copy_n"

    external matrix_copy :
      ('a, 'b) Vector.vector ->
      int ->
      int ->
      int ->
      ('a, 'b) Vector.vector ->
      int ->
      int ->
      int ->
      int ->
      int ->
      generalInfo ->
      int ->
      unit = "spoc_cuda_matrix_copy_b" "spoc_cuda_matrix_copy_n"

    external custom_matrix_copy :
      ('a, 'b) Vector.vector ->
      int ->
      int ->
      int ->
      ('a, 'b) Vector.vector ->
      int ->
      int ->
      int ->
      int ->
      int ->
      generalInfo ->
      int ->
      unit = "spoc_cuda_custom_matrix_copy_b" "spoc_cuda_custom_matrix_copy_n"
  end

  module Vector = struct
    open Vector

    external init_device : unit -> device_vec = "spoc_init_cuda_device_vec"

    external custom_alloc : ('a, 'b) vector -> int -> generalInfo -> unit
      = "spoc_cuda_custom_alloc_vect"

    external alloc : ('a, 'b) vector -> int -> generalInfo -> unit
      = "spoc_cuda_alloc_vect"
  end
end

let _ = Framework_plugins.register (module F)
