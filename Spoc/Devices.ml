[@@@ocamlformat "disable"]
(******************************************************************************
 * Mathias Bourgoin, Université Pierre et Marie Curie (2011)
 *
 * Mathias.Bourgoin@gmail.com
 *
 * This software is a computer program whose purpose is to allow
 * GPU programming with the OCaml language.
 *
 * This software is governed by the CeCILL - B license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify	and / or redistribute the software under the terms of the CeCILL - B
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty and the software's author, the holder of the
 * economic rights, and the successive licensors have only limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading, using, modifying and / or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean that it is complicated to manipulate, and that also
 * therefore means that it is reserved for developers and experienced
 * professionals having in - depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and / or
 * data to be ensured and, more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL - B license and that you accept its terms.
 *******************************************************************************)
external cuda_init : unit -> unit = "spoc_cuInit"

external cl_init : unit -> unit = "spoc_clInit"

type dim3 = {x : int; y : int; z : int}

type specificLibrary = Cuda | OpenCL | Both

type context

type generalInfo = {
  name : string;
  totalGlobalMem : int;
  localMemSize : int; (* corresponding cuda name : shared mem per block *)
  clockRate : int;
  totalConstMem : int;
      (* corresponding OpenCL name : MAX_CONSTANT_BUFFER_SIZE *)
  multiProcessorCount : int; (* corresponding OpenCL name : MAX_COMPUTE_UNIT *)
  eccEnabled : bool; (* corresponding OpenCL name : ERROR_CORRECTION_SUPPORT *)
  id : int;
  ctx : context;
}

type cudaInfo = {
  major : int;
  minor : int;
  regsPerBlock : int;
  warpSize : int;
  memPitch : int;
  maxThreadsPerBlock : int;
  maxThreadsDim : dim3;
  maxGridSize : dim3;
  textureAlignment : int;
  deviceOverlap : bool;
  kernelExecTimeoutEnabled : bool;
  integrated : bool;
  canMapHostMemory : bool;
  computeMode : int;
  concurrentKernels : bool;
  pciBusID : int;
  pciDeviceID : int;
  driverVersion : int;
}

type platformInfo = {
  platform_profile : string;
  platform_version : string;
  platform_name : string;
  platform_vendor : string;
  platform_extensions : string;
  num_devices : int;
}

type deviceType =
  | CL_DEVICE_TYPE_CPU
  | CL_DEVICE_TYPE_GPU
  | CL_DEVICE_TYPE_ACCELERATOR
  | CL_DEVICE_TYPE_DEFAULT

type clDeviceFPConfig =
  | CL_FP_DENORM
  | CL_FP_INF_NAN
  | CL_FP_ROUND_TO_NEAREST
  | CL_FP_ROUND_TO_ZERO
  | CL_FP_ROUND_TO_INF
  | CL_FP_FMA
  | CL_FP_NONE

type clDeviceQueueProperties =
  | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
  | CL_QUEUE_PROFILING_ENABLE

type clDeviceGlobalMemCacheType =
  | CL_READ_WRITE_CACHE
  | CL_READ_ONLY_CACHE
  | CL_NONE

type clDeviceLocalMemType = CL_LOCAL | CL_GLOBAL

type clDeviceExecutionCapabilities = CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL

type clDeviceID

type openCLInfo = {
  platform_info : platformInfo;
  device_type : deviceType;
  profile : string;
  version : string;
  vendor : string;
  extensions : string;
  vendor_id : int;
  max_work_item_dimensions : int;
  address_bits : int;
  max_mem_alloc_size : int;
  image_support : bool;
  max_read_image_args : int;
  max_write_image_args : int;
  max_samplers : int;
  mem_base_addr_align : int;
  min_data_type_align_size : int;
  global_mem_cacheline_size : int;
  global_mem_cache_size : int;
  max_constant_args : int;
  endian_little : bool;
  available : bool;
  compiler_available : bool;
  single_fp_config : clDeviceFPConfig;
  global_mem_cache_type : clDeviceGlobalMemCacheType;
  queue_properties : clDeviceQueueProperties;
  local_mem_type : clDeviceLocalMemType;
  double_fp_config : clDeviceFPConfig;
  max_constant_buffer_size : int;
  execution_capabilities : clDeviceExecutionCapabilities;
  half_fp_config : clDeviceFPConfig;
  max_work_group_size : int;
  image2D_max_height : int;
  image2D_max_width : int;
  image3D_max_depth : int;
  image3D_max_height : int;
  image3D_max_width : int;
  max_parameter_size : int;
  max_work_item_size : dim3;
  prefered_vector_width_char : int;
  prefered_vector_width_short : int;
  prefered_vector_width_int : int;
  prefered_vector_width_long : int;
  prefered_vector_width_float : int;
  prefered_vector_width_double : int;
  profiling_timer_resolution : int;
  driver_version : string;
  device_id : clDeviceID;
}

type interpreter_backend =
  | Sequential  (** Deterministic single-threaded, for debugging *)
  | Parallel    (** Multi-threaded using Domains, for performance *)

type interpreterInfo = {
  backend : interpreter_backend;
  num_cores : int;
  debug_mode : bool;
}

(** Native CPU runtime - uses PPX-generated OCaml code *)
type nativeInfo = {
  native_num_cores : int;
  native_parallel : bool;  (** Use parallel execution (Domain per thread) *)
  native_fission : bool;
      (** Use loop fission for supersteps (no effects, just plain loops) *)
}

type specificInfo =
  | CudaInfo of cudaInfo
  | OpenCLInfo of openCLInfo
  | InterpreterInfo of interpreterInfo
  | NativeInfo of nativeInfo

type gcInfo

type events

type device = {
  general_info : generalInfo;
  specific_info : specificInfo;
  gc_info : gcInfo;
  events : events;
}

external get_cuda_compatible_devices : unit -> int = "spoc_getCudaDevicesCount"

external get_opencl_compatible_devices : unit -> int
  = "spoc_getOpenCLDevicesCount"

external get_cuda_device : int -> device = "spoc_getCudaDevice"

external get_opencl_device : int -> int -> device = "spoc_getOpenCLDevice"

let cuda_compatible_devices = ref 0

let opencl_compatible_devices = ref 0

let total_num_devices = ref 0

(******************************************************************************************************)
let openOutput () = ()

let beginEvent _s = 0

let endEvent _s _i = ()

let emitDeviceList _ = ()

#ifdef SPOC_PROFILE

external prePrint : int -> unit = "pre_print_device"

external printInfo :
  string ->
  int ->
  int ->
  int ->
  int ->
  int ->
  bool ->
  int ->
  string ->
  bool ->
  unit = "print_info_bytecode" "print_info_native"

let emitDevice dev printComma =
  let devType =
    begin match dev.specific_info with
    | CudaInfo inf -> "Cuda"
    | OpenCLInfo inf -> "OpenCL"
    end
  in
  let genInf = dev.general_info in
  printInfo
    genInf.name
    genInf.totalGlobalMem
    genInf.localMemSize
    genInf.clockRate
    genInf.totalConstMem
    genInf.multiProcessorCount
    genInf.eccEnabled
    genInf.id
    devType
    printComma

let emitDeviceList devList =
  let nb = List.length devList in
  prePrint nb ;
  List.iteri (fun i dev -> emitDevice dev (i != nb - 1)) devList

external openOutput : unit -> unit = "open_output_profiling"

external closeOutput : unit -> unit = "close_output_profiling"

external beginEvent : string -> int = "begin_event"

external endEvent : string -> int -> unit = "end_event"

#endif
(**********************************************************************************************************)

external is_available : int -> bool = "spoc_opencl_is_available"

(* Counter for interpreter device IDs *)
let interpreter_device_id = ref 0

let create_interpreter_device ?(backend = Sequential) ?(debug = false) () =
  let id = !interpreter_device_id in
  incr interpreter_device_id ;
  let num_cores =
    match backend with
    | Sequential -> 1
    | Parallel -> (try Domain.recommended_domain_count () with _ -> 4)
  in
  let name =
    match backend with
    | Sequential -> "CPU Interpreter (Sequential)"
    | Parallel -> Printf.sprintf "CPU Interpreter (Parallel, %d cores)" num_cores
  in
  {
    general_info = {
      name;
      totalGlobalMem = Sys.max_array_length * 8; (* Approximate *)
      localMemSize = 64 * 1024; (* 64KB shared memory emulation *)
      clockRate = 0; (* Not applicable *)
      totalConstMem = 64 * 1024;
      multiProcessorCount = num_cores;
      eccEnabled = false;
      id;
      (* Use Obj.magic to create dummy opaque types - safe because
         interpreter never passes these to C code *)
      ctx = Obj.magic ();
    };
    specific_info = InterpreterInfo { backend; num_cores; debug_mode = debug };
    (* Dummy values for opaque types - interpreter doesn't use them *)
    gc_info = Obj.magic ();
    events = Obj.magic ();
  }

let interpreter_compatible_devices = ref 0

(* Counter for native device IDs *)
let native_device_id = ref 0

let create_native_device ?(parallel = false) ?(fission = false) () =
  let id = !native_device_id in
  incr native_device_id ;
  let num_cores = try Domain.recommended_domain_count () with _ -> 4 in
  let name =
    match (parallel, fission) with
    | false, _ ->
        Printf.sprintf "CPU Native Runtime Sequential (%d cores)" num_cores
    | true, false ->
        Printf.sprintf "CPU Native Runtime Parallel (%d cores)" num_cores
    | true, true ->
        Printf.sprintf "CPU Native Runtime Fission (%d cores)" num_cores
  in
  {
    general_info = {
      name;
      totalGlobalMem = Sys.max_array_length * 8;
      localMemSize = 64 * 1024;
      clockRate = 0;
      totalConstMem = 64 * 1024;
      multiProcessorCount = num_cores;
      eccEnabled = false;
      id;
      ctx = Obj.magic ();
    };
    specific_info =
      NativeInfo {
        native_num_cores = num_cores;
        native_parallel = parallel;
        native_fission = fission;
      };
    gc_info = Obj.magic ();
    events = Obj.magic ();
  }

let native_compatible_devices = ref 0

let init ?only:(s = Both) ?(interpreter = None) ?(native = true) () =
  openOutput () ;
  let idEvent = beginEvent "initialisation des devices" in
  begin match s with
  | Both ->
      cuda_init () ;
      cuda_compatible_devices := get_cuda_compatible_devices () ;
      cl_init () ;
      opencl_compatible_devices := get_opencl_compatible_devices ()
  | Cuda ->
      cuda_init () ;
      cuda_compatible_devices := get_cuda_compatible_devices ()
  | OpenCL ->
      cl_init () ;
      opencl_compatible_devices := get_opencl_compatible_devices ()
  end ;
  total_num_devices := !cuda_compatible_devices + !opencl_compatible_devices ;
  let devList = ref [] in
  for i = 0 to !cuda_compatible_devices - 1 do
    devList := !devList @ [get_cuda_device i]
  done ;
  let i = ref 0 and j = ref 0 in
  while !j < !opencl_compatible_devices do
    if is_available !i then (
      devList :=
        !devList @ [get_opencl_device !i (!i + !cuda_compatible_devices)] ;
      incr i) ;
    incr j
  done ;
  (* Optionally add interpreter device *)
  (match interpreter with
   | Some backend ->
       devList := !devList @ [create_interpreter_device ~backend ()] ;
       interpreter_compatible_devices := 1
   | None ->
       interpreter_compatible_devices := 0) ;
  (* Optionally add native CPU device (fission mode only) *)
  if native then begin
    devList := !devList @ [create_native_device ~parallel:true ~fission:true ()] ;
    native_compatible_devices := 1
  end
  else
    native_compatible_devices := 0 ;
  total_num_devices := List.length !devList ;
  opencl_compatible_devices := !i ;
  emitDeviceList !devList ;
  endEvent "fin initialisation des devices" idEvent ;
  Array.of_list !devList

let cuda_devices () = !cuda_compatible_devices

let opencl_devices () = !opencl_compatible_devices

let interpreter_devices () = !interpreter_compatible_devices

let native_devices () = !native_compatible_devices

let gpgpu_devices () = !total_num_devices

external cuda_flush : generalInfo -> device -> int -> unit = "spoc_cuda_flush"

external cuda_flush_all : generalInfo -> device -> unit = "spoc_cuda_flush_all"

external opencl_flush : generalInfo -> int -> unit = "spoc_opencl_flush"

(** Hook for native device flush - allows Sarek to register fission queue flush.
    The hook receives (device, queue_id option) and returns true if handled. *)
let native_flush_hook : (device -> int option -> bool) ref = ref (fun _ _ -> false)

(** Register a flush handler for native devices *)
let register_native_flush_hook f = native_flush_hook := f

let flush dev ?queue_id () =
  match (dev.specific_info, queue_id) with
  | CudaInfo _, None -> cuda_flush_all dev.general_info dev
  | CudaInfo _, Some q -> cuda_flush dev.general_info dev q
  | OpenCLInfo _, None -> opencl_flush dev.general_info 0
  | OpenCLInfo _, Some q -> opencl_flush dev.general_info q
  | InterpreterInfo _, _ -> () (* No-op for interpreter *)
  | NativeInfo _, _ ->
      (* Call the registered hook - Sarek registers fission queue flush *)
      ignore (!native_flush_hook dev queue_id)

let hasCLExtension dev ext =
  match dev.specific_info with
  | OpenCLInfo cli -> begin
      try
        ignore (Str.search_forward (Str.regexp ext) cli.extensions 0) ;
        true
      with _ -> false
    end
  | _ -> false

let allowDouble dev =
  match dev.specific_info with
  | OpenCLInfo _cli ->
      hasCLExtension dev "cl_khr_fp64" || hasCLExtension dev "cl_amd_fp64"
  | CudaInfo ci -> ci.major > 1 || ci.minor >= 3
  | InterpreterInfo _ -> true
  | NativeInfo _ -> true

(** Check if a device is the CPU interpreter *)
let is_interpreter dev =
  match dev.specific_info with
  | InterpreterInfo _ -> true
  | _ -> false

(** Find the interpreter device in an array, returns None if not present *)
let find_interpreter devs =
  Array.find_opt is_interpreter devs

(** Find the interpreter device index in an array, returns None if not present *)
let find_interpreter_id devs =
  let rec find i =
    if i >= Array.length devs then None
    else if is_interpreter devs.(i) then Some i
    else find (i + 1)
  in
  find 0

(** Check if a device is the native CPU runtime *)
let is_native dev =
  match dev.specific_info with
  | NativeInfo _ -> true
  | _ -> false

(** Check if a device is the native CPU runtime with parallel execution *)
let is_native_parallel dev =
  match dev.specific_info with
  | NativeInfo ni -> ni.native_parallel
  | _ -> false

(** Check if a device is the native CPU runtime with sequential execution *)
let is_native_sequential dev =
  match dev.specific_info with
  | NativeInfo ni -> not ni.native_parallel
  | _ -> false

(** Check if a device is the native CPU runtime with loop fission *)
let is_native_fission dev =
  match dev.specific_info with
  | NativeInfo ni -> ni.native_fission
  | _ -> false

(** Find the native device in an array, returns None if not present *)
let find_native devs =
  Array.find_opt is_native devs

(** Find the native device index in an array, returns None if not present.
    Returns the sequential native device (first one). *)
let find_native_id devs =
  let rec find i =
    if i >= Array.length devs then None
    else if is_native_sequential devs.(i) then Some i
    else find (i + 1)
  in
  find 0

(** Find the parallel native device index in an array *)
let find_native_parallel_id devs =
  let rec find i =
    if i >= Array.length devs then None
    else if is_native_parallel devs.(i) then Some i
    else find (i + 1)
  in
  find 0
