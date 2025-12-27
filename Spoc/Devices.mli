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
* modify and / or redistribute the software under the terms of the CeCILL - B
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

(** Manages gpgpu devices compatible with Spoc
within the system *)

type dim3 = { x : int; y : int; z : int; }

(** Different frameworks Spoc can use
(currently Nvidia Cuda and OpenCL) *)
type specificLibrary = Cuda | OpenCL | Both

type context

(** General information shared
by all devices (managed by Cuda or OpenCL) *)
type generalInfo = {
  name : string; (**name of the device *)
  totalGlobalMem : int; (** the total amount of global memory on the device *)
  localMemSize : int; (** the total amount of local memory on the device *)
  clockRate : int; (** the clock rate of the device *)
  totalConstMem : int; (** the total amount of constant memory on the device *)
  multiProcessorCount : int; (** the number of multi processor on the device *)
  eccEnabled : bool; (** is ECC (Error Correcting Code) enabled on the device *)
  id : int; (** the id of the device *)
  ctx : context; (** the context associated with this device *)
}

(** Specific information depending on the framework
used to manage the device *)

(** Specific information for Cuda-managed devices*)
type cudaInfo = {
  major : int; (** Major compute capability *)
  minor : int; (** Minor compute capability *)
  regsPerBlock : int; (** 32-bit registers available per block *)
  warpSize : int; (** Warp size in threads *)
  memPitch : int; (** maximum pitch in bytes allowed by memory copies *)
  maxThreadsPerBlock : int; (** Maximum number of threads per block *)
  maxThreadsDim : dim3; (** Maximum size of each dimension of a block *)
  maxGridSize : dim3; (** Maximum size of each dimension of a grid *)
  textureAlignment : int; (** Alignment requirement for textures *)
  deviceOverlap : bool; (** Device can concurrently copy memory and execute a kernel *)
  kernelExecTimeoutEnabled : bool; (** Specified whether there is a run time limit on kernels *)
  integrated : bool; (** Device is integrated as opposed to discrete *)
  canMapHostMemory : bool; (** Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer *)
  computeMode : int; (** Compute mode *)
  concurrentKernels : bool; (** Device can possibly execute multiple kernels concurrently *)
  pciBusID : int; (** PCI bus ID of the device *)
  pciDeviceID : int; (** PCI device ID of the device *)
  driverVersion: int;
}

(** Specific information for OpenCL-managed devices*)

(** information about the OpenCL platform *)
type platformInfo = {
  platform_profile : string; (** OpenCL profile string. Returns the profile name
  supported by the implementation.
  The profile name returned can be one of the
  following strings:
  - FULL_PROFILE - if the implementation
  supports the OpenCL specification
  (functionality defined as part of the
  core specification and does not require
  any extensions to be supported).
  - EMBEDDED_PROFILE - if the implementation
  supports the OpenCL embedded profile.
  The embedded profile is defined to be a
  subset for each version of OpenCL.
  *)
  platform_version : string; (** OpenCL version string. Returns the OpenCL
  version supported by the implementation.
  This version string has the following format:
  OpenCL < space >< major_version.minor_version >< space >< platform - specific information >
  The major_version.minor_version value returned will be 1.1. *)
  platform_name : string; (** Platform name string *)
  platform_vendor : string; (** Platform vendor string *)
  platform_extensions : string; (** Returns a space - separated list of extension
  names (the extension names themselves do not
  contain any spaces) supported by the
  platform. Extensions defined here must be
  supported by all devices associated with
  this platform. *)
  num_devices : int; (** Number of devices associated with this platform *)
}

(** Device type GPU|CPU|Accelerator *)
type deviceType =
    CL_DEVICE_TYPE_CPU
  | CL_DEVICE_TYPE_GPU
  | CL_DEVICE_TYPE_ACCELERATOR
  | CL_DEVICE_TYPE_DEFAULT

type clDeviceFPConfig =
    CL_FP_DENORM
  | CL_FP_INF_NAN
  | CL_FP_ROUND_TO_NEAREST
  | CL_FP_ROUND_TO_ZERO
  | CL_FP_ROUND_TO_INF
  | CL_FP_FMA
  | CL_FP_NONE

type clDeviceQueueProperties =
    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
  | CL_QUEUE_PROFILING_ENABLE

type clDeviceGlobalMemCacheType =
    CL_READ_WRITE_CACHE
  | CL_READ_ONLY_CACHE
  | CL_NONE

type clDeviceLocalMemType = CL_LOCAL | CL_GLOBAL

type clDeviceExecutionCapabilities = CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL

type clDeviceID

(** Specific information for OpenCL-managed devices *)
type openCLInfo = {
  platform_info : platformInfo;
  device_type : deviceType;
  profile : string; (** OpenCL profile string. Returns the profile name supported by the device (see note). The profile name returned can be one of the following strings:
  - FULL_PROFILE - if the device supports the OpenCL specification (functionality defined as part of the core specification and does not require any extensions to be supported).
  - EMBEDDED_PROFILE - if the device supports the OpenCL embedded profile. *)
  version : string; (** OpenCL version string. Returns the OpenCL version supported by the device. This version string has the following format:
  OpenCL < space >< major_version.minor_version >< space >< vendor - specific information >
  The major_version.minor_version value returned will be 1.1. *)
  vendor : string; (** Vendor name string *)
  extensions : string; (** Returns a space - separated list of extension names (the extension names themselves do not contain any spaces). The list of extension names returned currently can include one or more of the following approved extension names:
  - cl_khr_fp64
  - cl_khr_int64_base_atomics
  - cl_khr_int64_extended_atomics
  - cl_khr_fp16
  - cl_khr_gl_sharing
  - cl_khr_gl_event
  - cl_khr_d3d10_sharing *)
  vendor_id : int; (** A unique device vendor identifier. An example of a unique device identifier could be the PCIe ID*)
  max_work_item_dimensions : int; (** Maximum dimensions that specify the global and local work-item IDs used by the data parallel execution model. (Refer to clEnqueueNDRangeKernel). The minimum value is 3 *)
  address_bits : int; (** The default compute device address space size specified as an unsigned integer value in bits. Currently supported values are 32 or 64 bits. *)
  max_mem_alloc_size : int; (** Max size of memory object allocation in bytes. The minimum value is max (1/4th of CL_DEVICE_GLOBAL_MEM_SIZE, 128*1024*1024) *)
  image_support : bool; (** Is true if images are supported by the OpenCL device and false otherwise. *)
  max_read_image_args : int; (** Max number of simultaneous image objects that can be read by a kernel. The minimum value is 128 if image_support is true. *)
  max_write_image_args : int; (** Max number of simultaneous image objects that can be written to by a kernel. The minimum value is 8 if image_support is true. *)
  max_samplers : int; (** Maximum number of samplers that can be used in a kernel. The minimum value is 16 if image_support is true. *)
  mem_base_addr_align : int; (** Describes the alignment in bits of the base address of any allocated memory object. *)
  min_data_type_align_size : int; (** The smallest alignment in bytes which can be used for any data type. *)
  global_mem_cacheline_size : int; (** Size of global memory cache line in bytes. *)
  global_mem_cache_size : int; (** Size of global memory cache in bytes. *)
  max_constant_args : int; (** Max number of arguments declared with the __constant qualifier in a kernel. The minimum value is 8. *)
  endian_little : bool; (** Is true if the OpenCL device is a little endian device and false otherwise. *)
  available : bool; (** Is true if the device is available from Spoc *)
  compiler_available : bool; (** Is false if the implementation does not have a compiler available to compile the program source. Is true if the compiler is available. This can be false for the embedded platform profile only. *)
  single_fp_config : clDeviceFPConfig; (** Describes single precision floating - point capability of the device. This is a bit - field that describes one or more of the following values:
  - CL_FP_DENORM - denorms are supported
  - CL_FP_INF_NAN - INF and quiet NaNs are supported
  - CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported
  - CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported
  - CL_FP_ROUND_TO_INF - round to + ve and - ve infinity rounding modes supported
  - CL_FP_FMA - IEEE754 -2008 fused multiply - add is supported
  - CL_FP_SOFT_FLOAT - Basic floating - point operations (such as addition, subtraction, multiplication) are implemented in software.
  The mandated minimum floating - point capability is CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN. *)
  global_mem_cache_type : clDeviceGlobalMemCacheType; (** Type of global memory cache supported. Valid values are: CL_NONE, CL_READ_ONLY_CACHE, and CL_READ_WRITE_CACHE. *)
  queue_properties : clDeviceQueueProperties;  (** Describes the command - queue properties supported by the device. This is a bit - field that describes one or more of the following values:
  - CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
  - CL_QUEUE_PROFILING_ENABLE
  These properties are described in the table for clCreateCommandQueue. The mandated minimum capability is CL_QUEUE_PROFILING_ENABLE. *)
  local_mem_type : clDeviceLocalMemType; (** Type of local memory supported. This can be set to CL_LOCAL implying dedicated local memory storage such as SRAM, or CL_GLOBAL. *)
  double_fp_config : clDeviceFPConfig; (** Describes the OPTIONAL double precision floating - point capability of the OpenCL device. This is a bit - field that describes one or more of the following values:
  - CL_FP_DENORM - denorms are supported.
  - CL_FP_INF_NAN - INF and NaNs are supported.
  - CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported.
  - CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported.
  - CL_FP_ROUND_TO_INF - round to + ve and - ve infinity rounding modes supported.
  - CP_FP_FMA - IEEE754 -2008 fused multiply - add is supported.
  The mandated minimum double precision floating - point capability is CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM.*)
  max_constant_buffer_size : int; (** Max size in bytes of a constant buffer allocation. The minimum value is 64 KB. *)
  execution_capabilities : clDeviceExecutionCapabilities; (** Describes the execution capabilities of the device. This is a bit - field that describes one or more of the following values:
  - CL_EXEC_KERNEL - The OpenCL device can execute OpenCL kernels.
  - CL_EXEC_NATIVE_KERNEL - The OpenCL device can execute native kernels.
  The mandated minimum capability is CL_EXEC_KERNEL.*)
  half_fp_config : clDeviceFPConfig; (** Describes the OPTIONAL half precision floating - point capability of the OpenCL device. This is a bit - field that describes one or more of the following values:
  - CL_FP_DENORM - denorms are supported.
  - CL_FP_INF_NAN - INF and NaNs are supported.
  - CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported.
  - CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported.
  - CL_FP_ROUND_TO_INF - round to + ve and - ve infinity rounding modes supported.
  - CP_FP_FMA - IEEE754 -2008 fused multiply - add is supported.
  - CL_FP_SOFT_FLOAT - Basic floating - point operations (such as addition, subtraction, multiplication) are implemented in software.
  The required minimum half precision floating - point capability as implemented by this extension is CL_FP_ROUND_TO_ZERO or CL_FP_ROUND_TO_INF | CL_FP_INF_NAN. *)
  max_work_group_size : int; (** Maximum number of work-items in a work-group executing a kernel using the data parallel execution model. The minimum value is 1. *)
  image2D_max_height : int; (** Max height of 2D image in pixels. The minimum value is 8192 if image_support is true. *)
  image2D_max_width : int; (** Max width of 2D image in pixels. The minimum value is 8192 if image_support is true. *)
  image3D_max_depth : int; (** Max depth of 3D image in pixels. The minimum value is 2048 if image_support is true. *)
  image3D_max_height : int; (** Max height of 3D image in pixels. The minimum value is 2048 if image_support is true. *)
  image3D_max_width : int; (** Max width of 3D image in pixels. The minimum value is 2048 if image_support is true. *)
  max_parameter_size : int; (** Max size in bytes of the arguments that can be passed to a kernel. The minimum value is 1024. For this minimum value, only a maximum of 128 arguments can be passed to a kernel. *)
  max_work_item_size : dim3; (** Maximum number of work-items that can be specified in each dimension of the work-group to clEnqueueNDRangeKernel.  The minimum value is (1, 1, 1). *)
  prefered_vector_width_char : int; (** Preferred native vector width size for built - in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
  If the cl_khr_fp16 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF must return 0. *)
  prefered_vector_width_short : int; (** see prefered_vector_width_char *)
  prefered_vector_width_int : int; (** see prefered_vector_width_char *)
  prefered_vector_width_long : int; (** see prefered_vector_width_char *)
  prefered_vector_width_float : int; (** see prefered_vector_width_char *)
  prefered_vector_width_double : int; (** see prefered_vector_width_char *)
  profiling_timer_resolution : int; (** Describes the resolution of device timer. This is measured in nanoseconds.*)
  driver_version : string; (** OpenCL software driver version string in the form major_number.minor_number.*)
  device_id : clDeviceID; (** Device ID *)
}

(**/**)

(** Interpreter execution backend *)
type interpreter_backend =
  | Sequential  (** Deterministic single-threaded, for debugging *)
  | Parallel    (** Multi-threaded using Domains, for performance *)

(** Information for interpreter-based CPU devices *)
type interpreterInfo = {
  backend : interpreter_backend;
  num_cores : int;    (** Number of cores for parallel backend *)
  debug_mode : bool;  (** Enable extra validation and tracing *)
}

(** Information for native CPU runtime devices *)
type nativeInfo = {
  native_num_cores : int;  (** Number of cores for parallel execution *)
}

type specificInfo =
    CudaInfo of cudaInfo
  | OpenCLInfo of openCLInfo
  | InterpreterInfo of interpreterInfo
  | NativeInfo of nativeInfo

type gcInfo
type events

type device = {
  general_info : generalInfo;
  specific_info : specificInfo;
  gc_info : gcInfo;
  events: events;
}

external get_cuda_compatible_devices : unit -> int
= "spoc_getCudaDevicesCount"
external get_opencl_compatible_devices : unit -> int
= "spoc_getOpenCLDevicesCount"
(**/**)

#ifdef SPOC_PROFILE
external closeOutput : unit -> unit = "close_output_profiling"
#endif
                                        
(** Mandatory function to use Spoc
@param only allows to specify which library to use, by default, Spoc will search any device on the system
@param interpreter Defaults to [Some Sequential]. Use [None] to exclude the interpreter device.
@param native if true, adds a native CPU runtime device (default: true)
@return an array containing every compatible device found on the system *)
val init : ?only: specificLibrary -> ?interpreter:interpreter_backend option -> ?native:bool -> unit -> device array

(** @return the number of Cuda compatible devices found *)
val cuda_devices : unit -> int

(** @return the number of interpreter devices (0 or 1) *)
val interpreter_devices : unit -> int

(** @return the number of native CPU runtime devices (0 or 1) *)
val native_devices : unit -> int

(** @return the number of OpenCL compatible devices found *)
val opencl_devices : unit -> int

(** @return the total number of compatible devices found *)
val gpgpu_devices : unit -> int

(** Waits the command queues of a device to end
@param queue_id allows to specify only a specific command queue *)
val flush : device -> ?queue_id: int -> unit -> unit

(** Checks if a device offers an extension *)
val hasCLExtension : device -> string -> bool


val allowDouble : device -> bool

(** Check if a device is the CPU interpreter *)
val is_interpreter : device -> bool

(** Find the interpreter device in an array, returns None if not present *)
val find_interpreter : device array -> device option

(** Find the interpreter device index in an array, returns None if not present *)
val find_interpreter_id : device array -> int option

(** Create an interpreter-based CPU device.
    This device executes kernels using a pure OCaml interpreter,
    without requiring GPU hardware.
    @param backend execution backend (default: Sequential)
    @param debug enable extra validation (default: false)
    @return a device that can be used with Kirc.run *)
val create_interpreter_device :
  ?backend:interpreter_backend -> ?debug:bool -> unit -> device

(** Check if a device is the native CPU runtime *)
val is_native : device -> bool

(** Find the native device in an array, returns None if not present *)
val find_native : device array -> device option

(** Find the native device index in an array, returns None if not present *)
val find_native_id : device array -> int option

(** Create a native CPU runtime device.
    This device executes kernels using PPX-generated native OCaml code,
    which runs at full native speed without interpretation overhead.
    @return a device that can be used with Kirc.run *)
val create_native_device : unit -> device
