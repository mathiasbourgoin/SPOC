(******************************************************************************
 * OpenCL API - Ctypes Bindings
 *
 * Direct FFI bindings to OpenCL API via ctypes-foreign.
 * These are low-level bindings; see Opencl_api.ml for high-level wrappers.
 ******************************************************************************)

open Ctypes
open Foreign
open Opencl_types

(** {1 Library Loading} *)

let opencl_lib : Dl.library option ref = ref None

let get_opencl_lib () =
  match !opencl_lib with
  | Some lib -> lib
  | None ->
      let lib =
        try Dl.dlopen ~filename:"libOpenCL.so.1" ~flags:[Dl.RTLD_LAZY]
        with _ -> (
          try Dl.dlopen ~filename:"libOpenCL.so" ~flags:[Dl.RTLD_LAZY]
          with _ -> (
            try Dl.dlopen ~filename:"libOpenCL.dylib" ~flags:[Dl.RTLD_LAZY]
            with _ -> (
              try
                Dl.dlopen
                  ~filename:"/System/Library/Frameworks/OpenCL.framework/OpenCL"
                  ~flags:[Dl.RTLD_LAZY]
              with _ -> (
                try Dl.dlopen ~filename:"OpenCL.dll" ~flags:[Dl.RTLD_LAZY]
                with _ -> failwith "OpenCL library not found"))))
      in
      opencl_lib := Some lib ;
      lib

let foreign_cl name typ = foreign ~from:(get_opencl_lib ()) name typ

(** {1 Platform API} *)

(** Get platform IDs *)
let clGetPlatformIDs =
  foreign_cl
    "clGetPlatformIDs"
    (cl_uint
   (* num_entries *)
   @-> ptr cl_platform_id
    (* platforms *)
    @-> ptr cl_uint
    @->
    (* num_platforms *)
    returning cl_int)

(** Get platform info *)
let clGetPlatformInfo =
  foreign_cl
    "clGetPlatformInfo"
    (cl_platform_id
   (* platform *)
   @-> cl_uint
    (* param_name *)
    @-> size_t
    (* param_value_size *)
    @-> ptr void
    (* param_value *)
    @-> ptr size_t
    @->
    (* param_value_size_ret *)
    returning cl_int)

(** {1 Device API} *)

(** Get device IDs *)
let clGetDeviceIDs =
  foreign_cl
    "clGetDeviceIDs"
    (cl_platform_id
   (* platform *)
   @-> cl_bitfield
    (* device_type *)
    @-> cl_uint
    (* num_entries *)
    @-> ptr cl_device_id
    (* devices *)
    @-> ptr cl_uint
    @->
    (* num_devices *)
    returning cl_int)

(** Get device info *)
let clGetDeviceInfo =
  foreign_cl
    "clGetDeviceInfo"
    (cl_device_id
   (* device *)
   @-> cl_uint
    (* param_name *)
    @-> size_t
    (* param_value_size *)
    @-> ptr void
    (* param_value *)
    @-> ptr size_t
    @->
    (* param_value_size_ret *)
    returning cl_int)

(** {1 Context API} *)

(** Create context *)
let clCreateContext =
  foreign_cl
    "clCreateContext"
    (ptr cl_ulong
   (* properties (null-terminated) *)
   @-> cl_uint
    (* num_devices *)
    @-> ptr cl_device_id
    (* devices *)
    @-> ptr void
    (* pfn_notify callback *)
    @-> ptr void
    (* user_data *)
    @-> ptr cl_int
    @->
    (* errcode_ret *)
    returning cl_context)

(** Create context from type *)
let clCreateContextFromType =
  foreign_cl
    "clCreateContextFromType"
    (ptr cl_ulong
   (* properties *)
   @-> cl_bitfield
    (* device_type *)
    @-> ptr void
    (* pfn_notify *)
    @-> ptr void
    (* user_data *)
    @-> ptr cl_int
    @->
    (* errcode_ret *)
    returning cl_context)

(** Retain context *)
let clRetainContext =
  foreign_cl "clRetainContext" (cl_context @-> returning cl_int)

(** Release context *)
let clReleaseContext =
  foreign_cl "clReleaseContext" (cl_context @-> returning cl_int)

(** Get context info *)
let clGetContextInfo =
  foreign_cl
    "clGetContextInfo"
    (cl_context @-> cl_uint @-> size_t @-> ptr void @-> ptr size_t
   @-> returning cl_int)

(** {1 Command Queue API} *)

(** Create command queue (OpenCL 2.0+) *)
let clCreateCommandQueueWithProperties =
  foreign_cl
    "clCreateCommandQueueWithProperties"
    (cl_context
   (* context *)
   @-> cl_device_id
    (* device *)
    @-> ptr cl_ulong
    (* properties *)
    @-> ptr cl_int
    @->
    (* errcode_ret *)
    returning cl_command_queue)

(** Create command queue (OpenCL 1.x fallback) *)
let clCreateCommandQueue =
  try
    foreign_cl
      "clCreateCommandQueue"
      (cl_context @-> cl_device_id @-> cl_bitfield @-> ptr cl_int
     @-> returning cl_command_queue)
  with _ ->
    fun _ _ _ err ->
      err <-@ -34l ;
      (* CL_INVALID_CONTEXT *)
      from_voidp void null

(** Retain command queue *)
let clRetainCommandQueue =
  foreign_cl "clRetainCommandQueue" (cl_command_queue @-> returning cl_int)

(** Release command queue *)
let clReleaseCommandQueue =
  foreign_cl "clReleaseCommandQueue" (cl_command_queue @-> returning cl_int)

(** Flush command queue *)
let clFlush = foreign_cl "clFlush" (cl_command_queue @-> returning cl_int)

(** Finish (synchronize) command queue *)
let clFinish = foreign_cl "clFinish" (cl_command_queue @-> returning cl_int)

(** {1 Memory Object API} *)

(** Create buffer *)
let clCreateBuffer =
  foreign_cl
    "clCreateBuffer"
    (cl_context
   (* context *)
   @-> cl_bitfield
    (* flags *)
    @-> size_t
    (* size *)
    @-> ptr void
    (* host_ptr *)
    @-> ptr cl_int
    @->
    (* errcode_ret *)
    returning cl_mem)

(** Retain memory object *)
let clRetainMemObject =
  foreign_cl "clRetainMemObject" (cl_mem @-> returning cl_int)

(** Release memory object *)
let clReleaseMemObject =
  foreign_cl "clReleaseMemObject" (cl_mem @-> returning cl_int)

(** Get memory object info *)
let clGetMemObjectInfo =
  foreign_cl
    "clGetMemObjectInfo"
    (cl_mem @-> cl_uint @-> size_t @-> ptr void @-> ptr size_t
   @-> returning cl_int)

(** {1 Enqueue Operations} *)

(** Enqueue read buffer *)
let clEnqueueReadBuffer =
  foreign_cl
    "clEnqueueReadBuffer"
    (cl_command_queue
   (* command_queue *)
   @-> cl_mem
    (* buffer *)
    @-> cl_bool
    (* blocking_read *)
    @-> size_t
    (* offset *)
    @-> size_t
    (* size *)
    @-> ptr void
    (* ptr *)
    @-> cl_uint
    (* num_events_in_wait_list *)
    @-> ptr cl_event
    (* event_wait_list *)
    @-> ptr cl_event
    @->
    (* event *)
    returning cl_int)

(** Enqueue write buffer *)
let clEnqueueWriteBuffer =
  foreign_cl
    "clEnqueueWriteBuffer"
    (cl_command_queue @-> cl_mem @-> cl_bool @-> size_t @-> size_t @-> ptr void
   @-> cl_uint @-> ptr cl_event @-> ptr cl_event @-> returning cl_int)

(** Enqueue copy buffer *)
let clEnqueueCopyBuffer =
  foreign_cl
    "clEnqueueCopyBuffer"
    (cl_command_queue @-> cl_mem
   (* src_buffer *)
   @-> cl_mem
    (* dst_buffer *)
    @-> size_t
    (* src_offset *)
    @-> size_t
    (* dst_offset *)
    @-> size_t
    (* size *)
    @-> cl_uint
    @-> ptr cl_event @-> ptr cl_event @-> returning cl_int)

(** Enqueue fill buffer *)
let clEnqueueFillBuffer =
  try
    foreign_cl
      "clEnqueueFillBuffer"
      (cl_command_queue @-> cl_mem @-> ptr void
     (* pattern *)
     @-> size_t
      (* pattern_size *)
      @-> size_t
      (* offset *)
      @-> size_t
      (* size *)
      @-> cl_uint
      @-> ptr cl_event @-> ptr cl_event @-> returning cl_int)
  with _ -> fun _ _ _ _ _ _ _ _ _ -> -30l (* CL_INVALID_VALUE *)

(** {1 Program API} *)

(** Create program from source *)
let clCreateProgramWithSource =
  foreign_cl
    "clCreateProgramWithSource"
    (cl_context
   (* context *)
   @-> cl_uint
    (* count *)
    @-> ptr string
    (* strings *)
    @-> ptr size_t
    (* lengths *)
    @-> ptr cl_int
    @->
    (* errcode_ret *)
    returning cl_program)

(** Create program from binary *)
let clCreateProgramWithBinary =
  foreign_cl
    "clCreateProgramWithBinary"
    (cl_context @-> cl_uint
   (* num_devices *)
   @-> ptr cl_device_id
    (* device_list *)
    @-> ptr size_t
    (* lengths *)
    @-> ptr (ptr uchar)
    (* binaries *)
    @-> ptr cl_int
    (* binary_status *)
    @-> ptr cl_int
    @->
    (* errcode_ret *)
    returning cl_program)

(** Build program *)
let clBuildProgram =
  foreign_cl
    "clBuildProgram"
    (cl_program
   (* program *)
   @-> cl_uint
    (* num_devices *)
    @-> ptr cl_device_id
    (* device_list *)
    @-> string
    (* options *)
    @-> ptr void
    (* pfn_notify *)
    @-> ptr void
    @->
    (* user_data *)
    returning cl_int)

(** Get program build info *)
let clGetProgramBuildInfo =
  foreign_cl
    "clGetProgramBuildInfo"
    (cl_program @-> cl_device_id @-> cl_uint
   (* param_name *)
   @-> size_t
    @-> ptr void @-> ptr size_t @-> returning cl_int)

(** Retain program *)
let clRetainProgram =
  foreign_cl "clRetainProgram" (cl_program @-> returning cl_int)

(** Release program *)
let clReleaseProgram =
  foreign_cl "clReleaseProgram" (cl_program @-> returning cl_int)

(** {1 Kernel API} *)

(** Create kernel *)
let clCreateKernel =
  foreign_cl
    "clCreateKernel"
    (cl_program
   (* program *)
   @-> string
    (* kernel_name *)
    @-> ptr cl_int
    @->
    (* errcode_ret *)
    returning cl_kernel)

(** Set kernel argument *)
let clSetKernelArg =
  foreign_cl
    "clSetKernelArg"
    (cl_kernel
   (* kernel *)
   @-> cl_uint
    (* arg_index *)
    @-> size_t
    (* arg_size *)
    @-> ptr void
    @->
    (* arg_value *)
    returning cl_int)

(** Retain kernel *)
let clRetainKernel = foreign_cl "clRetainKernel" (cl_kernel @-> returning cl_int)

(** Release kernel *)
let clReleaseKernel =
  foreign_cl "clReleaseKernel" (cl_kernel @-> returning cl_int)

(** Get kernel info *)
let clGetKernelInfo =
  foreign_cl
    "clGetKernelInfo"
    (cl_kernel @-> cl_uint @-> size_t @-> ptr void @-> ptr size_t
   @-> returning cl_int)

(** Get kernel work group info *)
let clGetKernelWorkGroupInfo =
  foreign_cl
    "clGetKernelWorkGroupInfo"
    (cl_kernel @-> cl_device_id @-> cl_uint @-> size_t @-> ptr void
   @-> ptr size_t @-> returning cl_int)

(** Enqueue ND range kernel *)
let clEnqueueNDRangeKernel =
  foreign_cl
    "clEnqueueNDRangeKernel"
    (cl_command_queue
   (* command_queue *)
   @-> cl_kernel
    (* kernel *)
    @-> cl_uint
    (* work_dim *)
    @-> ptr size_t
    (* global_work_offset *)
    @-> ptr size_t
    (* global_work_size *)
    @-> ptr size_t
    (* local_work_size *)
    @-> cl_uint
    (* num_events_in_wait_list *)
    @-> ptr cl_event
    (* event_wait_list *)
    @-> ptr cl_event
    @->
    (* event *)
    returning cl_int)

(** {1 Event API} *)

(** Wait for events *)
let clWaitForEvents =
  foreign_cl
    "clWaitForEvents"
    (cl_uint
   (* num_events *)
   @-> ptr cl_event
    @->
    (* event_list *)
    returning cl_int)

(** Retain event *)
let clRetainEvent = foreign_cl "clRetainEvent" (cl_event @-> returning cl_int)

(** Release event *)
let clReleaseEvent = foreign_cl "clReleaseEvent" (cl_event @-> returning cl_int)

(** Get event info *)
let clGetEventInfo =
  foreign_cl
    "clGetEventInfo"
    (cl_event @-> cl_uint @-> size_t @-> ptr void @-> ptr size_t
   @-> returning cl_int)

(** Get event profiling info *)
let clGetEventProfilingInfo =
  foreign_cl
    "clGetEventProfilingInfo"
    (cl_event @-> cl_uint
   (* param_name *)
   @-> size_t
    @-> ptr void @-> ptr size_t @-> returning cl_int)

(** {1 Synchronization} *)

(** Enqueue marker with wait list *)
let clEnqueueMarkerWithWaitList =
  try
    foreign_cl
      "clEnqueueMarkerWithWaitList"
      (cl_command_queue @-> cl_uint @-> ptr cl_event @-> ptr cl_event
     @-> returning cl_int)
  with _ -> fun _ _ _ _ -> -30l

(** Enqueue barrier with wait list *)
let clEnqueueBarrierWithWaitList =
  try
    foreign_cl
      "clEnqueueBarrierWithWaitList"
      (cl_command_queue @-> cl_uint @-> ptr cl_event @-> ptr cl_event
     @-> returning cl_int)
  with _ -> fun _ _ _ _ -> -30l

(** {1 Helpers} *)

(** Check if OpenCL is available *)
let is_available () : bool =
  try
    let _ = get_opencl_lib () in
    true
  with _ -> false
