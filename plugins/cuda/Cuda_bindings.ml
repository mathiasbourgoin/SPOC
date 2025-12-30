(******************************************************************************
 * CUDA Driver API - Ctypes Bindings
 *
 * Direct FFI bindings to CUDA driver API via ctypes-foreign.
 * These are low-level bindings; see Cuda_api.ml for high-level wrappers.
 ******************************************************************************)

open Ctypes
open Foreign
open Cuda_types

(** {1 Library Loading} *)

(** Load CUDA driver library dynamically *)
let cuda_lib : Dl.library option ref = ref None

let get_cuda_lib () =
  match !cuda_lib with
  | Some lib -> lib
  | None ->
      let lib =
        try
          Dl.dlopen
            ~filename:"libcuda.so.1"
            ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL]
        with _ -> (
          try
            Dl.dlopen
              ~filename:"libcuda.so"
              ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL]
          with _ -> (
            try
              Dl.dlopen
                ~filename:"libcuda.dylib"
                ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL]
            with _ -> (
              try
                Dl.dlopen
                  ~filename:"nvcuda.dll"
                  ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL]
              with _ -> failwith "CUDA driver library not found")))
      in
      cuda_lib := Some lib ;
      lib

(** Create a foreign binding to CUDA driver API *)
let foreign_cuda name typ = foreign ~from:(get_cuda_lib ()) name typ

(** {1 Initialization} *)

(** Initialize the CUDA driver API. Must be called before any other function. *)
let cuInit = foreign_cuda "cuInit" (int @-> returning cu_result)

(** {1 Device Management} *)

(** Get the number of CUDA-capable devices *)
let cuDeviceGetCount =
  foreign_cuda "cuDeviceGetCount" (ptr int @-> returning cu_result)

(** Get a device handle by ordinal *)
let cuDeviceGet =
  foreign_cuda "cuDeviceGet" (ptr cu_device @-> int @-> returning cu_result)

(** Get device name *)
let cuDeviceGetName =
  foreign_cuda
    "cuDeviceGetName"
    (ptr char @-> int @-> cu_device @-> returning cu_result)

(** Get total memory on device *)
let cuDeviceTotalMem =
  foreign_cuda
    "cuDeviceTotalMem_v2"
    (ptr size_t @-> cu_device @-> returning cu_result)

(** Get a device attribute *)
let cuDeviceGetAttribute =
  foreign_cuda
    "cuDeviceGetAttribute"
    (ptr int @-> int @-> cu_device @-> returning cu_result)

(** Get compute capability *)
let cuDeviceComputeCapability =
  foreign_cuda
    "cuDeviceComputeCapability"
    (ptr int @-> ptr int @-> cu_device @-> returning cu_result)

(** {1 Context Management} *)

(** Create a CUDA context *)
let cuCtxCreate =
  foreign_cuda
    "cuCtxCreate_v2"
    (ptr cu_context_ptr @-> uint @-> cu_device @-> returning cu_result)

(** Destroy a CUDA context *)
let cuCtxDestroy =
  foreign_cuda "cuCtxDestroy_v2" (cu_context_ptr @-> returning cu_result)

(** Push context onto the current thread's context stack *)
let cuCtxPushCurrent =
  foreign_cuda "cuCtxPushCurrent_v2" (cu_context_ptr @-> returning cu_result)

(** Pop context from current thread's context stack *)
let cuCtxPopCurrent =
  foreign_cuda "cuCtxPopCurrent_v2" (ptr cu_context_ptr @-> returning cu_result)

(** Set the current context *)
let cuCtxSetCurrent =
  foreign_cuda "cuCtxSetCurrent" (cu_context_ptr @-> returning cu_result)

(** Get the current context *)
let cuCtxGetCurrent =
  foreign_cuda "cuCtxGetCurrent" (ptr cu_context_ptr @-> returning cu_result)

(** Synchronize the current context *)
let cuCtxSynchronize =
  foreign_cuda "cuCtxSynchronize" (void @-> returning cu_result)

(** Get device for current context *)
let cuCtxGetDevice =
  foreign_cuda "cuCtxGetDevice" (ptr cu_device @-> returning cu_result)

(** {1 Memory Management} *)

(** Allocate device memory *)
let cuMemAlloc =
  foreign_cuda
    "cuMemAlloc_v2"
    (ptr cu_deviceptr @-> size_t @-> returning cu_result)

(** Free device memory *)
let cuMemFree =
  foreign_cuda "cuMemFree_v2" (cu_deviceptr @-> returning cu_result)

(** Copy memory from host to device *)
let cuMemcpyHtoD =
  foreign_cuda
    "cuMemcpyHtoD_v2"
    (cu_deviceptr @-> ptr void @-> size_t @-> returning cu_result)

(** Copy memory from device to host *)
let cuMemcpyDtoH =
  foreign_cuda
    "cuMemcpyDtoH_v2"
    (ptr void @-> cu_deviceptr @-> size_t @-> returning cu_result)

(** Copy memory from device to device *)
let cuMemcpyDtoD =
  foreign_cuda
    "cuMemcpyDtoD_v2"
    (cu_deviceptr @-> cu_deviceptr @-> size_t @-> returning cu_result)

(** Async copy from host to device *)
let cuMemcpyHtoDAsync =
  foreign_cuda
    "cuMemcpyHtoDAsync_v2"
    (cu_deviceptr @-> ptr void @-> size_t @-> cu_stream_ptr
   @-> returning cu_result)

(** Async copy from device to host *)
let cuMemcpyDtoHAsync =
  foreign_cuda
    "cuMemcpyDtoHAsync_v2"
    (ptr void @-> cu_deviceptr @-> size_t @-> cu_stream_ptr
   @-> returning cu_result)

(** Set memory to a value (8-bit) *)
let cuMemsetD8 =
  foreign_cuda
    "cuMemsetD8_v2"
    (cu_deviceptr @-> uchar @-> size_t @-> returning cu_result)

(** Set memory to a value (32-bit) *)
let cuMemsetD32 =
  foreign_cuda
    "cuMemsetD32_v2"
    (cu_deviceptr @-> uint32_t @-> size_t @-> returning cu_result)

(** Allocate host memory (page-locked) *)
let cuMemAllocHost =
  foreign_cuda
    "cuMemAllocHost_v2"
    (ptr (ptr void) @-> size_t @-> returning cu_result)

(** Free host memory *)
let cuMemFreeHost =
  foreign_cuda "cuMemFreeHost" (ptr void @-> returning cu_result)

(** Get memory info (free and total) *)
let cuMemGetInfo =
  foreign_cuda
    "cuMemGetInfo_v2"
    (ptr size_t @-> ptr size_t @-> returning cu_result)

(** {1 Module Management} *)

(** Load a module from a PTX/cubin image in memory *)
let cuModuleLoadData =
  foreign_cuda
    "cuModuleLoadData"
    (ptr cu_module_ptr @-> ptr void @-> returning cu_result)

(** Load a module from a PTX/cubin file *)
let cuModuleLoad =
  foreign_cuda
    "cuModuleLoad"
    (ptr cu_module_ptr @-> string @-> returning cu_result)

(** Load module with JIT options *)
let cuModuleLoadDataEx =
  foreign_cuda
    "cuModuleLoadDataEx"
    (ptr cu_module_ptr @-> ptr void @-> uint @-> ptr int
    @-> ptr (ptr void)
    @-> returning cu_result)

(** Unload a module *)
let cuModuleUnload =
  foreign_cuda "cuModuleUnload" (cu_module_ptr @-> returning cu_result)

(** Get a function handle from a module *)
let cuModuleGetFunction =
  foreign_cuda
    "cuModuleGetFunction"
    (ptr cu_function_ptr @-> cu_module_ptr @-> string @-> returning cu_result)

(** Get a global variable from a module *)
let cuModuleGetGlobal =
  foreign_cuda
    "cuModuleGetGlobal_v2"
    (ptr cu_deviceptr @-> ptr size_t @-> cu_module_ptr @-> string
   @-> returning cu_result)

(** {1 Kernel Execution} *)

(** Launch a kernel *)
let cuLaunchKernel =
  foreign_cuda
    "cuLaunchKernel"
    (cu_function_ptr
   (* function *)
   @-> uint
    @-> uint @-> uint
    (* grid dimensions (x, y, z) *)
    @-> uint
    @-> uint @-> uint
    (* block dimensions (x, y, z) *)
    @-> uint
    (* shared memory bytes *)
    @-> cu_stream_ptr
    (* stream (null for default) *)
    @-> ptr (ptr void)
    (* kernel parameters *)
    @-> ptr (ptr void)
    @->
    (* extra (usually null) *)
    returning cu_result)

(** Get function attribute *)
let cuFuncGetAttribute =
  foreign_cuda
    "cuFuncGetAttribute"
    (ptr int @-> int @-> cu_function_ptr @-> returning cu_result)

(** Set function cache configuration *)
let cuFuncSetCacheConfig =
  foreign_cuda
    "cuFuncSetCacheConfig"
    (cu_function_ptr @-> int @-> returning cu_result)

(** Set function shared memory configuration *)
let cuFuncSetSharedMemConfig =
  foreign_cuda
    "cuFuncSetSharedMemConfig"
    (cu_function_ptr @-> int @-> returning cu_result)

(** {1 Stream Management} *)

(** Create a stream *)
let cuStreamCreate =
  foreign_cuda
    "cuStreamCreate"
    (ptr cu_stream_ptr @-> uint @-> returning cu_result)

(** Create a stream with priority *)
let cuStreamCreateWithPriority =
  foreign_cuda
    "cuStreamCreateWithPriority"
    (ptr cu_stream_ptr @-> uint @-> int @-> returning cu_result)

(** Destroy a stream *)
let cuStreamDestroy =
  foreign_cuda "cuStreamDestroy_v2" (cu_stream_ptr @-> returning cu_result)

(** Synchronize a stream *)
let cuStreamSynchronize =
  foreign_cuda "cuStreamSynchronize" (cu_stream_ptr @-> returning cu_result)

(** Query stream status *)
let cuStreamQuery =
  foreign_cuda "cuStreamQuery" (cu_stream_ptr @-> returning cu_result)

(** Wait for an event in a stream *)
let cuStreamWaitEvent =
  foreign_cuda
    "cuStreamWaitEvent"
    (cu_stream_ptr @-> cu_event_ptr @-> uint @-> returning cu_result)

(** Get stream priority range *)
let cuCtxGetStreamPriorityRange =
  foreign_cuda
    "cuCtxGetStreamPriorityRange"
    (ptr int @-> ptr int @-> returning cu_result)

(** {1 Event Management} *)

(** Create an event *)
let cuEventCreate =
  foreign_cuda
    "cuEventCreate"
    (ptr cu_event_ptr @-> uint @-> returning cu_result)

(** Destroy an event *)
let cuEventDestroy =
  foreign_cuda "cuEventDestroy_v2" (cu_event_ptr @-> returning cu_result)

(** Record an event *)
let cuEventRecord =
  foreign_cuda
    "cuEventRecord"
    (cu_event_ptr @-> cu_stream_ptr @-> returning cu_result)

(** Synchronize an event *)
let cuEventSynchronize =
  foreign_cuda "cuEventSynchronize" (cu_event_ptr @-> returning cu_result)

(** Query event status *)
let cuEventQuery =
  foreign_cuda "cuEventQuery" (cu_event_ptr @-> returning cu_result)

(** Get elapsed time between events *)
let cuEventElapsedTime =
  foreign_cuda
    "cuEventElapsedTime"
    (ptr float @-> cu_event_ptr @-> cu_event_ptr @-> returning cu_result)

(** {1 Error Handling} *)

(** Get error name *)
let cuGetErrorName =
  foreign_cuda
    "cuGetErrorName"
    (cu_result @-> ptr string @-> returning cu_result)

(** Get error string *)
let cuGetErrorString =
  foreign_cuda
    "cuGetErrorString"
    (cu_result @-> ptr string @-> returning cu_result)

(** {1 Version} *)

(** Get CUDA driver version *)
let cuDriverGetVersion =
  foreign_cuda "cuDriverGetVersion" (ptr int @-> returning cu_result)

(** {1 Occupancy} *)

(** Get max active blocks per multiprocessor *)
let cuOccupancyMaxActiveBlocksPerMultiprocessor =
  foreign_cuda
    "cuOccupancyMaxActiveBlocksPerMultiprocessor"
    (ptr int @-> cu_function_ptr @-> int @-> size_t @-> returning cu_result)

(** {1 Peer Access} *)

(** Check if peer access is possible *)
let cuDeviceCanAccessPeer =
  foreign_cuda
    "cuDeviceCanAccessPeer"
    (ptr int @-> cu_device @-> cu_device @-> returning cu_result)

(** Enable peer access *)
let cuCtxEnablePeerAccess =
  foreign_cuda
    "cuCtxEnablePeerAccess"
    (cu_context_ptr @-> uint @-> returning cu_result)

(** Disable peer access *)
let cuCtxDisablePeerAccess =
  foreign_cuda "cuCtxDisablePeerAccess" (cu_context_ptr @-> returning cu_result)

(** {1 Profiler Control} *)

(** Start profiler *)
let cuProfilerStart =
  try foreign_cuda "cuProfilerStart" (void @-> returning cu_result)
  with _ -> fun () -> CUDA_SUCCESS (* Profiler may not be available *)

(** Stop profiler *)
let cuProfilerStop =
  try foreign_cuda "cuProfilerStop" (void @-> returning cu_result)
  with _ -> fun () -> CUDA_SUCCESS
