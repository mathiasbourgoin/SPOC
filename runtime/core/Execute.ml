(******************************************************************************
 * Execute - Unified Kernel Execution Dispatcher
 *
 * Provides a unified interface for executing Sarek kernels across different
 * backends. Dispatches based on the backend's execution model:
 *
 * - JIT (CUDA, OpenCL): Generate source → compile → launch
 * - Direct (Native): Call pre-compiled OCaml function
 * - Custom: Delegate to backend's custom pipeline
 ******************************************************************************)

open Sarek_framework

(** Kernel execution error *)
exception Execution_error of string

(** {1 Execution Dispatch} *)

(** Execute a kernel on a device using the unified dispatch mechanism.

    @param device Target device
    @param name Kernel name
    @param ir Sarek IR kernel (lazy, only forced for JIT backends)
    @param native_fn Pre-compiled native function (for Direct backends)
    @param block Block dimensions
    @param grid Grid dimensions
    @param args Kernel arguments
    @raise Execution_error if execution fails *)
let run_v2 ~(device : Device.t) ~(name : string)
    ~(ir : Sarek_ir.kernel Lazy.t option)
    ~(native_fn :
       (block:Framework_sig.dims ->
       grid:Framework_sig.dims ->
       Obj.t array ->
       unit)
       option) ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
    (args : Obj.t array) : unit =
  match Framework_registry.find_backend_v2 device.framework with
  | Some (module B : Framework_sig.BACKEND_V2) -> (
      match B.execution_model with
      | Framework_sig.JIT -> (
          (* JIT path: generate source, compile, launch *)
          match ir with
          | None ->
              raise
                (Execution_error "JIT backend requires IR but none provided")
          | Some ir_lazy -> (
              let ir = Lazy.force ir_lazy in
              match B.generate_source ir with
              | None ->
                  raise
                    (Execution_error "JIT backend failed to generate source")
              | Some source ->
                  let dev = B.Device.get device.backend_id in
                  let compiled = B.Kernel.compile_cached dev ~name ~source in
                  let kargs = B.Kernel.create_args () in
                  (* Bind arguments - simplified, assumes proper ordering *)
                  Array.iteri
                    (fun i arg ->
                      (* Type dispatch based on runtime type *)
                      if Obj.is_int arg then
                        B.Kernel.set_arg_int32 kargs i (Obj.obj arg)
                      else B.Kernel.set_arg_int32 kargs i 0l
                        (* TODO: proper type dispatch *))
                    args ;
                  B.Kernel.launch
                    compiled
                    ~args:kargs
                    ~grid
                    ~block
                    ~shared_mem:0
                    ~stream:None))
      | Framework_sig.Direct ->
          (* Direct path: call native function *)
          B.execute_direct ~native_fn ~block ~grid args
      | Framework_sig.Custom ->
          (* Custom path: delegate to backend *)
          (* For now, try native_fn fallback *)
          B.execute_direct ~native_fn ~block ~grid args)
  | None -> (
      (* Fall back to BACKEND (non-V2) if available *)
      match Framework_registry.find_backend device.framework with
      | None ->
          raise
            (Execution_error
               ("No backend found for framework: " ^ device.framework))
      | Some (module B : Framework_sig.BACKEND) -> (
          (* Legacy path: use old BACKEND interface *)
          match native_fn with
          | Some fn -> fn ~block ~grid args
          | None ->
              raise
                (Execution_error
                   "Legacy backend requires native_fn for execution")))

(** {1 Simple Execution Interface} *)

(** Execute with minimal parameters, for common use cases *)
let run_simple ~(device : Device.t) ~(name : string) ~(source : string)
    ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
    (args : Obj.t array) : unit =
  match Framework_registry.find_backend device.framework with
  | None ->
      raise
        (Execution_error ("No backend found for framework: " ^ device.framework))
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let compiled = B.Kernel.compile_cached dev ~name ~source in
      let kargs = B.Kernel.create_args () in
      (* Simple binding - just set all as int32 for now *)
      Array.iteri (fun i _arg -> B.Kernel.set_arg_int32 kargs i 0l) args ;
      B.Kernel.launch
        compiled
        ~args:kargs
        ~grid
        ~block
        ~shared_mem:0
        ~stream:None

(** {1 Argument Binding Helpers} *)

(** Argument types for explicit binding *)
type arg =
  | ArgBuffer of Memory.buffer
  | ArgInt32 of int32
  | ArgInt64 of int64
  | ArgFloat32 of float
  | ArgFloat64 of float

(** Convert typed arguments to Obj.t array *)
let args_to_obj_array (args : arg list) : Obj.t array =
  Array.of_list
    (List.map
       (function
         | ArgBuffer b -> Obj.repr b
         | ArgInt32 n -> Obj.repr n
         | ArgInt64 n -> Obj.repr n
         | ArgFloat32 f -> Obj.repr f
         | ArgFloat64 f -> Obj.repr f)
       args)

(** Bind typed arguments to a kernel *)
let bind_args (module B : Framework_sig.BACKEND) (kargs : B.Kernel.args)
    (args : arg list) : unit =
  List.iteri
    (fun i arg ->
      match arg with
      | ArgBuffer b ->
          let backend_buf : _ B.Memory.buffer = Obj.obj b.Memory.handle in
          B.Kernel.set_arg_buffer kargs i backend_buf
      | ArgInt32 n -> B.Kernel.set_arg_int32 kargs i n
      | ArgInt64 n -> B.Kernel.set_arg_int64 kargs i n
      | ArgFloat32 f -> B.Kernel.set_arg_float32 kargs i f
      | ArgFloat64 f -> B.Kernel.set_arg_float64 kargs i f)
    args

(** {1 Typed Execution Interface} *)

(** Execute a kernel with explicitly typed arguments *)
let run_typed ~(device : Device.t) ~(name : string) ~(source : string)
    ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims) (args : arg list)
    : unit =
  match Framework_registry.find_backend device.framework with
  | None ->
      raise
        (Execution_error ("No backend found for framework: " ^ device.framework))
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let compiled = B.Kernel.compile_cached dev ~name ~source in
      let kargs = B.Kernel.create_args () in
      bind_args (module B) kargs args ;
      B.Kernel.launch
        compiled
        ~args:kargs
        ~grid
        ~block
        ~shared_mem:0
        ~stream:None

(** {1 IR-based Execution} *)

(** Execute a kernel from Sarek IR (Phase 4 path) *)
let run_from_ir ~(device : Device.t) ~(ir : Sarek_ir.kernel)
    ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims) (args : arg list)
    : unit =
  let source =
    match device.framework with
    | "CUDA" -> Sarek_ir_cuda.generate ir
    | "OpenCL" -> Sarek_ir_opencl.generate ir
    | fw ->
        raise
          (Execution_error
             ("IR-based execution not supported for framework: " ^ fw))
  in
  run_typed ~device ~name:ir.kern_name ~source ~block ~grid args

(** {1 Convenience Functions} *)

(** Create 1D grid and block dimensions *)
let dims1d size = Framework_sig.dims_1d size

(** Create 2D grid and block dimensions *)
let dims2d x y = Framework_sig.dims_2d x y

(** Create 3D grid and block dimensions *)
let dims3d x y z = Framework_sig.dims_3d x y z

(** Calculate grid size for a given problem size and block size *)
let grid_for_size ~problem_size ~block_size =
  (problem_size + block_size - 1) / block_size
