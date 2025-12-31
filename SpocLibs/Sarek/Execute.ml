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
open Sarek_core

(** Kernel execution error *)
exception Execution_error of string

(** {1 Argument Binding Helpers} *)

(** Argument types for explicit binding. ArgBuffer stores the buffer handle as
    Obj.t to erase the type parameter. *)
type arg =
  | ArgBuffer of Obj.t (* Memory.buffer erased *)
  | ArgInt32 of int32
  | ArgInt64 of int64
  | ArgFloat32 of float
  | ArgFloat64 of float
  | ArgCustom of Obj.t * int (* buffer + elem_size for custom types *)
  | ArgRaw of Obj.t (* passthrough for SPOC Vector compatibility *)

(** Create a buffer argument from any Memory.buffer *)
let arg_buffer (buf : _ Memory.buffer) : arg = ArgBuffer (Obj.repr buf)

(** Create a custom buffer argument with explicit element size *)
let arg_custom (buf : _ Memory.buffer) ~(elem_size : int) : arg =
  ArgCustom (Obj.repr buf, elem_size)

(** Create a raw argument (for SPOC Vector compatibility) *)
let arg_raw (v : 'a) : arg = ArgRaw (Obj.repr v)

(** Convert typed arguments to Obj.t array *)
let args_to_obj_array (args : arg list) : Obj.t array =
  Array.of_list
    (List.map
       (function
         | ArgBuffer b -> b
         | ArgInt32 n -> Obj.repr n
         | ArgInt64 n -> Obj.repr n
         | ArgFloat32 f -> Obj.repr f
         | ArgFloat64 f -> Obj.repr f
         | ArgCustom (b, _) -> b
         | ArgRaw o -> o)
       args)

(** Bind typed arguments to a kernel (for BACKEND) *)
let bind_args (type kargs)
    (module B : Framework_sig.BACKEND with type Kernel.args = kargs)
    (kargs : kargs) (args : arg list) : unit =
  List.iteri
    (fun i arg ->
      match arg with
      | ArgBuffer obj ->
          (* obj is Obj.repr of Memory.buffer, extract the backend handle *)
          let buf : _ Memory.buffer = Obj.obj obj in
          let backend_buf : _ B.Memory.buffer = Obj.obj buf.Memory.handle in
          B.Kernel.set_arg_buffer kargs i backend_buf
      | ArgInt32 n -> B.Kernel.set_arg_int32 kargs i n
      | ArgInt64 n -> B.Kernel.set_arg_int64 kargs i n
      | ArgFloat32 f -> B.Kernel.set_arg_float32 kargs i f
      | ArgFloat64 f -> B.Kernel.set_arg_float64 kargs i f
      | ArgCustom (obj, _elem_size) ->
          (* Same as ArgBuffer - custom types use same buffer mechanism *)
          let buf : _ Memory.buffer = Obj.obj obj in
          let backend_buf : _ B.Memory.buffer = Obj.obj buf.Memory.handle in
          B.Kernel.set_arg_buffer kargs i backend_buf
      | ArgRaw _obj ->
          (* Raw args need type info - for now skip (caller must handle) *)
          ())
    args

(** Bind typed arguments to a kernel (for BACKEND_V2) *)
let bind_args_v2 (type kargs)
    (module B : Framework_sig.BACKEND_V2 with type Kernel.args = kargs)
    (kargs : kargs) (args : arg list) : unit =
  List.iteri
    (fun i arg ->
      match arg with
      | ArgBuffer obj ->
          (* obj is Obj.repr of Memory.buffer, extract the backend handle *)
          let buf : _ Memory.buffer = Obj.obj obj in
          let backend_buf : _ B.Memory.buffer = Obj.obj buf.Memory.handle in
          B.Kernel.set_arg_buffer kargs i backend_buf
      | ArgInt32 n -> B.Kernel.set_arg_int32 kargs i n
      | ArgInt64 n -> B.Kernel.set_arg_int64 kargs i n
      | ArgFloat32 f -> B.Kernel.set_arg_float32 kargs i f
      | ArgFloat64 f -> B.Kernel.set_arg_float64 kargs i f
      | ArgCustom (obj, _elem_size) ->
          (* Same as ArgBuffer - custom types use same buffer mechanism *)
          let buf : _ Memory.buffer = Obj.obj obj in
          let backend_buf : _ B.Memory.buffer = Obj.obj buf.Memory.handle in
          B.Kernel.set_arg_buffer kargs i backend_buf
      | ArgRaw _obj ->
          (* Raw args need type info - for now skip (caller must handle) *)
          ())
    args

(** {1 Execution Dispatch} *)

(** Execute a kernel on a device using the unified dispatch mechanism.

    @param device Target device
    @param name Kernel name
    @param ir Sarek IR kernel (lazy, only forced for JIT backends)
    @param native_fn Pre-compiled native function (for Direct backends)
    @param block Block dimensions
    @param grid Grid dimensions
    @param shared_mem Shared memory size in bytes (default 0)
    @param args Kernel arguments (typed)
    @raise Execution_error if execution fails *)
let run_v2 ~(device : Device.t) ~(name : string)
    ~(ir : Sarek_ir.kernel Lazy.t option)
    ~(native_fn :
       (block:Framework_sig.dims ->
       grid:Framework_sig.dims ->
       Obj.t array ->
       unit)
       option) ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
    ?(shared_mem : int = 0) (args : arg list) : unit =
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
              match B.generate_source (Obj.repr ir) with
              | None ->
                  raise
                    (Execution_error "JIT backend failed to generate source")
              | Some source ->
                  let dev = B.Device.get device.backend_id in
                  let compiled = B.Kernel.compile_cached dev ~name ~source in
                  let kargs = B.Kernel.create_args () in
                  bind_args_v2 (module B) kargs args ;
                  B.Kernel.launch
                    compiled
                    ~args:kargs
                    ~grid
                    ~block
                    ~shared_mem
                    ~stream:None))
      | Framework_sig.Direct ->
          (* Direct path: call native function *)
          let obj_args = args_to_obj_array args in
          B.execute_direct ~native_fn ~block ~grid obj_args
      | Framework_sig.Custom ->
          (* Custom path: delegate to backend *)
          let obj_args = args_to_obj_array args in
          B.execute_direct ~native_fn ~block ~grid obj_args)
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
          | Some fn ->
              let obj_args = args_to_obj_array args in
              fn ~block ~grid obj_args
          | None ->
              raise
                (Execution_error
                   "Legacy backend requires native_fn for execution")))

(** {1 Typed Execution Interface} *)

(** Execute a kernel with explicitly typed arguments *)
let run_typed ~(device : Device.t) ~(name : string) ~(source : string)
    ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
    ?(shared_mem : int = 0) (args : arg list) : unit =
  match Framework_registry.find_backend device.framework with
  | None ->
      raise
        (Execution_error ("No backend found for framework: " ^ device.framework))
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let compiled = B.Kernel.compile_cached dev ~name ~source in
      let kargs = B.Kernel.create_args () in
      bind_args (module B) kargs args ;
      B.Kernel.launch compiled ~args:kargs ~grid ~block ~shared_mem ~stream:None

(** {1 IR-based Execution} *)

(** Execute a kernel from Sarek IR (Phase 4 path) *)
let run_from_ir ~(device : Device.t) ~(ir : Sarek_ir.kernel)
    ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
    ?(shared_mem : int = 0)
    ?(native_fn :
       (block:Framework_sig.dims ->
       grid:Framework_sig.dims ->
       Obj.t array ->
       unit)
       option) (args : arg list) : unit =
  match device.framework with
  | "CUDA" ->
      let source = Sarek_ir_cuda.generate ir in
      run_typed ~device ~name:ir.kern_name ~source ~block ~grid ~shared_mem args
  | "OpenCL" ->
      let source = Sarek_ir_opencl.generate ir in
      run_typed ~device ~name:ir.kern_name ~source ~block ~grid ~shared_mem args
  | "Native" -> (
      (* Native path: use native_fn directly *)
      match native_fn with
      | Some fn ->
          let obj_args = args_to_obj_array args in
          fn ~block ~grid obj_args
      | None -> raise (Execution_error "Native backend requires native_fn"))
  | fw ->
      raise
        (Execution_error
           ("IR-based execution not supported for framework: " ^ fw))

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
