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

open Spoc_framework
open Spoc_framework_registry
open Spoc_core

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
  | ArgDeviceBuffer of (module Vector.DEVICE_BUFFER)
(* V2 Vector buffer - backend binds itself *)

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
         | ArgRaw o -> o
         | ArgDeviceBuffer buf ->
             let (module B : Vector.DEVICE_BUFFER) = buf in
             Obj.repr B.ptr)
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
          (* Raw args not handled in JIT path - use expand_vector_args *)
          failwith "bind_args: ArgRaw not supported, use expand_vector_args"
      | ArgDeviceBuffer buf ->
          (* V2 Vector buffer - let the backend bind itself *)
          let (module DB : Vector.DEVICE_BUFFER) = buf in
          DB.bind_to_kernel ~kargs:(Obj.repr kargs) ~idx:i)
    args

(** {1 V2 Vector Argument Type} *)

(** V2 Vector argument type - supports automatic transfers and length expansion.
    Defined before run so it can be used in the signature. *)
type vector_arg =
  | Vec : ('a, 'b) Vector.t -> vector_arg
      (** V2 Vector - expands to (buffer, length) for JIT *)
  | Int : int -> vector_arg  (** Integer scalar *)
  | Int32 : int32 -> vector_arg  (** 32-bit integer scalar *)
  | Int64 : int64 -> vector_arg  (** 64-bit integer scalar *)
  | Float32 : float -> vector_arg  (** 32-bit float scalar *)
  | Float64 : float -> vector_arg  (** 64-bit float scalar *)

(** Convert vector_arg list to Obj.t array for backend dispatch. Passes V2
    Vectors directly - backends handle extraction. *)
let vector_args_to_obj_array (args : vector_arg list) : Obj.t array =
  Array.of_list
    (List.map
       (function
         | Vec v -> Obj.repr v
         | Int n -> Obj.repr (Int32.of_int n)
         | Int32 n -> Obj.repr n
         | Int64 n -> Obj.repr n
         | Float32 f -> Obj.repr f
         | Float64 f -> Obj.repr f)
       args)

(** Get device buffer for a V2 Vector *)
let get_device_buffer (type a b) (v : (a, b) Vector.t) (dev : Device.t) :
    (module Vector.DEVICE_BUFFER) =
  Log.debugf Log.Execute "get_device_buffer for dev=%d" dev.Device.id ;
  match Vector.get_buffer v dev with
  | Some buf ->
      let (module B : Vector.DEVICE_BUFFER) = buf in
      Log.debugf
        Log.Execute
        "got buffer: ptr=%Ld size=%d"
        (Int64.of_nativeint B.ptr)
        B.size ;
      buf
  | None -> failwith "Vector has no device buffer"

(** Transfer all V2 Vector args to device *)
let transfer_vectors_to_device (args : vector_arg list) (dev : Device.t) : unit
    =
  List.iter (function Vec v -> Transfer.to_device v dev | _ -> ()) args

(** Expand V2 Vector args to (buffer, length) pairs for kernel binding. Each Vec
    expands to two args: buffer and length (matches OpenCL/CUDA codegen which
    adds a length parameter for each vector). *)
let expand_vector_args (args : vector_arg list) (dev : Device.t) : arg list =
  List.concat_map
    (function
      | Vec v ->
          let buf = get_device_buffer v dev in
          [ArgDeviceBuffer buf; ArgInt32 (Int32.of_int (Vector.length v))]
      | Int n -> [ArgInt32 (Int32.of_int n)]
      | Int32 n -> [ArgInt32 n]
      | Int64 n -> [ArgInt64 n]
      | Float32 f -> [ArgFloat32 f]
      | Float64 f -> [ArgFloat64 f])
    args

(** Expand vector args to run_source_arg format.
    @param inject_lengths
      If true (default), auto-inject vector length as Int32 after each buffer.
      This matches Sarek-generated kernels which expect (ptr, len) pairs. Set to
      false for external kernels with different signatures. *)
let expand_to_run_source_args ?(inject_lengths = true) (args : vector_arg list)
    (dev : Device.t) : Framework_sig.run_source_arg list =
  List.concat_map
    (function
      | Vec v ->
          let buf = get_device_buffer v dev in
          let (module B : Vector.DEVICE_BUFFER) = buf in
          let len = Vector.length v in
          let buf_arg =
            Framework_sig.RSA_Buffer {binder = B.bind_to_kernel; length = len}
          in
          if inject_lengths then
            [buf_arg; Framework_sig.RSA_Int32 (Int32.of_int len)]
          else [buf_arg]
      | Int n -> [Framework_sig.RSA_Int32 (Int32.of_int n)]
      | Int32 n -> [Framework_sig.RSA_Int32 n]
      | Int64 n -> [Framework_sig.RSA_Int64 n]
      | Float32 f -> [Framework_sig.RSA_Float32 f]
      | Float64 f -> [Framework_sig.RSA_Float64 f])
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
    @param vector_args Original V2 vector args (optional, for unified dispatch)
    @param args Kernel arguments (typed) - used if vector_args not provided
    @raise Execution_error if execution fails *)
let run ~(device : Device.t) ~(name : string)
    ~(ir : Sarek_ir.kernel Lazy.t option)
    ~(native_fn :
       (block:Framework_sig.dims ->
       grid:Framework_sig.dims ->
       Obj.t array ->
       unit)
       option) ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
    ?(shared_mem : int = 0) ?vector_args (args : arg list) : unit =
  match Framework_registry.find_backend device.framework with
  | None ->
      raise
        (Execution_error ("No backend found for framework: " ^ device.framework))
  | Some (module B : Framework_sig.BACKEND) -> (
      match B.execution_model with
      | Framework_sig.JIT -> (
          (* JIT path: generate source, compile, launch
             For JIT, expand vector_args to (buffer, length) format *)
          let expanded_args =
            match vector_args with
            | Some vargs -> expand_vector_args vargs device
            | None -> args
          in
          match ir with
          | None ->
              raise
                (Execution_error "JIT backend requires IR but none provided")
          | Some ir_lazy -> (
              let ir = Lazy.force ir_lazy in
              match B.generate_source ~block ir with
              | None ->
                  raise
                    (Execution_error "JIT backend failed to generate source")
              | Some source ->
                  let dev = B.Device.get device.backend_id in
                  let compiled = B.Kernel.compile_cached dev ~name ~source in
                  let kargs = B.Kernel.create_args () in
                  bind_args (module B) kargs expanded_args ;
                  B.Kernel.launch
                    compiled
                    ~args:kargs
                    ~grid
                    ~block
                    ~shared_mem
                    ~stream:None))
      | Framework_sig.Direct ->
          (* Direct path: call native function or interpret IR
             For Direct, pass V2 Vectors directly (not expanded) *)
          let dev = B.Device.get device.backend_id in
          B.Device.set_current dev ;
          let obj_args =
            match vector_args with
            | Some vargs -> vector_args_to_obj_array vargs
            | None -> args_to_obj_array args
          in
          let ir_val = Option.map Lazy.force ir in
          B.execute_direct ~native_fn ~ir:ir_val ~block ~grid obj_args
      | Framework_sig.Custom ->
          (* Custom path: delegate to backend with IR
             For Custom (Interpreter), pass V2 Vectors directly *)
          let dev = B.Device.get device.backend_id in
          B.Device.set_current dev ;
          let obj_args =
            match vector_args with
            | Some vargs -> vector_args_to_obj_array vargs
            | None -> args_to_obj_array args
          in
          let ir_val = Option.map Lazy.force ir in
          B.execute_direct ~native_fn ~ir:ir_val ~block ~grid obj_args)

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
      let t0 = Unix.gettimeofday () in
      let dev = B.Device.get device.backend_id in
      let compiled = B.Kernel.compile_cached dev ~name ~source in
      let t1 = Unix.gettimeofday () in
      let kargs = B.Kernel.create_args () in
      bind_args (module B) kargs args ;
      let t2 = Unix.gettimeofday () in
      B.Kernel.launch compiled ~args:kargs ~grid ~block ~shared_mem ~stream:None ;
      let t3 = Unix.gettimeofday () in
      Log.debugf
        Log.Execute
        "    compile=%.3fms bind=%.3fms launch=%.3fms"
        ((t1 -. t0) *. 1000.0)
        ((t2 -. t1) *. 1000.0)
        ((t3 -. t2) *. 1000.0)

(** {1 IR-based Execution} *)

(** Execute a kernel from Sarek IR (Phase 4 path). Dispatches to the appropriate
    backend via plugin registry. JIT backends (CUDA, OpenCL) use
    B.generate_source. Direct/Custom backends (Native, Interpreter) use
    B.execute_direct. *)
let run_from_ir ~(device : Device.t) ~(ir : Sarek_ir.kernel)
    ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
    ?(shared_mem : int = 0)
    ?(native_fn :
       (block:Framework_sig.dims ->
       grid:Framework_sig.dims ->
       Obj.t array ->
       unit)
       option) (args : arg list) : unit =
  Log.debugf
    Log.Execute
    "run_from_ir: kernel='%s' framework=%s device=%d"
    ir.kern_name
    device.framework
    device.id ;
  Log.debugf
    Log.Execute
    "  grid=(%d,%d,%d) block=(%d,%d,%d) args=%d"
    grid.x
    grid.y
    grid.z
    block.x
    block.y
    block.z
    (List.length args) ;

  (* Dispatch via plugin registry - no hardcoded framework checks *)
  run
    ~device
    ~name:ir.kern_name
    ~ir:(Some (lazy ir))
    ~native_fn
    ~block
    ~grid
    ~shared_mem
    args

(** {1 V2 Vector Execution Helpers} *)

(** Mark all vectors as Stale_CPU after kernel execution. CPU backends with
    zero-copy don't need this since host memory is directly modified. Uses
    device capabilities, not framework names. *)
let mark_vectors_stale (args : vector_arg list) (dev : Device.t) : unit =
  (* CPU devices use zero-copy - no stale marking needed *)
  if dev.capabilities.is_cpu then ()
  else
    List.iter
      (function
        | Vec v -> (
            (* Mark as Stale_CPU: GPU has authoritative data, CPU is stale *)
            match v.Vector.location with
            | Vector.Both _ -> v.Vector.location <- Vector.Stale_CPU dev
            | _ -> ())
        | _ -> ())
      args

(** Convert a V2 Vector to interpreter value array *)
let vector_to_interp_array : type a b.
    (a, b) Vector.t -> Sarek_ir_interp.value array =
 fun vec ->
  let len = Vector.length vec in
  match Vector.kind vec with
  | Vector.Scalar Vector.Int32 ->
      Array.init len (fun i -> Sarek_ir_interp.VInt32 (Vector.get vec i))
  | Vector.Scalar Vector.Int64 ->
      Array.init len (fun i -> Sarek_ir_interp.VInt64 (Vector.get vec i))
  | Vector.Scalar Vector.Float32 ->
      Array.init len (fun i -> Sarek_ir_interp.VFloat32 (Vector.get vec i))
  | Vector.Scalar Vector.Float64 ->
      Array.init len (fun i -> Sarek_ir_interp.VFloat64 (Vector.get vec i))
  | _ ->
      (* For other types, use float32 as default *)
      Array.init len (fun i ->
          Sarek_ir_interp.VFloat32 (Obj.magic (Vector.get vec i) : float))

(** Copy interpreter value array back to V2 Vector *)
let interp_array_to_vector : type a b.
    Sarek_ir_interp.value array -> (a, b) Vector.t -> unit =
 fun arr vec ->
  let len = min (Array.length arr) (Vector.length vec) in
  match Vector.kind vec with
  | Vector.Scalar Vector.Int32 ->
      for i = 0 to len - 1 do
        Vector.set vec i (Sarek_ir_interp.to_int32 arr.(i))
      done
  | Vector.Scalar Vector.Int64 ->
      for i = 0 to len - 1 do
        Vector.set vec i (Sarek_ir_interp.to_int64 arr.(i))
      done
  | Vector.Scalar Vector.Float32 ->
      for i = 0 to len - 1 do
        Vector.set vec i (Sarek_ir_interp.to_float32 arr.(i))
      done
  | Vector.Scalar Vector.Float64 ->
      for i = 0 to len - 1 do
        Vector.set vec i (Sarek_ir_interp.to_float64 arr.(i))
      done
  | _ ->
      for i = 0 to len - 1 do
        Vector.set vec i (Obj.magic (Sarek_ir_interp.to_float32 arr.(i)))
      done

(** Run kernel via interpreter with V2 Vectors. Note: Interpreter works with IR
    params directly - one arg per param. Vectors map to ArgArray (length is
    intrinsic to array). *)
let run_interpreter_vectors ~(ir : Sarek_ir.kernel) ~(args : vector_arg list)
    ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
    ~(parallel : bool) : unit =
  (* Set interpreter parallel mode *)
  Sarek_ir_interp.parallel_mode := parallel ;
  (* Convert vector args to interpreter format, tracking arrays for writeback *)
  let interp_arrays : (Obj.t * Sarek_ir_interp.value array) list ref = ref [] in

  (* Extract param names from kernel IR (only DParam entries) *)
  let param_names =
    List.filter_map
      (function
        | Sarek_ir.DParam (v, _) -> Some v.Sarek_ir.var_name | _ -> None)
      ir.Sarek_ir.kern_params
  in

  (* Build args matching kernel params 1:1, using actual param names *)
  let interp_args =
    List.mapi
      (fun i arg ->
        let name =
          if i < List.length param_names then List.nth param_names i
          else Printf.sprintf "param%d" i
        in
        match arg with
        | Vec v ->
            let arr = vector_to_interp_array v in
            interp_arrays := (Obj.repr v, arr) :: !interp_arrays ;
            (name, Sarek_ir_interp.ArgArray arr)
        | Int n ->
            ( name,
              Sarek_ir_interp.ArgScalar
                (Sarek_ir_interp.VInt32 (Int32.of_int n)) )
        | Int32 n -> (name, Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VInt32 n))
        | Int64 n -> (name, Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VInt64 n))
        | Float32 f ->
            (name, Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VFloat32 f))
        | Float64 f ->
            (name, Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VFloat64 f)))
      args
  in

  (* Run interpreter *)
  Sarek_ir_interp.run_kernel
    ir
    ~block:(block.x, block.y, block.z)
    ~grid:(grid.x, grid.y, grid.z)
    interp_args ;

  (* Copy results back to vectors - use Obj.magic to recover the type *)
  List.iter
    (fun (vec_obj, arr) ->
      (* We stored the vector as Obj.t, recover it with magic *)
      let vec : (float, Bigarray.float32_elt) Vector.t = Obj.magic vec_obj in
      interp_array_to_vector arr vec)
    !interp_arrays

(** Execute a kernel with V2 Vectors. Auto-transfers, dispatches to backend.
    Unified path for all backends - run handles expansion based on backend. *)
let run_vectors ~(device : Device.t) ~(ir : Sarek_ir.kernel)
    ~(args : vector_arg list) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ?(shared_mem : int = 0) () : unit =
  (* Unified path for all backends:
     1. Transfer to device (CPU backends: zero-copy, no actual transfer)
     2. Pass vector_args to run (it expands for JIT, passes direct for Direct)
     3. Mark stale (CPU backends: no-op due to zero-copy) *)

  (* 1. Transfer all vectors to device *)
  transfer_vectors_to_device args device ;

  (* 2. Dispatch via run with vector_args - it handles expansion per backend *)
  run
    ~device
    ~name:ir.kern_name
    ~ir:(Some (lazy ir))
    ~native_fn:None
    ~block
    ~grid
    ~shared_mem
    ~vector_args:args
    [] ;

  (* 3. Mark vectors as stale (no-op for CPU backends due to zero-copy) *)
  mark_vectors_stale args device

(** Sync all V2 Vector outputs back to CPU *)
let sync_vectors_to_cpu (args : vector_arg list) : unit =
  List.iter (function Vec v -> Transfer.to_cpu v | _ -> ()) args

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

(** Calculate 1D grid dimensions for a problem size *)
let grid_for ~problem_size ~block_size =
  dims1d (grid_for_size ~problem_size ~block_size)

(** {1 External Kernel Execution} *)

(** Re-export source language type *)
type source_lang = Framework_sig.source_lang =
  | CUDA_Source
  | OpenCL_Source
  | PTX
  | SPIR_V

(** Check if a device supports a given source language *)
let supports_lang (dev : Device.t) (lang : source_lang) : bool =
  match Framework_registry.find_backend dev.framework with
  | Some (module B : Framework_sig.BACKEND) ->
      List.mem lang B.supported_source_langs
  | None -> false

(** Execute an external kernel from source code.

    This function allows running pre-written GPU kernels (CUDA, OpenCL, PTX)
    directly without going through the Sarek DSL.

    @param device Target device
    @param source Kernel source code as string
    @param lang Source language (CUDA_Source, OpenCL_Source, PTX)
    @param kernel_name Name of the kernel function in the source
    @param block Block dimensions
    @param grid Grid dimensions
    @param shared_mem Shared memory size in bytes (default 0)
    @param inject_lengths
      If true (default), auto-inject vector length as Int32 after each buffer
      argument. Sarek-generated kernels expect (ptr, len) pairs. Set to false
      for external kernels that don't follow this convention.
    @param args Kernel arguments as vector_arg list
    @raise Execution_error if device doesn't support the source language *)
let run_source ~(device : Device.t) ~(source : string) ~(lang : source_lang)
    ~(kernel_name : string) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ?(shared_mem : int = 0)
    ?(inject_lengths : bool = true) (args : vector_arg list) : unit =
  (* Transfer vectors to device first *)
  transfer_vectors_to_device args device ;

  match Framework_registry.find_backend device.framework with
  | Some (module B : Framework_sig.BACKEND) ->
      if not (List.mem lang B.supported_source_langs) then
        raise
          (Execution_error
             (Printf.sprintf
                "%s backend does not support %s"
                device.framework
                (match lang with
                | CUDA_Source -> "CUDA source"
                | OpenCL_Source -> "OpenCL source"
                | PTX -> "PTX"
                | SPIR_V -> "SPIR-V"))) ;

      (* Expand vector args to run_source_arg format for external kernels *)
      let rs_args = expand_to_run_source_args ~inject_lengths args device in

      (* Set current device and run *)
      let dev = B.Device.get device.backend_id in
      B.Device.set_current dev ;
      B.run_source ~source ~lang ~kernel_name ~block ~grid ~shared_mem rs_args ;

      (* Mark vectors as stale *)
      mark_vectors_stale args device
  | None ->
      raise
        (Execution_error
           ("No V2 backend found for framework: " ^ device.framework))

(** Load kernel source from a file *)
let load_source (path : string) : string =
  In_channel.with_open_text path In_channel.input_all

(** Detect source language from file extension *)
let detect_lang (path : string) : source_lang =
  if String.ends_with ~suffix:".cu" path then CUDA_Source
  else if String.ends_with ~suffix:".cl" path then OpenCL_Source
  else if String.ends_with ~suffix:".ptx" path then PTX
  else if String.ends_with ~suffix:".spv" path then SPIR_V
  else failwith ("Unknown source file extension: " ^ path)

(** Execute an external kernel from a file. Source language is detected from
    file extension (.cu, .cl, .ptx, .spv) *)
let run_source_file ~(device : Device.t) ~(path : string)
    ~(kernel_name : string) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ?(shared_mem : int = 0)
    ?(inject_lengths : bool = true) (args : vector_arg list) : unit =
  let source = load_source path in
  let lang = detect_lang path in
  run_source
    ~device
    ~source
    ~lang
    ~kernel_name
    ~block
    ~grid
    ~shared_mem
    ~inject_lengths
    args
