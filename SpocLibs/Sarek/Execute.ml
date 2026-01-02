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
          (* Raw args need type info - for now skip (caller must handle) *)
          ()
      | ArgDeviceBuffer buf ->
          (* V2 Vector buffer - let the backend bind itself *)
          let (module DB : Vector.DEVICE_BUFFER) = buf in
          DB.bind_to_kernel ~kargs:(Obj.repr kargs) ~idx:i)
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
          ()
      | ArgDeviceBuffer buf ->
          (* V2 Vector buffer - let the backend bind itself *)
          let (module DB : Vector.DEVICE_BUFFER) = buf in
          DB.bind_to_kernel ~kargs:(Obj.repr kargs) ~idx:i)
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
          (* Direct path: call native function or interpret IR *)
          let obj_args = args_to_obj_array args in
          let ir_obj = Option.map (fun l -> Obj.repr (Lazy.force l)) ir in
          B.execute_direct ~native_fn ~ir:ir_obj ~block ~grid obj_args
      | Framework_sig.Custom ->
          (* Custom path: delegate to backend with IR *)
          let obj_args = args_to_obj_array args in
          let ir_obj = Option.map (fun l -> Obj.repr (Lazy.force l)) ir in
          B.execute_direct ~native_fn ~ir:ir_obj ~block ~grid obj_args)
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
  match device.framework with
  | "CUDA" ->
      Log.debug Log.Execute "  generating CUDA source..." ;
      let source = Sarek_ir_cuda.generate_with_types ~types:ir.kern_types ir in
      Log.debugf Log.Execute "  CUDA source (%d bytes)" (String.length source) ;
      run_typed ~device ~name:ir.kern_name ~source ~block ~grid ~shared_mem args
  | "OpenCL" ->
      let t0 = Unix.gettimeofday () in
      Log.debug Log.Execute "  generating OpenCL source..." ;
      Log.debugf
        Log.Execute
        "  kern_types count: %d"
        (List.length ir.kern_types) ;
      let source =
        Sarek_ir_opencl.generate_with_types ~types:ir.kern_types ir
      in
      let t1 = Unix.gettimeofday () in
      Log.debugf
        Log.Execute
        "  OpenCL source (%d bytes, gen=%.3fms)"
        (String.length source)
        ((t1 -. t0) *. 1000.0) ;
      Log.debugf Log.Execute "  OpenCL code:\n%s" source ;
      run_typed ~device ~name:ir.kern_name ~source ~block ~grid ~shared_mem args ;
      let t2 = Unix.gettimeofday () in
      Log.debugf Log.Execute "  run_typed took %.3fms" ((t2 -. t1) *. 1000.0)
  | "Native" -> (
      (* Native path: use native_fn directly *)
      match native_fn with
      | Some fn ->
          let obj_args = args_to_obj_array args in
          fn ~block ~grid obj_args
      | None -> raise (Execution_error "Native backend requires native_fn"))
  | "Interpreter" ->
      (* Interpreter path: use Sarek_ir_interp directly.
         Note: For interpreter, use Sarek_ir_interp.run_kernel directly
         with pre-prepared value arrays rather than going through Execute. *)
      Log.debug Log.Execute "  interpreting IR..." ;
      (* Convert Execute.arg list to interpreter format - scalars only *)
      let interp_args =
        List.mapi
          (fun i arg ->
            let name = Printf.sprintf "param%d" i in
            match arg with
            | ArgInt32 n ->
                (name, Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VInt32 n))
            | ArgInt64 n ->
                (name, Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VInt64 n))
            | ArgFloat32 f ->
                (name, Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VFloat32 f))
            | ArgFloat64 f ->
                (name, Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VFloat64 f))
            | ArgDeviceBuffer _ ->
                (* Device buffers not directly supported - use interpreter API directly *)
                failwith
                  "Interpreter: use Sarek_ir_interp.run_kernel for array args"
            | _ -> failwith "Interpreter: unsupported arg type")
          args
      in
      Sarek_ir_interp.run_kernel
        ir
        ~block:(block.x, block.y, block.z)
        ~grid:(grid.x, grid.y, grid.z)
        interp_args
  | fw ->
      raise
        (Execution_error
           ("IR-based execution not supported for framework: " ^ fw))

(** {1 V2 Vector Execution} *)

(** V2 Vector argument type - supports automatic transfers and length expansion
*)
type vector_arg =
  | Vec : ('a, 'b) Vector.t -> vector_arg
      (** V2 Vector - expands to (buffer, length) *)
  | Int : int -> vector_arg  (** Integer scalar *)
  | Int32 : int32 -> vector_arg  (** 32-bit integer scalar *)
  | Int64 : int64 -> vector_arg  (** 64-bit integer scalar *)
  | Float32 : float -> vector_arg  (** 32-bit float scalar *)
  | Float64 : float -> vector_arg  (** 64-bit float scalar *)

(** Get device buffer for a V2 Vector *)
let get_device_buffer (type a b) (v : (a, b) Vector.t) (dev : Device.t) :
    (module Vector.DEVICE_BUFFER) =
  match Vector.get_buffer v dev with
  | Some buf -> buf
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

(** Mark all vectors as Stale_CPU after kernel execution (GPU backends only).
    Native backend doesn't need this since CPU memory is directly modified. *)
let mark_vectors_stale (args : vector_arg list) (dev : Device.t) : unit =
  (* Only mark stale for GPU backends *)
  if dev.framework = "Native" then ()
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
    ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims) : unit =
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

(** Execute a kernel with V2 Vectors. Auto-transfers, expands args,
    compiles/runs. *)
let run_vectors ~(device : Device.t) ~(ir : Sarek_ir.kernel)
    ~(args : vector_arg list) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ?(shared_mem : int = 0)
    ?(native_fn :
       (block:Framework_sig.dims ->
       grid:Framework_sig.dims ->
       Obj.t array ->
       unit)
       option) () : unit =
  (* Special path for CPU backends - work directly with vectors via interpreter *)
  if device.framework = "Interpreter" || device.framework = "Native" then
    run_interpreter_vectors ~ir ~args ~block ~grid
  else begin
    (* 1. Transfer all vectors to device *)
    transfer_vectors_to_device args device ;

    (* 2. Expand vector args to (buffer, length) pairs *)
    let expanded_args = expand_vector_args args device in

    (* 3. Dispatch to appropriate backend *)
    run_from_ir ~device ~ir ~block ~grid ~shared_mem ?native_fn expanded_args ;

    (* 4. Mark vectors as stale (GPU wrote to them, CPU copy is outdated) *)
    mark_vectors_stale args device
  end

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
