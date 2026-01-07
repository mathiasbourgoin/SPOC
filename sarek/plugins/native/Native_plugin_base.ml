(******************************************************************************
 * Native Plugin - CPU Execution Backend
 *
 * Implements the Framework_sig.BACKEND interface for CPU execution.
 * This is a Direct backend - no JIT compilation needed.
 *
 * Kernels are pre-compiled OCaml functions registered by the PPX at module
 * load time. The "compile" step is a no-op - we look up registered functions.
 ******************************************************************************)

open Spoc_framework

(** Registry for native kernel functions.

    Kernels are registered by name. The function signature uses exec_arg array
    for type-safe argument passing (no Obj.t):
    - args: Framework_sig.exec_arg array (typed kernel arguments)
    - grid: (gx, gy, gz) grid dimensions
    - block: (bx, by, bz) block dimensions

    Defined at module level so it can be accessed by both the Native module and
    the external registration functions. *)
let native_kernels :
    ( string,
      Framework_sig.exec_arg array -> int * int * int -> int * int * int -> unit
    )
    Hashtbl.t =
  Hashtbl.create 16

module Native : sig
  val name : string

  val version : int * int * int

  module Device : sig
    type t

    type id = int

    val init : unit -> unit

    val count : unit -> int

    val get : int -> t

    val id : t -> id

    val name : t -> string

    val capabilities : t -> Framework_sig.capabilities

    val set_current : t -> unit

    val synchronize : t -> unit
  end

  module Memory : sig
    type 'a buffer

    val alloc : Device.t -> int -> ('a, 'b) Bigarray.kind -> 'a buffer

    val alloc_custom : Device.t -> size:int -> elem_size:int -> 'a buffer

    val alloc_zero_copy :
      Device.t ->
      ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
      ('a, 'b) Bigarray.kind ->
      'a buffer option

    val is_zero_copy : 'a buffer -> bool

    val free : 'a buffer -> unit

    val host_to_device :
      src:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> dst:'a buffer -> unit

    val device_to_host :
      src:'a buffer -> dst:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> unit

    val host_ptr_to_device :
      src_ptr:unit Ctypes.ptr -> byte_size:int -> dst:'a buffer -> unit

    val device_to_host_ptr :
      src:'a buffer -> dst_ptr:unit Ctypes.ptr -> byte_size:int -> unit

    val device_to_device : src:'a buffer -> dst:'a buffer -> unit

    val size : 'a buffer -> int

    val device_ptr : 'a buffer -> nativeint
  end

  module Stream : sig
    type t

    val create : Device.t -> t

    val destroy : t -> unit

    val synchronize : t -> unit

    val default : Device.t -> t
  end

  module Event : sig
    type t

    val create : unit -> t

    val destroy : t -> unit

    val record : t -> Stream.t -> unit

    val synchronize : t -> unit

    val elapsed : start:t -> stop:t -> float
  end

  module Kernel : sig
    type t

    type args

    val compile : Device.t -> name:string -> source:string -> t

    val compile_cached : Device.t -> name:string -> source:string -> t

    val create_args : unit -> args

    val set_arg_buffer : args -> int -> _ Memory.buffer -> unit

    val set_arg_int32 : args -> int -> int32 -> unit

    val set_arg_int64 : args -> int -> int64 -> unit

    val set_arg_float32 : args -> int -> float -> unit

    val set_arg_float64 : args -> int -> float -> unit

    val set_arg_ptr : args -> int -> nativeint -> unit

    val launch :
      t ->
      args:args ->
      grid:Framework_sig.dims ->
      block:Framework_sig.dims ->
      shared_mem:int ->
      stream:Stream.t option ->
      unit

    val clear_cache : unit -> unit
  end

  val enable_profiling : unit -> unit

  val disable_profiling : unit -> unit

  val is_available : unit -> bool
end = struct
  let name = "Native"

  let version = (1, 0, 0)

  module Device = struct
    type t = {id : int; name : string; num_cores : int; parallel : bool}

    type id = int

    let devices : t array ref = ref [||]

    let current : t option ref = ref None

    let init () =
      if Array.length !devices = 0 then begin
        let num_cores = try Domain.recommended_domain_count () with _ -> 1 in
        devices :=
          [|
            {
              id = 0;
              name = Printf.sprintf "CPU Native (Parallel, %d cores)" num_cores;
              num_cores;
              parallel = true;
            };
          |]
      end

    let count () = Array.length !devices

    let get idx =
      if idx < 0 || idx >= Array.length !devices then
        failwith (Printf.sprintf "Native.Device.get: invalid index %d" idx)
      else !devices.(idx)

    let id d = d.id

    let name d = d.name

    let capabilities d : Framework_sig.capabilities =
      (* Get approximate system memory - use 16GB as reasonable default *)
      let total_mem =
        try
          let ic = open_in "/proc/meminfo" in
          let line = input_line ic in
          close_in ic ;
          (* Parse "MemTotal:       32657436 kB" *)
          Scanf.sscanf line "MemTotal: %Ld kB" (fun kb -> Int64.mul kb 1024L)
        with _ -> Int64.of_int (16 * 1024 * 1024 * 1024)
        (* 16 GB default *)
      in
      {
        max_threads_per_block = d.num_cores;
        max_block_dims = (d.num_cores, 1, 1);
        max_grid_dims = (max_int, max_int, max_int);
        shared_mem_per_block = 1024 * 1024 * 1024;
        total_global_mem = total_mem;
        compute_capability = (0, 0);
        (* Native has no compute capability *)
        supports_fp64 = true;
        supports_atomics = true;
        warp_size = 1;
        max_registers_per_block = 0;
        clock_rate_khz = 0;
        multiprocessor_count = d.num_cores;
        is_cpu = true;
      }

    let set_current d = current := Some d

    let synchronize _d = () (* CPU is always synchronized *)
  end

  module Memory = struct
    (* For CPU, buffers are just bigarrays or raw ctypes memory.
       No actual device transfer - we just keep pointers. *)
    type storage_kind = Bigarray_buf | Ctypes_buf

    type 'a buffer = {
      data : Obj.t;
      size : int;
      elem_size : int;
      device : Device.t;
      storage : storage_kind;
    }

    let alloc device size kind =
      let arr = Bigarray.Array1.create kind Bigarray.c_layout size in
      let elem_size = Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind) in
      {data = Obj.repr arr; size; elem_size; device; storage = Bigarray_buf}

    (** Allocate buffer for custom types with explicit element size in bytes.
        For native, we allocate raw ctypes memory. *)
    let alloc_custom device ~size ~elem_size =
      let bytes = size * elem_size in
      let ptr = Ctypes.allocate_n Ctypes.char ~count:bytes in
      {data = Obj.repr ptr; size; elem_size; device; storage = Ctypes_buf}

    (** For Native CPU, zero-copy is the default - we just wrap the bigarray *)
    let alloc_zero_copy device ba _kind =
      let size = Bigarray.Array1.dim ba in
      let elem_size = Bigarray.kind_size_in_bytes (Bigarray.Array1.kind ba) in
      Some {data = Obj.repr ba; size; elem_size; device; storage = Bigarray_buf}

    let is_zero_copy _buf = true (* Native is always zero-copy *)

    let free _buf = ()

    let host_to_device ~src ~dst =
      let dst_arr = Obj.obj dst.data in
      let len = min (Bigarray.Array1.dim src) dst.size in
      Bigarray.Array1.blit
        (Bigarray.Array1.sub src 0 len)
        (Bigarray.Array1.sub dst_arr 0 len)

    let device_to_host ~src ~dst =
      let src_arr = Obj.obj src.data in
      let len = min src.size (Bigarray.Array1.dim dst) in
      Bigarray.Array1.blit
        (Bigarray.Array1.sub src_arr 0 len)
        (Bigarray.Array1.sub dst 0 len)

    (** Transfer from raw pointer to native buffer (for custom types). For
        native, this is a memcpy using ctypes. *)
    let host_ptr_to_device ~src_ptr ~byte_size ~dst =
      let open Ctypes in
      let dst_ptr : char ptr = Obj.obj dst.data in
      let src_char_ptr = from_voidp char src_ptr in
      for i = 0 to byte_size - 1 do
        dst_ptr +@ i <-@ !@(src_char_ptr +@ i)
      done

    (** Transfer from native buffer to raw pointer (for custom types). *)
    let device_to_host_ptr ~src ~dst_ptr ~byte_size =
      let open Ctypes in
      let src_ptr : char ptr = Obj.obj src.data in
      let dst_char_ptr = from_voidp char dst_ptr in
      for i = 0 to byte_size - 1 do
        dst_char_ptr +@ i <-@ !@(src_ptr +@ i)
      done

    let device_to_device ~src ~dst =
      let src_arr = Obj.obj src.data in
      let dst_arr = Obj.obj dst.data in
      let len = min src.size dst.size in
      Bigarray.Array1.blit
        (Bigarray.Array1.sub src_arr 0 len)
        (Bigarray.Array1.sub dst_arr 0 len)

    let size buf = buf.size

    let device_ptr buf =
      match buf.storage with
      | Bigarray_buf ->
          (* Bigarray storage - get pointer from bigarray *)
          let arr : (_, _, Bigarray.c_layout) Bigarray.Array1.t =
            Obj.obj buf.data
          in
          Ctypes.bigarray_start Ctypes.array1 arr |> Ctypes.raw_address_of_ptr
      | Ctypes_buf ->
          (* Ctypes storage - data is already a pointer *)
          let ptr : char Ctypes.ptr = Obj.obj buf.data in
          Ctypes.to_voidp ptr |> Ctypes.raw_address_of_ptr
  end

  module Stream = struct
    type t = unit

    let create _dev = ()

    let destroy _s = ()

    let synchronize _s = ()

    let default _dev = ()
  end

  module Event = struct
    type t = {mutable time : float}

    let create () = {time = 0.0}

    let destroy _e = ()

    let record e _stream = e.time <- Unix.gettimeofday ()

    let synchronize _e = ()

    let elapsed ~start ~stop = (stop.time -. start.time) *. 1000.0
  end

  module Kernel = struct
    (* Native kernels are pre-compiled OCaml functions registered by the PPX.
       The "compile" here is a no-op - we look up registered functions. *)
    type t = {name : string}

    (** Use exec_arg directly - no intermediate type needed! *)
    type args = {mutable list : Framework_sig.exec_arg list}

    let compile _device ~name ~source:_ = {name}

    let compile_cached = compile

    let create_args () = {list = []}

    let set_arg_buffer args _idx buf =
      (* Wrap buffer in EXEC_VECTOR for exec_arg *)
      let module EV : Typed_value.EXEC_VECTOR = struct
        let length = Memory.size buf

        let type_name = "buffer"

        let elem_size = match buf with {elem_size; _} -> elem_size

        let underlying_obj () = Obj.repr buf

        let device_ptr () = Memory.device_ptr buf

        let get _i = failwith "Native buffer: get not implemented"

        let set _i _v = failwith "Native buffer: set not implemented"
      end in
      args.list <- Framework_sig.EA_Vec (module EV) :: args.list

    let set_arg_int32 args _idx v =
      args.list <- Framework_sig.EA_Int32 v :: args.list

    let set_arg_int64 args _idx v =
      args.list <- Framework_sig.EA_Int64 v :: args.list

    let set_arg_float32 args _idx v =
      args.list <- Framework_sig.EA_Float32 v :: args.list

    let set_arg_float64 args _idx v =
      args.list <- Framework_sig.EA_Float64 v :: args.list

    let set_arg_ptr _args _idx _ptr =
      failwith "Native backend does not support raw pointer arguments"

    (** Set a raw OCaml value argument (for SPOC Vector/customarray). Note: Not
        yet implemented with exec_arg. *)
    let[@warning "-32"] set_arg_raw _args _idx _v =
      failwith "Native backend: set_arg_raw not implemented with exec_arg"

    let launch kernel ~args ~(grid : Framework_sig.dims)
        ~(block : Framework_sig.dims) ~shared_mem:_ ~stream:_ =
      match Hashtbl.find_opt native_kernels kernel.name with
      | Some fn ->
          (* Just reverse and convert to array - no Obj.t conversion needed! *)
          let arg_array = args.list |> List.rev |> Array.of_list in
          fn arg_array (grid.x, grid.y, grid.z) (block.x, block.y, block.z)
      | None ->
          failwith
            (Printf.sprintf
               "Native.Kernel.launch: kernel '%s' not registered"
               kernel.name)

    let clear_cache () = Hashtbl.clear native_kernels
  end

  let profiling_enabled = ref false

  let enable_profiling () = profiling_enabled := true

  let disable_profiling () = profiling_enabled := false

  let is_available () = true
end

(* Legacy init retained for compatibility; backend registration now handled by
   Native_plugin. *)
let init () = ()

(** Register a native kernel function by name.

    This is called by PPX-generated code at module load time to register the
    pre-compiled OCaml kernel function. The function signature matches what
    Sarek_cpu_runtime expects:
    - args: Obj.t array of kernel arguments
    - grid: (gx, gy, gz) grid dimensions
    - block: (bx, by, bz) block dimensions

    Example:
    {[
      let () =
        Sarek_native.Native_plugin.register_kernel
          "my_kernel"
          (fun args grid block ->
            Sarek_cpu_runtime.run_threadpool
              ~has_barriers:false
              ~block
              ~grid
              my_kernel_fn
              args)
    ]} *)
let register_kernel name fn = Hashtbl.replace native_kernels name fn

(** Check if a kernel is registered *)
let kernel_registered name = Hashtbl.mem native_kernels name

(** List all registered kernels (for debugging) *)
let list_kernels () =
  Hashtbl.fold (fun name _ acc -> name :: acc) native_kernels []

(** Run a kernel directly with typed arguments. This bypasses the Runtime API.

    @param name The registered kernel name
    @param args Array of exec_arg (typed kernel arguments)
    @param grid (gx, gy, gz) grid dimensions
    @param block (bx, by, bz) block dimensions *)
let run_kernel_raw ~name ~args ~grid ~block =
  match Hashtbl.find_opt native_kernels name with
  | Some fn -> fn args grid block
  | None ->
      failwith
        (Printf.sprintf
           "Native_plugin.run_kernel_raw: kernel '%s' not registered"
           name)
