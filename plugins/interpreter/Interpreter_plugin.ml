(******************************************************************************
 * Interpreter Plugin - CPU Interpreter Backend
 *
 * Implements the Framework_sig.BACKEND interface for CPU interpretation.
 * This is a Direct backend that interprets Sarek V2 IR without compilation.
 *
 * Key differences from Native:
 * - Native uses pre-compiled OCaml functions (from PPX)
 * - Interpreter walks the IR tree at runtime (slower, but no compilation)
 ******************************************************************************)

open Sarek_framework

(** Registry for interpreter kernels. Maps kernel name to IR for interpretation.
*)
let interpreter_kernels : (string, Sarek.Sarek_ir.kernel) Hashtbl.t =
  Hashtbl.create 16

module Interpreter : sig
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
  let name = "Interpreter"

  let version = (1, 0, 0)

  module Device = struct
    type t = {id : int; name : string}

    type id = int

    let devices : t array ref = ref [||]

    let current : t option ref = ref None

    let init () =
      if Array.length !devices = 0 then
        devices := [|{id = 0; name = "CPU Interpreter (Sequential)"}|]

    let count () = Array.length !devices

    let get idx =
      if idx < 0 || idx >= Array.length !devices then
        failwith (Printf.sprintf "Interpreter.Device.get: invalid index %d" idx)
      else !devices.(idx)

    let id d = d.id

    let name d = d.name

    let capabilities _d : Framework_sig.capabilities =
      {
        max_threads_per_block = 256;
        (* Reasonable limit for interpretation *)
        max_block_dims = (256, 256, 64);
        max_grid_dims = (max_int, max_int, max_int);
        shared_mem_per_block = 1024 * 1024;
        total_global_mem = Int64.of_int (16 * 1024 * 1024 * 1024);
        compute_capability = (0, 0);
        supports_fp64 = true;
        supports_atomics = true;
        warp_size = 1;
        (* Sequential execution *)
        max_registers_per_block = 0;
        clock_rate_khz = 0;
        multiprocessor_count = 1;
        is_cpu = true;
      }

    let set_current d = current := Some d

    let synchronize _d = ()
  end

  module Memory = struct
    type 'a buffer = {
      data : Obj.t;
      size : int;
      elem_size : int;
      device : Device.t;
    }

    let alloc device size kind =
      let arr = Bigarray.Array1.create kind Bigarray.c_layout size in
      let elem_size = Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind) in
      {data = Obj.repr arr; size; elem_size; device}

    let alloc_custom device ~size ~elem_size =
      let bytes = size * elem_size in
      let ptr = Ctypes.allocate_n Ctypes.char ~count:bytes in
      {data = Obj.repr ptr; size; elem_size; device}

    let alloc_zero_copy device ba _kind =
      let size = Bigarray.Array1.dim ba in
      let elem_size = Bigarray.kind_size_in_bytes (Bigarray.Array1.kind ba) in
      Some {data = Obj.repr ba; size; elem_size; device}

    let is_zero_copy _buf = true

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

    let host_ptr_to_device ~src_ptr ~byte_size ~dst =
      let open Ctypes in
      let dst_ptr : char ptr = Obj.obj dst.data in
      let src_char_ptr = from_voidp char src_ptr in
      for i = 0 to byte_size - 1 do
        dst_ptr +@ i <-@ !@(src_char_ptr +@ i)
      done

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
      let arr : (_, _, Bigarray.c_layout) Bigarray.Array1.t =
        Obj.obj buf.data
      in
      Ctypes.bigarray_start Ctypes.array1 arr |> Ctypes.raw_address_of_ptr
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
    type t = {name : string}

    type arg =
      | ArgBuffer of Obj.t
      | ArgInt32 of int32
      | ArgInt64 of int64
      | ArgFloat32 of float
      | ArgFloat64 of float

    type args = {mutable list : arg list}

    let compile _device ~name ~source:_ = {name}

    let compile_cached = compile

    let create_args () = {list = []}

    let set_arg_buffer args _idx buf =
      args.list <- ArgBuffer (Obj.repr buf) :: args.list

    let set_arg_int32 args _idx v = args.list <- ArgInt32 v :: args.list

    let set_arg_int64 args _idx v = args.list <- ArgInt64 v :: args.list

    let set_arg_float32 args _idx v = args.list <- ArgFloat32 v :: args.list

    let set_arg_float64 args _idx v = args.list <- ArgFloat64 v :: args.list

    let set_arg_ptr _args _idx _ptr =
      failwith "Interpreter backend does not support raw pointer arguments"

    let launch kernel ~args ~(grid : Framework_sig.dims)
        ~(block : Framework_sig.dims) ~shared_mem:_ ~stream:_ =
      match Hashtbl.find_opt interpreter_kernels kernel.name with
      | Some ir ->
          (* Convert args to interpreter format *)
          let arg_list = List.rev args.list in
          let param_args =
            List.mapi
              (fun i arg ->
                let name = Printf.sprintf "param%d" i in
                match arg with
                | ArgBuffer o ->
                    let buf : _ Memory.buffer = Obj.obj o in
                    (* Convert bigarray to value array - detect type dynamically *)
                    let arr =
                      Array.init buf.size (fun j ->
                          (* Use elem_size to detect type: 4=float32/int32, 8=float64/int64 *)
                          if buf.elem_size = 4 then begin
                            let ba :
                                ( float,
                                  Bigarray.float32_elt,
                                  Bigarray.c_layout )
                                Bigarray.Array1.t =
                              Obj.obj buf.Memory.data
                            in
                            Sarek.Sarek_ir_interp.VFloat32
                              (Bigarray.Array1.get ba j)
                          end
                          else begin
                            let ba :
                                ( float,
                                  Bigarray.float64_elt,
                                  Bigarray.c_layout )
                                Bigarray.Array1.t =
                              Obj.obj buf.Memory.data
                            in
                            Sarek.Sarek_ir_interp.VFloat64
                              (Bigarray.Array1.get ba j)
                          end)
                    in
                    (name, Sarek.Sarek_ir_interp.ArgArray arr)
                | ArgInt32 n ->
                    ( name,
                      Sarek.Sarek_ir_interp.ArgScalar
                        (Sarek.Sarek_ir_interp.VInt32 n) )
                | ArgInt64 n ->
                    ( name,
                      Sarek.Sarek_ir_interp.ArgScalar
                        (Sarek.Sarek_ir_interp.VInt64 n) )
                | ArgFloat32 f ->
                    ( name,
                      Sarek.Sarek_ir_interp.ArgScalar
                        (Sarek.Sarek_ir_interp.VFloat32 f) )
                | ArgFloat64 f ->
                    ( name,
                      Sarek.Sarek_ir_interp.ArgScalar
                        (Sarek.Sarek_ir_interp.VFloat64 f) ))
              arg_list
          in
          Sarek.Sarek_ir_interp.run_kernel
            ir
            ~block:(block.x, block.y, block.z)
            ~grid:(grid.x, grid.y, grid.z)
            param_args
      | None ->
          failwith
            (Printf.sprintf
               "Interpreter.Kernel.launch: kernel '%s' not registered"
               kernel.name)

    let clear_cache () = Hashtbl.clear interpreter_kernels
  end

  let profiling_enabled = ref false

  let enable_profiling () = profiling_enabled := true

  let disable_profiling () = profiling_enabled := false

  let is_available () = true
end

(* Auto-register when module is loaded *)
let registered =
  lazy
    (if Interpreter.is_available () then
       Framework_registry.register_backend
         ~priority:5 (* Lower than Native and GPU backends *)
         (module Interpreter : Framework_sig.BACKEND))

let () = Lazy.force registered

let init () = Lazy.force registered

(** Register an IR kernel for interpretation *)
let register_kernel name ir = Hashtbl.replace interpreter_kernels name ir

(** Check if a kernel is registered *)
let kernel_registered name = Hashtbl.mem interpreter_kernels name

(** List all registered kernels *)
let list_kernels () =
  Hashtbl.fold (fun name _ acc -> name :: acc) interpreter_kernels []
