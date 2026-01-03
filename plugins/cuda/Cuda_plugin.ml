(******************************************************************************
 * CUDA Plugin - Framework Implementation
 *
 * Implements the Framework_sig.S interface for CUDA devices.
 * This plugin is auto-registered when loaded.
 ******************************************************************************)

open Spoc_framework

module Cuda : sig
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

    val get_current_device : unit -> t option

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

    type args = Cuda_api.Kernel.arg list ref

    val compile : Device.t -> name:string -> source:string -> t

    val compile_cached : Device.t -> name:string -> source:string -> t

    val clear_cache : unit -> unit

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
  end

  val enable_profiling : unit -> unit

  val disable_profiling : unit -> unit

  val is_available : unit -> bool
end = struct
  let name = "CUDA"

  let version = (12, 0, 0)

  (** Current device for kernel compilation/execution *)
  let current_device : Cuda_api.Device.t option ref = ref None

  module Device = struct
    type t = Cuda_api.Device.t

    type id = int

    let init = Cuda_api.Device.init

    let count = Cuda_api.Device.count

    let get = Cuda_api.Device.get

    let id (d : t) = d.Cuda_api.Device.id

    let name (d : t) = d.Cuda_api.Device.name

    let set_current (d : t) =
      Cuda_api.Device.set_current d ;
      current_device := Some d

    let get_current_device () = !current_device

    let synchronize = Cuda_api.Device.synchronize

    let capabilities (d : t) : Framework_sig.capabilities =
      let open Cuda_api.Device in
      {
        Framework_sig.max_threads_per_block = d.max_threads_per_block;
        max_block_dims = d.max_block_dims;
        max_grid_dims = d.max_grid_dims;
        shared_mem_per_block = d.shared_mem_per_block;
        total_global_mem = d.total_mem;
        compute_capability = d.compute_capability;
        supports_fp64 = true;
        supports_atomics = true;
        warp_size = d.warp_size;
        max_registers_per_block = 65536;
        (* Typical for modern GPUs *)
        clock_rate_khz = 0;
        (* Would need another query *)
        multiprocessor_count = d.multiprocessor_count;
        is_cpu = false;
      }
  end

  module Memory = struct
    type 'a buffer = 'a Cuda_api.Memory.buffer

    let alloc = Cuda_api.Memory.alloc

    let alloc_custom = Cuda_api.Memory.alloc_custom

    (** CUDA doesn't support zero-copy with host memory *)
    let alloc_zero_copy _device _ba _kind = None

    let is_zero_copy _buf = false

    let free = Cuda_api.Memory.free

    let host_to_device = Cuda_api.Memory.host_to_device

    let device_to_host = Cuda_api.Memory.device_to_host

    let host_ptr_to_device = Cuda_api.Memory.host_ptr_to_device

    let device_to_host_ptr = Cuda_api.Memory.device_to_host_ptr

    let device_to_device = Cuda_api.Memory.device_to_device

    let size (buf : 'a buffer) = buf.Cuda_api.Memory.size

    let device_ptr (buf : 'a buffer) =
      Unsigned.UInt64.to_int64 buf.Cuda_api.Memory.ptr |> Int64.to_nativeint
  end

  module Stream = struct
    type t = Cuda_api.Stream.t

    let create = Cuda_api.Stream.create

    let destroy = Cuda_api.Stream.destroy

    let synchronize = Cuda_api.Stream.synchronize

    let default = Cuda_api.Stream.default
  end

  module Event = struct
    type t = Cuda_api.Event.t

    let create = Cuda_api.Event.create

    let destroy = Cuda_api.Event.destroy

    let record = Cuda_api.Event.record

    let synchronize = Cuda_api.Event.synchronize

    let elapsed = Cuda_api.Event.elapsed
  end

  module Kernel = struct
    type t = Cuda_api.Kernel.t

    type args = Cuda_api.Kernel.arg list ref

    let compile dev ~name ~source = Cuda_api.Kernel.compile dev ~name ~source

    let compile_cached dev ~name ~source =
      Cuda_api.Kernel.compile_cached dev ~name ~source

    let clear_cache = Cuda_api.Kernel.clear_cache

    let create_args () = ref []

    let set_arg_buffer args _idx buf =
      args := !args @ [Cuda_api.Kernel.ArgBuffer buf]

    let set_arg_int32 args _idx v = args := !args @ [Cuda_api.Kernel.ArgInt32 v]

    let set_arg_int64 args _idx v = args := !args @ [Cuda_api.Kernel.ArgInt64 v]

    let set_arg_float32 args _idx v =
      args := !args @ [Cuda_api.Kernel.ArgFloat32 v]

    let set_arg_float64 args _idx v =
      args := !args @ [Cuda_api.Kernel.ArgFloat64 v]

    let set_arg_ptr args _idx ptr = args := !args @ [Cuda_api.Kernel.ArgPtr ptr]

    let launch kernel ~args ~grid ~block ~shared_mem ~stream =
      let open Framework_sig in
      Cuda_api.Kernel.launch
        kernel
        ~args:!args
        ~grid:(grid.x, grid.y, grid.z)
        ~block:(block.x, block.y, block.z)
        ~shared_mem
        ~stream
  end

  let enable_profiling () =
    let _ = Cuda_bindings.cuProfilerStart () in
    ()

  let disable_profiling () =
    let _ = Cuda_bindings.cuProfilerStop () in
    ()

  let is_available = Cuda_api.is_available
end

(* Legacy init retained for compatibility; backend registration now handled by
   Cuda_plugin_v2. *)
let init () = ()
