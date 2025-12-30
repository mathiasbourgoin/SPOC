(******************************************************************************
 * Sarek Framework - Plugin Interface
 *
 * Defines the interface that GPU backend plugins must implement.
 * Plugins (CUDA, OpenCL) register themselves and provide device, memory,
 * stream, event, and kernel management functionality.
 ******************************************************************************)

(** {1 Common Types} *)

(** 3D dimensions for grid and block *)
type dims = {x : int; y : int; z : int}

let dims_1d x = {x; y = 1; z = 1}

let dims_2d x y = {x; y; z = 1}

let dims_3d x y z = {x; y; z}

(** Device capabilities - queried from hardware *)
type capabilities = {
  max_threads_per_block : int;
  max_block_dims : int * int * int;
  max_grid_dims : int * int * int;
  shared_mem_per_block : int;
  total_global_mem : int64;
  compute_capability : int * int;
      (** (major, minor) for CUDA, (0,0) for OpenCL *)
  supports_fp64 : bool;
  supports_atomics : bool;
  warp_size : int;
  max_registers_per_block : int;
  clock_rate_khz : int;
  multiprocessor_count : int;
}

(** {1 Plugin Module Signature} *)

(** Minimal framework signature for plugin registration. Used by
    Framework_registry for basic plugin management. *)
module type S = sig
  val name : string

  val version : int * int * int

  val is_available : unit -> bool
end

(** Full plugin signature with Device, Memory, Stream, Event, and Kernel
    modules. This is the complete interface that backends (CUDA, OpenCL)
    implement. *)
module type BACKEND = sig
  val name : string

  val version : int * int * int

  val is_available : unit -> bool

  (** Device management *)
  module Device : sig
    type t

    type id = int

    val init : unit -> unit

    val count : unit -> int

    val get : int -> t

    val id : t -> id

    val name : t -> string

    val capabilities : t -> capabilities

    val set_current : t -> unit

    val synchronize : t -> unit
  end

  (** GPU memory allocation and transfer *)
  module Memory : sig
    type 'a buffer

    val alloc : Device.t -> int -> ('a, 'b) Bigarray.kind -> 'a buffer

    val free : 'a buffer -> unit

    val host_to_device :
      src:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> dst:'a buffer -> unit

    val device_to_host :
      src:'a buffer -> dst:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> unit

    val device_to_device : src:'a buffer -> dst:'a buffer -> unit

    val size : 'a buffer -> int

    val device_ptr : 'a buffer -> nativeint
  end

  (** Async execution streams / command queues *)
  module Stream : sig
    type t

    val create : Device.t -> t

    val destroy : t -> unit

    val synchronize : t -> unit

    val default : Device.t -> t
  end

  (** Timing events *)
  module Event : sig
    type t

    val create : unit -> t

    val destroy : t -> unit

    val record : t -> Stream.t -> unit

    val synchronize : t -> unit

    val elapsed : start:t -> stop:t -> float
  end

  (** Kernel compilation and execution *)
  module Kernel : sig
    type t

    type args

    val compile : Device.t -> name:string -> source:string -> t

    val compile_cached : Device.t -> name:string -> source:string -> t

    val clear_cache : unit -> unit

    val create_args : unit -> args

    val set_arg_buffer : args -> int -> _ Memory.buffer -> unit

    val set_arg_int32 : args -> int -> int32 -> unit

    val set_arg_int64 : args -> int -> int64 -> unit

    val set_arg_float32 : args -> int -> float -> unit

    val set_arg_float64 : args -> int -> float -> unit

    val launch :
      t ->
      args:args ->
      grid:dims ->
      block:dims ->
      shared_mem:int ->
      stream:Stream.t option ->
      unit
  end

  val enable_profiling : unit -> unit

  val disable_profiling : unit -> unit
end
