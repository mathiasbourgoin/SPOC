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
  is_cpu : bool;  (** True for CPU devices - enables zero-copy optimization *)
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

  (** Async execution streams / command queues *)
  module Stream : sig
    type t

    val create : Device.t -> t

    val destroy : t -> unit

    val synchronize : t -> unit

    val default : Device.t -> t
  end

  (** GPU memory allocation and transfer *)
  module Memory : sig
    type 'a buffer

    val alloc : Device.t -> int -> ('a, 'b) Bigarray.kind -> 'a buffer

    (** Allocate buffer for custom types with explicit element size in bytes *)
    val alloc_custom : Device.t -> size:int -> elem_size:int -> 'a buffer

    (** Allocate zero-copy buffer using host bigarray memory directly. For CPU
        OpenCL devices, this avoids memory copies entirely. Returns None if
        zero-copy not supported by this backend. *)
    val alloc_zero_copy :
      Device.t ->
      ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
      ('a, 'b) Bigarray.kind ->
      'a buffer option

    val free : 'a buffer -> unit

    (** {2 Synchronous Transfers} *)

    val host_to_device :
      src:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> dst:'a buffer -> unit

    val device_to_host :
      src:'a buffer -> dst:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> unit

    (** Transfer from raw pointer to device (for custom types). src_ptr should
        be obtained via Ctypes from a customarray. *)
    val host_ptr_to_device :
      src_ptr:unit Ctypes.ptr -> byte_size:int -> dst:'a buffer -> unit

    (** Transfer from device to raw pointer (for custom types) *)
    val device_to_host_ptr :
      src:'a buffer -> dst_ptr:unit Ctypes.ptr -> byte_size:int -> unit

    val device_to_device : src:'a buffer -> dst:'a buffer -> unit

    (** {2 Buffer Info} *)

    val size : 'a buffer -> int

    val device_ptr : 'a buffer -> nativeint

    (** Check if buffer uses zero-copy (no transfers needed) *)
    val is_zero_copy : 'a buffer -> bool
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

    val set_arg_ptr : args -> int -> nativeint -> unit

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

(** {1 Phase 4: Unified Execution Architecture} *)

(** Execution model for backends.
    - JIT: Generate source code at runtime, compile with GPU compiler (CUDA,
      OpenCL)
    - Direct: Execute pre-compiled OCaml functions directly (Native CPU)
    - Custom: Full control over compilation pipeline (LLVM, SPIR-V, future) *)
type execution_model = JIT | Direct | Custom

(** Convergence behavior of an intrinsic.
    - Uniform: All threads in warp/wavefront compute same value
    - Divergent: Threads may compute different values
    - Sync: Intrinsic contains synchronization (barrier) *)
type convergence = Uniform | Divergent | Sync

(** Intrinsic registry interface for backend-specific intrinsics. Note: The
    actual intrinsic_impl type is defined in each backend's intrinsic registry
    module to avoid circular dependencies with Sarek_ir. *)
module type INTRINSIC_REGISTRY = sig
  (** Backend-specific intrinsic implementation type *)
  type intrinsic_impl

  (** Register an intrinsic by name *)
  val register : string -> intrinsic_impl -> unit

  (** Look up an intrinsic by name *)
  val find : string -> intrinsic_impl option

  (** List all registered intrinsic names *)
  val list_all : unit -> string list
end

(** Extended backend signature for Phase 4 unified execution. Adds execution
    model discrimination and IR-based code generation.

    Note: generate_source takes an Obj.t to avoid circular dependency with
    Sarek_ir. Backends should cast to Sarek_ir.kernel internally. *)
module type BACKEND_V2 = sig
  include BACKEND

  (** The execution model this backend uses *)
  val execution_model : execution_model

  (** Generate source code from Sarek IR (for JIT backends). The ir parameter is
      Sarek_ir.kernel wrapped as Obj.t to avoid circular dependencies. Returns
      None for Direct/Custom backends. *)
  val generate_source : Obj.t -> string option

  (** Execute a kernel directly (for Direct backends). JIT backends should raise
      an error if this is called.
      @param native_fn Pre-compiled OCaml function (from PPX)
      @param block Block dimensions
      @param grid Grid dimensions
      @param args Kernel arguments as Obj.t array *)
  val execute_direct :
    native_fn:(block:dims -> grid:dims -> Obj.t array -> unit) option ->
    block:dims ->
    grid:dims ->
    Obj.t array ->
    unit

  (** Backend-specific intrinsic registry *)
  module Intrinsics : INTRINSIC_REGISTRY
end
