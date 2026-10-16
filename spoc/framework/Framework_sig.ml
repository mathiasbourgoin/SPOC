(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

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
  device_features : Sarek_ir_analysis.feature list;
      (** The wide element types this DEVICE provides, as the same vocabulary
          {!Sarek_ir_analysis.kernel_uses} asks a kernel for. The pairing is the
          point: a kernel says what it REQUIRES, this says what the device
          PROVIDES, and a launch gate is one [List.mem] rather than a per-width
          boolean added to this record every time a width is.

          Replaces the former [supports_fp64 : bool] (#142). It was a single
          boolean for a question that already had three answers — fp64, f16 and
          int64 — and the missing one had teeth: an int64 kernel reached
          [vkCreateShaderModule] declaring [OpCapability Int64] against a
          logical device that had never enabled [shaderInt64], because nothing
          in this record could express the int64 half of the question. Read it
          through {!Sarek_capability.permits}, never by testing membership
          directly: an unprobed device must not fall into the permitted bucket.

          [Spoc_core.Device.allows_fp64] and [allows_int64] are the derived
          accessors, so this stays the single source and the two cannot drift.
      *)
  coopmat : Sarek_coopmat.device_support option;
      (** The cooperative-matrix (tensor-core) configurations this DEVICE
          advertises, or [None] when the backend does not probe for them
          (backlog-62 slice 2).

          An OPTION rather than a plain list, for the same reason
          {!Sarek_capability.device_verdict} takes one: an empty list and an
          unprobed device are different facts, and collapsing them makes
          "advertises nothing" indistinguishable from "nobody asked". Only the
          first is evidence. {!Sarek_coopmat.verdict} maps [None] to
          {!Sarek_capability.Unknown}, which does not permit.

          Read it through {!Sarek_coopmat.verdict} or
          {!Sarek_coopmat.find_config}, never by matching on the list directly.
      *)
  supports_atomics : bool;
  warp_size : int;
      (** Invocations that execute in lockstep — CUDA warp, AMD wavefront,
          Vulkan/Metal subgroup. Load-bearing for cooperative matrices, whose
          fragments are distributed across exactly this many invocations
          ({!Sarek_coopmat.components_per_invocation}), so a wrong value here is
          a wrong ABI rather than a wrong statistic. The Vulkan backend now
          reports [VkPhysicalDeviceSubgroupProperties.subgroupSize] instead of a
          hard-coded 32; on the RX 7900 XTX under radv / Mesa 26.1.4-arch3.1
          that is 64. *)
  max_registers_per_block : int;
  clock_rate_khz : int;
  multiprocessor_count : int;
  is_cpu : bool;  (** True for CPU devices (OpenCL CPU runtimes, Native) *)
}

(** Why a backend refused [Memory.alloc_host_ptr]. *)
type host_ptr_refusal =
  | Host_ptr_unsupported
      (** The backend never creates device buffers over host memory. *)
  | Host_ptr_not_cpu_device
      (** The backend can, but only on CPU devices: on a GPU the driver would
          copy or migrate the pages behind the caller's back. *)
  | Host_ptr_empty  (** Zero-length host array. *)
  | Host_ptr_misaligned of {address : nativeint; required_alignment : int}
      (** The array's address is not a multiple of the device's base-address
          alignment (CL_DEVICE_MEM_BASE_ADDR_ALIGN, in bytes): the runtime would
          keep a private copy instead of using the host memory. *)
  | Host_ptr_too_large of {byte_size : int; max_byte_size : int}
      (** Larger than one device allocation may be
          (CL_DEVICE_MAX_MEM_ALLOC_SIZE). *)

(** Device representation - SDK layer type shared across all backends *)
type device = {
  id : int;  (** Global device ID (0, 1, 2...) *)
  backend_id : int;  (** ID within the backend (0, 1...) *)
  name : string;  (** Human-readable device name *)
  framework : string;  (** Backend name: "CUDA", "OpenCL", "Vulkan", "Native" *)
  capabilities : capabilities;
}

(** {1 Plugin Module Signature} *)

(** Minimal framework signature for plugin registration. Used by
    Framework_registry for basic plugin management. *)
module type S = sig
  val name : string

  val version : int * int * int

  val is_available : unit -> bool
end

(** Execution model for backends.
    - JIT: Generate source code at runtime, compile with GPU compiler (CUDA,
      OpenCL)
    - Direct: Execute pre-compiled OCaml functions directly (Native CPU)
    - Custom: Full control over compilation pipeline (LLVM, SPIR-V, future) *)
type execution_model = JIT | Direct | Custom

(** Source language for external kernels *)
type source_lang =
  | CUDA_Source  (** CUDA C/C++ source (.cu) *)
  | OpenCL_Source  (** OpenCL C source (.cl) *)
  | PTX  (** NVIDIA PTX assembly *)
  | SPIR_V  (** SPIR-V binary *)
  | GLSL_Source  (** Vulkan GLSL compute shader *)

(** Extensible type for backend-specific kernel arguments. Each backend extends
    this type with its own variant. This allows type-safe passing of kernel args
    across the framework boundary with typed witnesses. *)
type kargs = ..

(** Placeholder kargs for testing - not associated with any backend *)
type kargs += No_kargs

(** Argument type for run_source. Buffer binder receives typed kargs. *)
type run_source_arg =
  | RSA_Buffer of {
      binder : kargs -> int -> unit;  (** Binds buffer to kernel arg *)
      length : int;  (** Vector length for generated kernels *)
    }
  | RSA_Vector_Length of int32
      (** Auto-injected vector length (see
          [Execute.expand_to_run_source_args ~inject_lengths]), distinguished
          from a genuine caller-supplied [RSA_Int32] so backends that derive
          lengths from the bound buffer itself (e.g. Vulkan) can drop it without
          relying on its position in the argument list - "immediately after a
          buffer" is not a safe test, since [~inject_lengths:false] callers can
          put a real [RSA_Int32] there instead. Backends that need the length as
          an ordinary scalar kernel argument (CUDA/Metal/OpenCL) treat this
          identically to [RSA_Int32]. *)
  | RSA_Int32 of int32
  | RSA_Int64 of int64
  | RSA_Float32 of float
  | RSA_Float64 of float

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

(** Re-export typed value types for convenience *)
type exec_arg = Typed_value.exec_arg =
  | EA_Int32 of int32
  | EA_Int64 of int64
  | EA_Float32 of float
  | EA_Float64 of float
  | EA_Scalar :
      (module Typed_value.SCALAR_TYPE with type t = 'a) * 'a
      -> exec_arg
  | EA_Composite :
      (module Typed_value.COMPOSITE_TYPE with type t = 'a) * 'a
      -> exec_arg
  | EA_Vec of (module Typed_value.EXEC_VECTOR)

(** {1 Ranged / asynchronous transfers, pinned memory, stream ordering} *)

(** Backend-tagged stream and event handles, extended by each backend exactly
    like {!kargs}. A backend's [Stream.t] is a fresh abstract type at every
    first-class-module unpacking, so a stream created by
    [Spoc_core.Transfer.create_stream] cannot be handed to a buffer closure
    built by another unpacking of the same backend. Wrapping it in one of these
    and unwrapping it with the backend's own [unwrap_stream] is the only route
    that type-checks, and it fails closed ([None]) when the stream belongs to a
    different backend. *)
type stream_ref = ..

type event_ref = ..

(** What a backend's {!ASYNC} module actually implements. Read by tests and by
    callers that want to choose a path up front; the operations themselves still
    refuse on their own, so a stale or wrong record cannot turn into a silent
    fallback. *)
type async_support = {
  ranged_copy : bool;  (** Synchronous host<->device copies at byte offsets. *)
  async_copy : bool;  (** The same copies enqueued on a stream. *)
  pinned_host : bool;
      (** Page-locked host allocation ([alloc_pinned]/[free_pinned]). On a CPU
          backend this is ordinary host memory: pinning has no meaning when
          device memory is host memory. The record is per backend: a backend may
          still refuse the call on a device that lacks what it needs, with
          [Feature_not_supported] from the call itself. A consumer that must
          know up front tries the call. *)
  host_register : bool;  (** Page-locking existing host memory. *)
  stream_launch : bool;  (** Kernel launches enqueued on a given stream. *)
  stream_events : bool;
      (** [stream_wait_event] and [event_query] with real ordering semantics. *)
}

let no_async_support =
  {
    ranged_copy = false;
    async_copy = false;
    pinned_host = false;
    host_register = false;
    stream_launch = false;
    stream_events = false;
  }

let full_async_support =
  {
    ranged_copy = true;
    async_copy = true;
    pinned_host = true;
    host_register = true;
    stream_launch = true;
    stream_events = true;
  }

(** Ranged and stream-ordered host<->device transfers, pinned host memory and
    inter-stream ordering, as one backend provides them. Byte offsets and sizes
    throughout; bounds are checked by the caller ([Spoc_core.Transfer]).

    [~stream:None] means synchronous: the call returns once the copy is
    complete. [~stream:(Some s)] enqueues the copy on [s] and may return before
    it has run. The raw addresses passed here are NOT GC roots: the caller keeps
    the memory alive until the stream (or an event recorded after the copy) has
    completed — [Spoc_core.Transfer] does this for every Bigarray it passes.

    A backend that does not provide an operation raises
    [Sarek_backend_error.Backend_error.Backend_error] with a
    [Feature_not_supported] payload from it — see {!Async_refused}. It never
    degrades to a different operation (for instance a synchronous whole-buffer
    copy). *)
module type ASYNC = sig
  type device

  type stream

  type event

  type 'a buffer

  val support : async_support

  val host_to_device :
    src_ptr:nativeint ->
    dst:'a buffer ->
    dst_offset:int ->
    byte_size:int ->
    stream:stream option ->
    unit

  val device_to_host :
    src:'a buffer ->
    src_offset:int ->
    dst_ptr:nativeint ->
    byte_size:int ->
    stream:stream option ->
    unit

  (** All work enqueued on the stream after this call waits until the work
      recorded by the event has completed. *)
  val stream_wait_event : stream -> event -> unit

  (** [true] once the work recorded by the event has completed (or the event was
      never recorded); never blocks. *)
  val event_query : event -> bool

  (** Page-locked host memory of [byte_size] bytes, returned as a raw address.
      Released by {!free_pinned}. A backend whose pinned memory belongs to the
      device (Vulkan: a driver mapping) unmaps it when the device is destroyed;
      the address is then invalid and only {!free_pinned} may still be called on
      it. *)
  val alloc_pinned : device -> byte_size:int -> nativeint

  val free_pinned : device -> nativeint -> unit

  (** Page-lock existing host memory for DMA. The caller keeps it alive until
      {!unregister_host}. *)
  val register_host : device -> nativeint -> byte_size:int -> unit

  val unregister_host : device -> nativeint -> unit
end

(** The refusing implementation of {!ASYNC}, for backends that do not provide
    it. Every operation raises [Feature_not_supported] naming the backend and
    the operation. *)
module Async_refused (X : sig
  val backend : string

  type device

  type stream

  type event

  type 'a buffer
end) :
  ASYNC
    with type device = X.device
     and type stream = X.stream
     and type event = X.event
     and type 'a buffer = 'a X.buffer = struct
  type device = X.device

  type stream = X.stream

  type event = X.event

  type 'a buffer = 'a X.buffer

  let support = no_async_support

  let refuse op =
    Sarek_backend_error.Backend_error.raise_error
      (Sarek_backend_error.Backend_error.feature_not_supported
         ~backend:X.backend
         (op
        ^ " (ranged/async transfers, pinned host memory and stream ordering \
           are implemented on HIP, CUDA, Metal, Vulkan, Native and Interpreter \
           only)"))

  let host_to_device ~src_ptr:_ ~dst:_ ~dst_offset:_ ~byte_size:_ ~stream:_ =
    refuse "ranged host-to-device copy"

  let device_to_host ~src:_ ~src_offset:_ ~dst_ptr:_ ~byte_size:_ ~stream:_ =
    refuse "ranged device-to-host copy"

  let stream_wait_event _ _ = refuse "stream wait on event"

  let event_query _ = refuse "event query"

  let alloc_pinned _ ~byte_size:_ = refuse "pinned host allocation"

  let free_pinned _ _ = refuse "pinned host free"

  let register_host _ _ ~byte_size:_ = refuse "host memory registration"

  let unregister_host _ _ = refuse "host memory unregistration"
end

(** The refusing [run_source_on_stream] for backends whose
    [async_support.stream_launch] is [false]. Never falls back to the default
    stream: a launch the caller ordered on a stream and that ran elsewhere would
    break the ordering the caller relies on. *)
let refuse_stream_launch ~backend : 'a =
  Sarek_backend_error.Backend_error.raise_error
    (Sarek_backend_error.Backend_error.feature_not_supported
       ~backend
       "kernel launch on a stream (implemented on HIP, CUDA, Metal and Vulkan \
        only)")

(** Fresh [stream_ref]/[event_ref] constructors for one backend. Each
    application mints its own extension constructors at run time, so a stream of
    one backend never unwraps as another's. *)
module Make_refs (T : sig
  type stream

  type event
end) : sig
  val wrap_stream : T.stream -> stream_ref

  val unwrap_stream : stream_ref -> T.stream option

  val wrap_event : T.event -> event_ref

  val unwrap_event : event_ref -> T.event option
end = struct
  type stream_ref += Stream_ref of T.stream

  type event_ref += Event_ref of T.event

  let wrap_stream s = Stream_ref s

  let unwrap_stream = function Stream_ref s -> Some s | _ -> None

  let wrap_event e = Event_ref e

  let unwrap_event = function Event_ref e -> Some e | _ -> None
end

(** Extended backend signature for Phase 4 unified execution. Adds execution
    model discrimination and IR-based code generation. *)
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

    (** A stream that does NOT synchronise implicitly with the device's default
        stream (CUDA/HIP: the non-blocking flag). Backends whose queues are
        already independent return the same as {!create}; backends with no
        asynchrony (Native, Interpreter) return a unit stream. Backends that
        cannot provide the property raise [Feature_not_supported]. *)
    val create_non_blocking : Device.t -> t

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

    (** A buffer that IS the host array, for backends whose device memory is
        host memory (Native, Interpreter: [is_zero_copy] is [true], no transfer
        and no synchronisation). Used for every vector by default, so it must
        preserve copy semantics: backends with device memory of their own,
        OpenCL CPU devices included, return [None]. *)
    val alloc_zero_copy :
      Device.t ->
      ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
      ('a, 'b) Bigarray.kind ->
      'a buffer option

    (** Explicit opt-in: a device buffer created over the host array's memory
        (OpenCL [CL_MEM_USE_HOST_PTR]). Never used implicitly. The caller keeps
        the array alive and in place for the buffer's lifetime. Host access goes
        through [host_to_device]/[device_to_host] (or the [_ptr] variants) on
        the same array, which synchronise (map/unmap) instead of copying. Native
        and Interpreter return their [alloc_zero_copy] buffer. *)
    val alloc_host_ptr :
      Device.t ->
      ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
      ('a, 'b) Bigarray.kind ->
      ('a buffer, host_ptr_refusal) result

    val free : 'a buffer -> unit

    (** {2 Synchronous Transfers} *)

    val host_to_device :
      src:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> dst:'a buffer -> unit

    val device_to_host :
      src:'a buffer -> dst:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> unit

    (** Transfer from raw pointer to device (for custom types). [src_ptr] is a
        raw machine address (as produced by [raw_address_of_ptr]); a [nativeint]
        is NOT a GC root, so the CALLER must keep the pointed-to allocation live
        across this call. Transfers are synchronous today (a live source on the
        caller's stack suffices); an async backend (e.g. the worker+SAB readback
        planned for the jsoo path) must root the source explicitly. *)
    val host_ptr_to_device :
      src_ptr:nativeint -> byte_size:int -> dst:'a buffer -> unit

    (** Transfer from device to raw pointer (for custom types). Same keep-alive
        contract as {!host_ptr_to_device}: the caller owns [dst_ptr]'s liveness.
    *)
    val device_to_host_ptr :
      src:'a buffer -> dst_ptr:nativeint -> byte_size:int -> unit

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

    val load_from_ptx : name:string -> ptx:string -> t

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

  (** Ranged / stream-ordered transfers, pinned host memory and inter-stream
      ordering (see {!ASYNC}). Backends that do not provide them use
      {!Async_refused}. *)
  module Async :
    ASYNC
      with type device := Device.t
       and type stream := Stream.t
       and type event := Event.t
       and type 'a buffer := 'a Memory.buffer

  val enable_profiling : unit -> unit

  val disable_profiling : unit -> unit

  (** The execution model this backend uses *)
  val execution_model : execution_model

  (** Generate source code from Sarek IR (for JIT backends). Returns None for
      Direct/Custom backends.
      @param block
        Optional block dimensions (required for Vulkan/GLSL which embeds
        workgroup size in shader)
      @param soa_params
        Names of custom (record) vector parameters to lower as
        Structure-of-Arrays: each expands to N per-leaf base pointers + one
        shared length instead of a single packed AoS pointer (the #260 PTX
        emitter ABI). Honoured only by the CUDA/PTX backend. A backend that
        cannot honour it must REFUSE a non-empty list rather than ignore it —
        see {!Sarek_backend_error.Backend_error.reject_soa_params}, which the
        five source-generating backends without an SoA lowering (CUDA/C, HIP,
        OpenCL, Vulkan, Metal) call — because emitting AoS source for an SoA
        argument list is an ABI mismatch that no compiler catches, and what
        happens next is per backend: rejected at bind or launch on OpenCL and
        Vulkan; on CUDA/C, HIP and Metal nothing compares the expanded list
        against the compiled kernel, so the positional shift goes unchecked
        (backlog-214; {!Sarek_backend_error.Backend_error.reject_soa_params}
        carries the per-backend detail). The backends that return [None]
        unconditionally (Native, Interpreter, WebGPU) generate no source at all
        and so do not carry that refusal. Defaults to [[]] (all AoS), so
        existing callers are byte-identical. *)
  val generate_source :
    ?block:dims ->
    ?soa_params:string list ->
    Sarek_ir_types.kernel ->
    string option

  (** Execute a kernel directly (for Direct/Custom backends). JIT backends
      should raise an error if this is called. The backend chooses which
      component to use:
      - Direct backends (Native): use native_fn
      - Custom backends (Interpreter): use ir

      @param native_fn Pre-compiled OCaml function (from PPX)
      @param ir Sarek IR kernel (for interpretation)
      @param block Block dimensions
      @param grid Grid dimensions
      @param args Kernel arguments - fully typed *)
  val execute_direct :
    native_fn:(block:dims -> grid:dims -> exec_arg array -> unit) option ->
    ir:Sarek_ir_types.kernel option ->
    block:dims ->
    grid:dims ->
    exec_arg array ->
    unit

  (** Backend-specific intrinsic registry *)
  module Intrinsics : INTRINSIC_REGISTRY

  (** {2 External Kernel Execution} *)

  (** List of source languages this backend supports *)
  val supported_source_langs : source_lang list

  (** Execute an external kernel from source code.
      @param source The kernel source code (CUDA/OpenCL/PTX string)
      @param lang The source language
      @param kernel_name The name of the kernel function to execute
      @param block Block dimensions
      @param grid Grid dimensions
      @param shared_mem Shared memory size in bytes
      @param args Kernel arguments as run_source_arg list
      @raise Failure if source language is not supported *)
  val run_source :
    source:string ->
    lang:source_lang ->
    kernel_name:string ->
    block:dims ->
    grid:dims ->
    shared_mem:int ->
    run_source_arg list ->
    unit

  (** {2 Kernel Args Wrapping}
      Type-safe wrapping/unwrapping of backend-specific kernel args into the
      extensible kargs type. Each backend extends kargs with its own variant. *)

  (** Wrap this backend's kernel args into a kargs variant *)
  val wrap_kargs : Kernel.args -> kargs

  (** Unwrap kargs to this backend's kernel args. Returns None if wrong backend.
  *)
  val unwrap_kargs : kargs -> Kernel.args option

  (** {2 Streams and events across module unpackings} *)

  val wrap_stream : Stream.t -> stream_ref

  val unwrap_stream : stream_ref -> Stream.t option

  val wrap_event : Event.t -> event_ref

  val unwrap_event : event_ref -> Event.t option

  (** {!run_source} enqueued on [stream] instead of the default stream. The call
      may return before the kernel has run. Backends whose
      [async_support.stream_launch] is [false] refuse with
      [Feature_not_supported]; Direct/Custom backends (Native, Interpreter) are
      never called through this entry point — their launches are synchronous. *)
  val run_source_on_stream :
    stream:Stream.t ->
    source:string ->
    lang:source_lang ->
    kernel_name:string ->
    block:dims ->
    grid:dims ->
    shared_mem:int ->
    run_source_arg list ->
    unit
end

(** Low-level backend implementation interface shared by the CUDA, OpenCL, and
    Metal plugin-base modules. This is the device/memory/stream/event/kernel
    core that each backend's [*_plugin_base.ml] implements directly on top of
    its FFI bindings; the full {!BACKEND} interface (source generation, direct
    execution, intrinsics, external-kernel execution, kargs wrapping) is
    assembled on top of it in each backend's [*_plugin.ml].

    [args] and [buffer] are abstract: callers construct args via
    {!Kernel.create_args} and the [set_arg_*] functions, never by relying on the
    underlying representation. *)
module type PLUGIN_BASE = sig
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

    val capabilities : t -> capabilities

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

    val alloc_host_ptr :
      Device.t ->
      ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
      ('a, 'b) Bigarray.kind ->
      ('a buffer, host_ptr_refusal) result

    val is_zero_copy : 'a buffer -> bool

    val free : 'a buffer -> unit

    val host_to_device :
      src:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> dst:'a buffer -> unit

    val device_to_host :
      src:'a buffer -> dst:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> unit

    val host_ptr_to_device :
      src_ptr:nativeint -> byte_size:int -> dst:'a buffer -> unit

    val device_to_host_ptr :
      src:'a buffer -> dst_ptr:nativeint -> byte_size:int -> unit

    val device_to_device : src:'a buffer -> dst:'a buffer -> unit

    val size : 'a buffer -> int

    val device_ptr : 'a buffer -> nativeint
  end

  module Stream : sig
    type t

    val create : Device.t -> t

    (** A stream that does NOT synchronise implicitly with the device's default
        stream (CUDA/HIP: the non-blocking flag). Backends whose queues are
        already independent return the same as {!create}; backends with no
        asynchrony (Native, Interpreter) return a unit stream. Backends that
        cannot provide the property raise [Feature_not_supported]. *)
    val create_non_blocking : Device.t -> t

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

    val clear_cache : unit -> unit

    val load_from_ptx : name:string -> ptx:string -> t

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

  (** Ranged / stream-ordered transfers, pinned host memory and inter-stream
      ordering (see {!ASYNC}). Backends that do not provide them use
      {!Async_refused}. *)
  module Async :
    ASYNC
      with type device := Device.t
       and type stream := Stream.t
       and type event := Event.t
       and type 'a buffer := 'a Memory.buffer

  val enable_profiling : unit -> unit

  val disable_profiling : unit -> unit

  val is_available : unit -> bool
end
