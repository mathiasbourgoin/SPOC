(** Sarek_cpu_runtime - CPU runtime for generated native kernels

    This module provides the runtime support for kernels compiled to native
    OCaml code by the Sarek PPX. Unlike Sarek_interp which interprets the AST,
    this module is used by generated code that runs at full native speed. *)

(** {1 Float32 Module}

    Re-exported for use in generated native kernels. Provides true float32
    semantics matching GPU behavior. *)
module Float32 = Sarek_float32

(** {1 Execution Mode} *)

(** Execution mode for native kernels *)
type exec_mode =
  | Sequential  (** Single-threaded, barriers are no-ops *)
  | Parallel  (** Spawn domains per kernel launch *)
  | Threadpool  (** Use persistent thread pool (fission mode) *)

(** {1 Thread State} *)

(** Thread state passed to each kernel invocation. Contains thread/block/grid
    indices and dimensions, plus a barrier function.

    All indices are int32 to match GPU semantics (Sarek_stdlib.Gpu uses int32).
*)
type thread_state = {
  thread_idx_x : int32;
  thread_idx_y : int32;
  thread_idx_z : int32;
  block_idx_x : int32;
  block_idx_y : int32;
  block_idx_z : int32;
  block_dim_x : int32;
  block_dim_y : int32;
  block_dim_z : int32;
  grid_dim_x : int32;
  grid_dim_y : int32;
  grid_dim_z : int32;
  barrier : unit -> unit;
}

(** {1 Global Index Helpers} *)

val global_idx_x : thread_state -> int32

val global_idx_y : thread_state -> int32

val global_idx_z : thread_state -> int32

val global_size_x : thread_state -> int32

val global_size_y : thread_state -> int32

val global_size_z : thread_state -> int32

(** {1 Shared Memory} *)

(** Shared memory container for a block. Uses per-type hashtables for type
    safety on common types. *)
type shared_mem

(** Create a new empty shared memory container. *)
val create_shared : unit -> shared_mem

(** {2 Typed Allocators}

    These allocators are type-safe and don't use Obj.magic. *)

val alloc_shared_int : shared_mem -> string -> int -> int -> int array

val alloc_shared_float : shared_mem -> string -> int -> float -> float array

val alloc_shared_int32 : shared_mem -> string -> int -> int32 -> int32 array

val alloc_shared_int64 : shared_mem -> string -> int -> int64 -> int64 array

(** {2 Generic Allocator}

    For custom types not covered by typed allocators. Uses Obj.t internally. *)
val alloc_shared : shared_mem -> string -> int -> 'a -> 'a array

(** {1 Execution Modes} *)

(** Run kernel sequentially. All threads execute in order, barriers are no-ops.
    Good for debugging and correctness testing. The kernel function receives
    thread_state, shared_mem (for the block), and args. *)
val run_sequential :
  block:int * int * int ->
  grid:int * int * int ->
  (thread_state -> shared_mem -> 'a -> unit) ->
  'a ->
  unit

(** Run kernel in parallel. Threads within each block run in parallel using
    OCaml 5 Domains. Barriers properly synchronize all threads within a block
    using Mutex/Condition. Blocks run sequentially. *)
val run_parallel :
  block:int * int * int ->
  grid:int * int * int ->
  (thread_state -> shared_mem -> 'a -> unit) ->
  'a ->
  unit

(** {1 Fission Mode - Thread Pool Execution}

    For the fission device, kernels are executed by a persistent thread pool.
    This eliminates per-launch Domain spawn/join overhead for workloads with
    many consecutive kernel launches (like odd-even sort with 512 launches).

    Architecture:
    - A persistent pool of N worker domains (one per core) stays alive
    - Each kernel launch distributes work to workers via condition variables
    - Workers execute their portion and signal completion
    - No Domain spawn/join per kernel - just signaling overhead *)

(** Run kernel using the persistent thread pool. Like run_parallel but uses
    pre-created worker domains. Best for workloads with many consecutive kernel
    launches. *)
val run_threadpool :
  block:int * int * int ->
  grid:int * int * int ->
  (thread_state -> shared_mem -> 'a -> unit) ->
  'a ->
  unit

(** {2 Asynchronous Queue API}

    Multiple queues are supported (like CUDA/OpenCL command queues):
    - Same queue_id: kernels execute in order (serialized)
    - Different queue_id: kernels can run in parallel (one dispatcher per queue)
*)

(** Enqueue a kernel for fission execution on a specific queue. The kernel
    starts executing immediately in the background via thread pool. Kernels on
    the same queue execute in order; different queues run in parallel. Default
    queue_id is 0. *)
val enqueue_fission :
  ?queue_id:int ->
  kernel:
    (mode:exec_mode ->
    block:int * int * int ->
    grid:int * int * int ->
    Obj.t array ->
    unit) ->
  args:Obj.t array ->
  block:int * int * int ->
  grid:int * int * int ->
  unit ->
  unit

(** Wait for a specific queue to complete. *)
val flush_fission_queue : int -> unit

(** Wait for all fission queues to complete. Called by Devices.flush. *)
val flush_fission : unit -> unit
