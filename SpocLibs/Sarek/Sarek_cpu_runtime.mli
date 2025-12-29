(** Sarek_cpu_runtime - CPU runtime for generated native kernels

    This module provides the runtime support for kernels compiled to native
    OCaml code by the Sarek PPX. Unlike Sarek_interp which interprets the AST,
    this module is used by generated code that runs at full native speed. *)

(** {1 Float32 Module}

    Re-exported for use in generated native kernels. Provides true float32
    semantics matching GPU behavior. *)
module Float32 = Sarek_float32

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

(** Shared memory container for a block. *)
type shared_mem

(** Create a new empty shared memory container. *)
val create_shared : unit -> shared_mem

(** Allocate or retrieve a shared array by name. Uses regular OCaml arrays to
    support custom types. *)
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
