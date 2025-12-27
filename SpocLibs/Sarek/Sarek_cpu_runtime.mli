(** Sarek_cpu_runtime - CPU runtime for generated native kernels

    This module provides the runtime support for kernels compiled to native
    OCaml code by the Sarek PPX. Unlike Sarek_interp which interprets the AST,
    this module is used by generated code that runs at full native speed. *)

(** {1 Thread State} *)

(** Thread state passed to each kernel invocation. Contains thread/block/grid
    indices and dimensions, plus a barrier function. *)
type thread_state = {
  thread_idx_x : int;
  thread_idx_y : int;
  thread_idx_z : int;
  block_idx_x : int;
  block_idx_y : int;
  block_idx_z : int;
  block_dim_x : int;
  block_dim_y : int;
  block_dim_z : int;
  grid_dim_x : int;
  grid_dim_y : int;
  grid_dim_z : int;
  barrier : unit -> unit;
}

(** {1 Global Index Helpers} *)

val global_idx_x : thread_state -> int

val global_idx_y : thread_state -> int

val global_idx_z : thread_state -> int

val global_size_x : thread_state -> int

val global_size_y : thread_state -> int

val global_size_z : thread_state -> int

(** {1 Bigarray Type Aliases} *)

type float32_vec =
  (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t

type float64_vec =
  (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t

type int32_vec =
  (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t

type int64_vec =
  (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t

type char_vec =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(** {1 Shared Memory} *)

(** Shared memory container for a block. *)
type shared_mem

(** Create a new empty shared memory container. *)
val create_shared : unit -> shared_mem

(** Allocate or retrieve a shared float32 array by name. *)
val alloc_shared_float32 : shared_mem -> string -> int -> float32_vec

(** Allocate or retrieve a shared int32 array by name. *)
val alloc_shared_int32 : shared_mem -> string -> int -> int32_vec

(** {1 Execution Modes} *)

(** Run kernel sequentially. All threads execute in order, barriers are no-ops.
    Good for debugging and correctness testing. *)
val run_sequential :
  block:int * int * int ->
  grid:int * int * int ->
  (thread_state -> 'a -> unit) ->
  'a ->
  unit

(** Run kernel in parallel. Uses Domain per block, fiber per thread. Barriers
    synchronize all fibers within a block. Currently falls back to sequential -
    parallel implementation pending. *)
val run_parallel :
  block:int * int * int ->
  grid:int * int * int ->
  (thread_state -> 'a -> unit) ->
  'a ->
  unit
