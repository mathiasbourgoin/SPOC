(** Sarek_cpu_runtime - CPU runtime for generated native kernels

    This module provides the runtime support for kernels compiled to native
    OCaml code by the Sarek PPX. Unlike Sarek_interp which interprets the AST,
    this module is used by generated code that runs at full native speed. *)

(** {1 Thread State}

    Thread state is passed to each generated kernel function. The kernel reads
    thread/block/grid indices from this record. *)

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
      (** Barrier function - no-op in sequential, effect in parallel *)
}

(** {1 Global Index Helpers} *)

let global_idx_x st = (st.block_idx_x * st.block_dim_x) + st.thread_idx_x

let global_idx_y st = (st.block_idx_y * st.block_dim_y) + st.thread_idx_y

let global_idx_z st = (st.block_idx_z * st.block_dim_z) + st.thread_idx_z

let global_size_x st = st.grid_dim_x * st.block_dim_x

let global_size_y st = st.grid_dim_y * st.block_dim_y

let global_size_z st = st.grid_dim_z * st.block_dim_z

(** {1 Bigarray Type Aliases}

    Kernel arguments are passed as bigarrays for efficient access. *)

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

(** {1 Shared Memory}

    Shared memory is allocated per-block and accessible by all threads in the
    block. *)

type shared_mem = {data : (string, Obj.t) Hashtbl.t}

let create_shared () = {data = Hashtbl.create 8}

(** Allocate a shared float32 array. If already allocated, returns existing. *)
let alloc_shared_float32 (shared : shared_mem) name size : float32_vec =
  match Hashtbl.find_opt shared.data name with
  | Some arr -> Obj.obj arr
  | None ->
      let arr =
        Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout size
      in
      Hashtbl.add shared.data name (Obj.repr arr) ;
      arr

(** Allocate a shared int32 array. If already allocated, returns existing. *)
let alloc_shared_int32 (shared : shared_mem) name size : int32_vec =
  match Hashtbl.find_opt shared.data name with
  | Some arr -> Obj.obj arr
  | None ->
      let arr = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout size in
      Hashtbl.add shared.data name (Obj.repr arr) ;
      arr

(** {1 Sequential Execution}

    Runs all threads in sequence. Barriers are no-ops. *)

let run_sequential ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (kernel : thread_state -> 'a -> unit) (args : 'a) : unit =
  (* Iterate over all blocks *)
  for block_idx_z = 0 to gz - 1 do
    for block_idx_y = 0 to gy - 1 do
      for block_idx_x = 0 to gx - 1 do
        (* Iterate over all threads in block *)
        for thread_idx_z = 0 to bz - 1 do
          for thread_idx_y = 0 to by - 1 do
            for thread_idx_x = 0 to bx - 1 do
              let state =
                {
                  thread_idx_x;
                  thread_idx_y;
                  thread_idx_z;
                  block_idx_x;
                  block_idx_y;
                  block_idx_z;
                  block_dim_x = bx;
                  block_dim_y = by;
                  block_dim_z = bz;
                  grid_dim_x = gx;
                  grid_dim_y = gy;
                  grid_dim_z = gz;
                  barrier = (fun () -> ());
                  (* No-op in sequential mode *)
                }
              in
              kernel state args
            done
          done
        done
      done
    done
  done

(** {1 Parallel Execution (Placeholder)}

    Will use OCaml 5 effects for fiber-based parallelism with proper barriers.
    For now, falls back to sequential. *)

let run_parallel ~block ~grid kernel args =
  (* TODO: Implement with Domain per block, fiber per thread *)
  run_sequential ~block ~grid kernel args
