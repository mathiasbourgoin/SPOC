(** Sarek_cpu_runtime - CPU runtime for generated native kernels

    This module provides the runtime support for kernels compiled to native
    OCaml code by the Sarek PPX. Unlike Sarek_interp which interprets the AST,
    this module is used by generated code that runs at full native speed. *)

(** {1 Thread State}

    Thread state is passed to each generated kernel function. The kernel reads
    thread/block/grid indices from this record.

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
      (** Barrier function - no-op in sequential, effect in parallel *)
}

(** {1 Global Index Helpers} *)

let global_idx_x st =
  Int32.add (Int32.mul st.block_idx_x st.block_dim_x) st.thread_idx_x

let global_idx_y st =
  Int32.add (Int32.mul st.block_idx_y st.block_dim_y) st.thread_idx_y

let global_idx_z st =
  Int32.add (Int32.mul st.block_idx_z st.block_dim_z) st.thread_idx_z

let global_size_x st = Int32.mul st.grid_dim_x st.block_dim_x

let global_size_y st = Int32.mul st.grid_dim_y st.block_dim_y

let global_size_z st = Int32.mul st.grid_dim_z st.block_dim_z

(** {1 Shared Memory}

    Shared memory is allocated per-block and accessible by all threads in the
    block. Uses regular OCaml arrays to support custom types. *)

type shared_mem = {data : (string, Obj.t) Hashtbl.t}

let create_shared () = {data = Hashtbl.create 8}

(** Allocate a shared array of any type. If already allocated, returns existing.
    Uses Obj.magic to allow any element type. *)
let alloc_shared (shared : shared_mem) name size (default : 'a) : 'a array =
  match Hashtbl.find_opt shared.data name with
  | Some arr -> Obj.obj arr
  | None ->
      let arr = Array.make size default in
      Hashtbl.add shared.data name (Obj.repr arr) ;
      arr

(** {1 Sequential Execution}

    Runs all threads in sequence. Barriers are no-ops. *)

let run_sequential ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  (* Iterate over all blocks *)
  for block_z = 0 to gz - 1 do
    for block_y = 0 to gy - 1 do
      for block_x = 0 to gx - 1 do
        (* Create shared memory for this block - shared across all threads *)
        let shared = create_shared () in
        (* Iterate over all threads in block *)
        for thread_z = 0 to bz - 1 do
          for thread_y = 0 to by - 1 do
            for thread_x = 0 to bx - 1 do
              let state =
                {
                  thread_idx_x = Int32.of_int thread_x;
                  thread_idx_y = Int32.of_int thread_y;
                  thread_idx_z = Int32.of_int thread_z;
                  block_idx_x = Int32.of_int block_x;
                  block_idx_y = Int32.of_int block_y;
                  block_idx_z = Int32.of_int block_z;
                  block_dim_x = Int32.of_int bx;
                  block_dim_y = Int32.of_int by;
                  block_dim_z = Int32.of_int bz;
                  grid_dim_x = Int32.of_int gx;
                  grid_dim_y = Int32.of_int gy;
                  grid_dim_z = Int32.of_int gz;
                  barrier = (fun () -> ());
                  (* No-op in sequential mode *)
                }
              in
              kernel state shared args
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
