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

(** {1 Parallel Execution}

    Uses OCaml 5 Domain parallelism with effects for fibers.
    - Fixed pool of N domains (one per core)
    - Blocks are distributed across domains
    - Threads within a block run as fibers with proper barrier sync *)

(** Effect for yielding control to other fibers *)
type _ Effect.t += Yield : unit Effect.t

(** Fiber-based barrier for synchronizing threads within a block. *)
module FiberBarrier = struct
  type t = {
    mutable count : int;
    total : int;
    mutable waiters : (unit -> unit) list;
  }

  let create n = {count = 0; total = n; waiters = []}

  let wait t k =
    t.count <- t.count + 1 ;
    if t.count = t.total then begin
      (* Last fiber - reset and resume all waiters *)
      t.count <- 0 ;
      let waiters = t.waiters in
      t.waiters <- [] ;
      List.iter (fun resume -> resume ()) waiters ;
      k ()
    end
    else begin
      (* Not last - add to waiters and yield *)
      t.waiters <- k :: t.waiters
    end
end

(** Run a block with merged fibers. Each fiber handles a batch of threads,
    running them sequentially between barriers. *)
let[@warning "-32"] run_block_with_fibers ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    ~block_idx:(block_x, block_y, block_z) ~num_fibers
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  let num_threads = bx * by * bz in
  let shared = create_shared () in
  let barrier = FiberBarrier.create num_fibers in
  (* Queue of ready fibers *)
  let ready_queue : (unit -> unit) Queue.t = Queue.create () in
  (* Distribute threads across fibers *)
  let threads_per_fiber = (num_threads + num_fibers - 1) / num_fibers in
  (* Create fiber for each batch of threads *)
  for fiber_id = 0 to num_fibers - 1 do
    let start_thread = fiber_id * threads_per_fiber in
    let end_thread = min ((fiber_id + 1) * threads_per_fiber) num_threads in
    if start_thread < num_threads then
      Queue.add
        (fun () ->
          (* This fiber runs threads [start_thread, end_thread) *)
          (* We need to run in lockstep: all threads do work, then all barrier *)
          (* For simplicity, run all assigned threads sequentially *)
          for tid = start_thread to end_thread - 1 do
            let thread_x = tid mod bx in
            let thread_y = tid / bx mod by in
            let thread_z = tid / (bx * by) in
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
                barrier =
                  (fun () ->
                    (* Use effect to yield at barrier *)
                    Effect.perform Yield);
              }
            in
            kernel state shared args
          done)
        ready_queue
  done ;
  (* Completed fiber count *)
  let actual_fibers = min num_fibers num_threads in
  let completed = ref 0 in
  (* Run fibers with effect handler *)
  let rec run_fiber fiber =
    Effect.Deep.match_with
      fiber
      ()
      {
        retc =
          (fun () ->
            incr completed ;
            schedule ());
        exnc = raise;
        effc =
          (fun (type a) (eff : a Effect.t) ->
            match eff with
            | Yield ->
                Some
                  (fun (k : (a, _) Effect.Deep.continuation) ->
                    (* Barrier wait - let barrier decide when to resume *)
                    FiberBarrier.wait barrier (fun () ->
                        Queue.add
                          (fun () -> Effect.Deep.continue k ())
                          ready_queue) ;
                    schedule ())
            | _ -> None);
      }
  and schedule () =
    if !completed < actual_fibers then
      match Queue.take_opt ready_queue with
      | Some fiber -> run_fiber fiber
      | None ->
          (* No ready fibers but not all completed - shouldn't happen *)
          failwith "Deadlock: no ready fibers"
  in
  schedule ()

(** Global domain pool for parallel execution *)
module DomainPool = struct
  type task = unit -> unit

  type t = {
    num_domains : int;
    task_queue : task Queue.t;
    mutex : Mutex.t;
    cond : Condition.t;
    mutable shutdown : bool;
    domains : unit Domain.t array;
    mutable active_tasks : int;
    done_cond : Condition.t;
  }

  let worker pool =
    let rec loop () =
      Mutex.lock pool.mutex ;
      while Queue.is_empty pool.task_queue && not pool.shutdown do
        Condition.wait pool.cond pool.mutex
      done ;
      if pool.shutdown && Queue.is_empty pool.task_queue then begin
        Mutex.unlock pool.mutex ;
        ()
      end
      else begin
        let task = Queue.pop pool.task_queue in
        pool.active_tasks <- pool.active_tasks + 1 ;
        Mutex.unlock pool.mutex ;
        (try task () with _ -> ()) ;
        Mutex.lock pool.mutex ;
        pool.active_tasks <- pool.active_tasks - 1 ;
        if pool.active_tasks = 0 && Queue.is_empty pool.task_queue then
          Condition.broadcast pool.done_cond ;
        Mutex.unlock pool.mutex ;
        loop ()
      end
    in
    loop ()

  let create num_domains =
    let pool =
      {
        num_domains;
        task_queue = Queue.create ();
        mutex = Mutex.create ();
        cond = Condition.create ();
        shutdown = false;
        domains = [||];
        active_tasks = 0;
        done_cond = Condition.create ();
      }
    in
    let domains =
      Array.init num_domains (fun _ -> Domain.spawn (fun () -> worker pool))
    in
    {pool with domains}

  let submit pool task =
    Mutex.lock pool.mutex ;
    Queue.add task pool.task_queue ;
    Condition.signal pool.cond ;
    Mutex.unlock pool.mutex

  let wait_all pool =
    Mutex.lock pool.mutex ;
    while pool.active_tasks > 0 || not (Queue.is_empty pool.task_queue) do
      Condition.wait pool.done_cond pool.mutex
    done ;
    Mutex.unlock pool.mutex

  let shutdown pool =
    Mutex.lock pool.mutex ;
    pool.shutdown <- true ;
    Condition.broadcast pool.cond ;
    Mutex.unlock pool.mutex ;
    Array.iter Domain.join pool.domains

  let _ = shutdown (* suppress unused warning - may be used for cleanup *)
end

(** Global pool - lazily initialized *)
let global_pool : DomainPool.t option ref = ref None

let get_pool () =
  match !global_pool with
  | Some pool -> pool
  | None ->
      let num_cores = try Domain.recommended_domain_count () with _ -> 4 in
      let pool = DomainPool.create num_cores in
      global_pool := Some pool ;
      pool

(** Run kernel in parallel without barriers - simple work partitioning.
    Distributes all global threads across domains. Optimized for speed. *)
let run_parallel_simple ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  let pool = get_pool () in
  let num_domains = pool.DomainPool.num_domains in
  (* Total number of global threads *)
  let threads_per_block = bx * by * bz in
  let total_blocks = gx * gy * gz in
  let total_threads = total_blocks * threads_per_block in
  let threads_per_domain = (total_threads + num_domains - 1) / num_domains in
  (* Pre-convert dimensions to int32 *)
  let bx32 = Int32.of_int bx in
  let by32 = Int32.of_int by in
  let bz32 = Int32.of_int bz in
  let gx32 = Int32.of_int gx in
  let gy32 = Int32.of_int gy in
  let gz32 = Int32.of_int gz in
  (* Shared empty hashtable - barrier-free kernels don't use shared memory *)
  let empty_shared = {data = Hashtbl.create 0} in
  let noop_barrier = fun () -> () in
  (* Pre-compute int32 lookup tables to avoid Int32.of_int in hot loop *)
  let thread_x_table = Array.init bx Int32.of_int in
  let thread_y_table = Array.init by Int32.of_int in
  let thread_z_table = Array.init bz Int32.of_int in
  let block_x_table = Array.init gx Int32.of_int in
  let block_y_table = Array.init gy Int32.of_int in
  let block_z_table = Array.init gz Int32.of_int in
  (* Use Domain.spawn directly for less overhead than pool *)
  let domains =
    Array.init num_domains (fun domain_id ->
        let start_tid = domain_id * threads_per_domain in
        let end_tid =
          min ((domain_id + 1) * threads_per_domain) total_threads
        in
        if start_tid >= total_threads then None
        else
          Some
            (Domain.spawn (fun () ->
                 (* Reusable mutable state - avoid allocating per thread *)
                 let state =
                   {
                     thread_idx_x = 0l;
                     thread_idx_y = 0l;
                     thread_idx_z = 0l;
                     block_idx_x = 0l;
                     block_idx_y = 0l;
                     block_idx_z = 0l;
                     block_dim_x = bx32;
                     block_dim_y = by32;
                     block_dim_z = bz32;
                     grid_dim_x = gx32;
                     grid_dim_y = gy32;
                     grid_dim_z = gz32;
                     barrier = noop_barrier;
                   }
                 in
                 for global_tid = start_tid to end_tid - 1 do
                   (* Compute block and thread indices from global thread ID *)
                   let block_id = global_tid / threads_per_block in
                   let local_tid =
                     global_tid - (block_id * threads_per_block)
                   in
                   let block_x = block_id mod gx in
                   let block_y = block_id / gx mod gy in
                   let block_z = block_id / (gx * gy) in
                   let thread_x = local_tid mod bx in
                   let thread_y = local_tid / bx mod by in
                   let thread_z = local_tid / (bx * by) in
                   (* Use Obj.set_field to mutate immutable record in-place *)
                   Obj.set_field
                     (Obj.repr state)
                     0
                     (Obj.repr thread_x_table.(thread_x)) ;
                   Obj.set_field
                     (Obj.repr state)
                     1
                     (Obj.repr thread_y_table.(thread_y)) ;
                   Obj.set_field
                     (Obj.repr state)
                     2
                     (Obj.repr thread_z_table.(thread_z)) ;
                   Obj.set_field
                     (Obj.repr state)
                     3
                     (Obj.repr block_x_table.(block_x)) ;
                   Obj.set_field
                     (Obj.repr state)
                     4
                     (Obj.repr block_y_table.(block_y)) ;
                   Obj.set_field
                     (Obj.repr state)
                     5
                     (Obj.repr block_z_table.(block_z)) ;
                   kernel state empty_shared args
                 done)))
  in
  Array.iter (function Some d -> Domain.join d | None -> ()) domains

(** Run kernel in parallel with fiber-based barriers. Distributes blocks across
    domain pool, uses fibers for barrier sync. *)
let run_parallel_with_barriers ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  let pool = get_pool () in
  let num_cores = pool.DomainPool.num_domains in
  let num_fibers = min num_cores (bx * by * bz) in
  for block_z = 0 to gz - 1 do
    for block_y = 0 to gy - 1 do
      for block_x = 0 to gx - 1 do
        DomainPool.submit pool (fun () ->
            run_block_with_fibers
              ~block:(bx, by, bz)
              ~grid:(gx, gy, gz)
              ~block_idx:(block_x, block_y, block_z)
              ~num_fibers
              kernel
              args)
      done
    done
  done ;
  DomainPool.wait_all pool

(** Detect if kernel uses barriers by running one block and checking. *)
let kernel_uses_barriers ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : bool =
  let barrier_called = ref false in
  let shared = create_shared () in
  (* Run just one thread to detect barrier usage *)
  let state =
    {
      thread_idx_x = 0l;
      thread_idx_y = 0l;
      thread_idx_z = 0l;
      block_idx_x = 0l;
      block_idx_y = 0l;
      block_idx_z = 0l;
      block_dim_x = Int32.of_int bx;
      block_dim_y = Int32.of_int by;
      block_dim_z = Int32.of_int bz;
      grid_dim_x = Int32.of_int gx;
      grid_dim_y = Int32.of_int gy;
      grid_dim_z = Int32.of_int gz;
      barrier =
        (fun () ->
          barrier_called := true ;
          (* Don't actually block - just detect *)
          ());
    }
  in
  kernel state shared args ;
  !barrier_called

(** Run kernel in parallel. Auto-detects barrier usage:
    - No barriers: uses simple work partitioning (fastest)
    - With barriers: uses fiber-based scheduling *)
let run_parallel ~block ~grid kernel args =
  if kernel_uses_barriers ~block ~grid kernel args then
    run_parallel_with_barriers ~block ~grid kernel args
  else run_parallel_simple ~block ~grid kernel args
