(** Sarek_cpu_runtime - CPU runtime for generated native kernels

    This module provides the runtime support for kernels compiled to native
    OCaml code by the Sarek PPX. Unlike Sarek_interp which interprets the AST,
    this module is used by generated code that runs at full native speed. *)

(** Re-export Float32 module for use in generated kernels *)
module Float32 = Sarek_float32

(** Execution mode for native kernels *)
type exec_mode =
  | Sequential  (** Single-threaded, barriers are no-ops *)
  | Parallel  (** Spawn domains per kernel launch *)
  | Threadpool  (** Use persistent thread pool (fission mode) *)

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
    block. Uses regular OCaml arrays to support custom types.

    Implementation note: We use a single hashtable with Obj.t values because
    shared memory can hold arrays of any type (int, float, custom records, etc.)
    and the type is only known at the call site. The typed accessors below
    provide type-safe access for common cases. *)

type shared_mem = {
  int_arrays : (string, int array) Hashtbl.t;
  float_arrays : (string, float array) Hashtbl.t;
  int32_arrays : (string, int32 array) Hashtbl.t;
  int64_arrays : (string, int64 array) Hashtbl.t;
  other_arrays : (string, Obj.t) Hashtbl.t;
}

let create_shared () =
  {
    int_arrays = Hashtbl.create 2;
    float_arrays = Hashtbl.create 2;
    int32_arrays = Hashtbl.create 2;
    int64_arrays = Hashtbl.create 2;
    other_arrays = Hashtbl.create 2;
  }

(** Typed allocators for common array types - no Obj.magic needed *)

let alloc_shared_int (shared : shared_mem) name size (default : int) : int array
    =
  match Hashtbl.find_opt shared.int_arrays name with
  | Some arr -> arr
  | None ->
      let arr = Array.make size default in
      Hashtbl.add shared.int_arrays name arr ;
      arr

let alloc_shared_float (shared : shared_mem) name size (default : float) :
    float array =
  match Hashtbl.find_opt shared.float_arrays name with
  | Some arr -> arr
  | None ->
      let arr = Array.make size default in
      Hashtbl.add shared.float_arrays name arr ;
      arr

let alloc_shared_int32 (shared : shared_mem) name size (default : int32) :
    int32 array =
  match Hashtbl.find_opt shared.int32_arrays name with
  | Some arr -> arr
  | None ->
      let arr = Array.make size default in
      Hashtbl.add shared.int32_arrays name arr ;
      arr

let alloc_shared_int64 (shared : shared_mem) name size (default : int64) :
    int64 array =
  match Hashtbl.find_opt shared.int64_arrays name with
  | Some arr -> arr
  | None ->
      let arr = Array.make size default in
      Hashtbl.add shared.int64_arrays name arr ;
      arr

(** Generic allocator for custom types - uses Obj.t for type erasure *)
let alloc_shared (shared : shared_mem) name size (default : 'a) : 'a array =
  match Hashtbl.find_opt shared.other_arrays name with
  | Some arr -> Obj.obj arr
  | None ->
      let arr = Array.make size default in
      Hashtbl.add shared.other_arrays name (Obj.repr arr) ;
      arr

(** {1 Sequential Execution}

    Runs all threads in sequence with proper BSP barrier synchronization. Uses
    the same effect-based approach as parallel, but without domains. *)

(** Effect for yielding control at barrier - declared here for sequential use *)
type _ Effect.t += Barrier : unit Effect.t

(** Run a block sequentially with BSP-style barrier synchronization. Same
    algorithm as run_block_with_barriers but simpler since no parallelism.

    Optimized version: pre-allocate thread states and reuse effect handlers. *)
let run_block_sequential_bsp ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    ~block_idx:(block_x, block_y, block_z)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  let num_threads = bx * by * bz in
  let shared = create_shared () in

  (* Continuations waiting at barrier *)
  let waiting : (unit, unit) Effect.Deep.continuation option array =
    Array.make num_threads None
  in
  let num_waiting = ref 0 in
  let num_completed = ref 0 in

  (* Pre-compute int32 values *)
  let bx32 = Int32.of_int bx in
  let by32 = Int32.of_int by in
  let bz32 = Int32.of_int bz in
  let gx32 = Int32.of_int gx in
  let gy32 = Int32.of_int gy in
  let gz32 = Int32.of_int gz in
  let block_x32 = Int32.of_int block_x in
  let block_y32 = Int32.of_int block_y in
  let block_z32 = Int32.of_int block_z in

  (* Pre-compute thread index lookup tables *)
  let thread_x_table = Array.init bx Int32.of_int in
  let thread_y_table = Array.init by Int32.of_int in
  let thread_z_table = if bz > 1 then Array.init bz Int32.of_int else [|0l|] in

  (* Pre-allocate all thread states *)
  let states =
    Array.init num_threads (fun tid ->
        let thread_x = tid mod bx in
        let thread_y = tid / bx mod by in
        let thread_z = tid / (bx * by) in
        {
          thread_idx_x = thread_x_table.(thread_x);
          thread_idx_y = thread_y_table.(thread_y);
          thread_idx_z = thread_z_table.(thread_z);
          block_idx_x = block_x32;
          block_idx_y = block_y32;
          block_idx_z = block_z32;
          block_dim_x = bx32;
          block_dim_y = by32;
          block_dim_z = bz32;
          grid_dim_x = gx32;
          grid_dim_y = gy32;
          grid_dim_z = gz32;
          barrier = (fun () -> Effect.perform Barrier);
        })
  in

  (* Shared effect handler - reused for all threads *)
  let handler tid =
    {
      Effect.Deep.retc = (fun () -> incr num_completed);
      exnc = raise;
      effc =
        (fun (type a) (eff : a Effect.t) ->
          match eff with
          | Barrier ->
              Some
                (fun (k : (a, unit) Effect.Deep.continuation) ->
                  waiting.(tid) <- Some (Obj.magic k) ;
                  incr num_waiting)
          | _ -> None);
    }
  in

  (* Run a single thread with effect handler *)
  let run_thread tid =
    Effect.Deep.match_with
      (fun () -> kernel states.(tid) shared args)
      ()
      (handler tid)
  in

  (* Resume a waiting thread *)
  let resume_thread tid =
    match waiting.(tid) with
    | Some k ->
        waiting.(tid) <- None ;
        Effect.Deep.match_with
          (fun () -> Effect.Deep.continue k ())
          ()
          (handler tid)
    | None -> ()
  in

  (* Initial run: start all threads *)
  for tid = 0 to num_threads - 1 do
    run_thread tid
  done ;

  (* Superstep loop: while threads are waiting at barriers *)
  while !num_waiting > 0 do
    let to_resume = !num_waiting in
    num_waiting := 0 ;
    for tid = 0 to num_threads - 1 do
      if Option.is_some waiting.(tid) then resume_thread tid
    done ;
    if !num_waiting = to_resume && !num_completed < num_threads then
      failwith "BSP deadlock: no progress made"
  done

let run_sequential ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  for block_z = 0 to gz - 1 do
    for block_y = 0 to gy - 1 do
      for block_x = 0 to gx - 1 do
        run_block_sequential_bsp
          ~block:(bx, by, bz)
          ~grid:(gx, gy, gz)
          ~block_idx:(block_x, block_y, block_z)
          kernel
          args
      done
    done
  done

(** {1 Parallel Execution}

    Uses OCaml 5 Domain parallelism with effects for fibers.
    - Fixed pool of N domains (one per core)
    - Blocks are distributed across domains
    - Threads within a block run as fibers with proper barrier sync

    BSP Model: For barrier-based kernels, we use a superstep execution model: 1.
    Each thread is a fiber that can be suspended at barriers 2. All threads run
    until they hit a barrier (or complete) 3. When all threads have reached the
    barrier, all are resumed 4. Repeat until all threads complete

    The Barrier effect is declared in the sequential section above. *)

(** Run a block with BSP-style barrier synchronization. Each thread is a
    separate fiber. All threads run until barrier, then all resume together.

    Uses Effect.Shallow for better performance - avoids reinstalling handlers on
    every resume, which is significant for BSP execution with many barriers. *)
let run_block_with_barriers ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    ~block_idx:(block_x, block_y, block_z)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  let num_threads = bx * by * bz in
  let shared = create_shared () in

  (* Thread status: 0 = running, 1 = waiting at barrier, 2 = completed *)
  let status = Array.make num_threads 0 in
  (* Shallow continuations stored as option for type safety *)
  let conts : (unit, unit) Effect.Shallow.continuation option array =
    Array.make num_threads None
  in
  let num_waiting = ref 0 in
  let num_completed = ref 0 in

  (* Pre-compute int32 values *)
  let bx32 = Int32.of_int bx in
  let by32 = Int32.of_int by in
  let bz32 = Int32.of_int bz in
  let gx32 = Int32.of_int gx in
  let gy32 = Int32.of_int gy in
  let gz32 = Int32.of_int gz in
  let block_x32 = Int32.of_int block_x in
  let block_y32 = Int32.of_int block_y in
  let block_z32 = Int32.of_int block_z in

  (* Pre-compute thread index lookup tables *)
  let thread_x_table = Array.init bx Int32.of_int in
  let thread_y_table = Array.init by Int32.of_int in
  let thread_z_table = if bz > 1 then Array.init bz Int32.of_int else [|0l|] in

  (* Pre-allocate all thread states *)
  let states =
    Array.init num_threads (fun tid ->
        let thread_x = tid mod bx in
        let thread_y = tid / bx mod by in
        let thread_z = tid / (bx * by) in
        {
          thread_idx_x = thread_x_table.(thread_x);
          thread_idx_y = thread_y_table.(thread_y);
          thread_idx_z = thread_z_table.(thread_z);
          block_idx_x = block_x32;
          block_idx_y = block_y32;
          block_idx_z = block_z32;
          block_dim_x = bx32;
          block_dim_y = by32;
          block_dim_z = bz32;
          grid_dim_x = gx32;
          grid_dim_y = gy32;
          grid_dim_z = gz32;
          barrier = (fun () -> Effect.perform Barrier);
        })
  in

  (* Shallow handler - much lighter weight than Deep handlers *)
  let handler tid =
    {
      Effect.Shallow.retc =
        (fun () ->
          status.(tid) <- 2 ;
          incr num_completed);
      exnc = raise;
      effc =
        (fun (type a) (eff : a Effect.t) ->
          match eff with
          | Barrier ->
              Some
                (fun (k : (a, unit) Effect.Shallow.continuation) ->
                  conts.(tid) <- Some (Obj.magic k) ;
                  status.(tid) <- 1 ;
                  incr num_waiting)
          | _ -> None);
    }
  in

  (* Create fibers for all threads *)
  let fibers =
    Array.init num_threads (fun tid ->
        Effect.Shallow.fiber (fun () -> kernel states.(tid) shared args))
  in

  (* Initial run: start all threads with shallow continue_with *)
  for tid = 0 to num_threads - 1 do
    Effect.Shallow.continue_with fibers.(tid) () (handler tid)
  done ;

  (* Superstep loop: while threads are waiting at barriers *)
  while !num_waiting > 0 do
    num_waiting := 0 ;
    for tid = 0 to num_threads - 1 do
      if status.(tid) = 1 then begin
        status.(tid) <- 0 ;
        match conts.(tid) with
        | Some k ->
            conts.(tid) <- None ;
            Effect.Shallow.continue_with k () (handler tid)
        | None -> ()
      end
    done
  done

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
  let empty_shared = create_shared () in
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

(** Run kernel in parallel with BSP-style barriers. Distributes blocks across
    domain pool, uses effect-based barriers for thread sync within each block.
*)
let run_parallel_with_barriers ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  let pool = get_pool () in
  for block_z = 0 to gz - 1 do
    for block_y = 0 to gy - 1 do
      for block_x = 0 to gx - 1 do
        DomainPool.submit pool (fun () ->
            run_block_with_barriers
              ~block:(bx, by, bz)
              ~grid:(gx, gy, gz)
              ~block_idx:(block_x, block_y, block_z)
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

(** {1 Persistent Thread Pool for Fission Mode}

    A pool of worker domains that stay alive across kernel launches. This
    eliminates the Domain spawn/join overhead that kills performance for
    workloads with many consecutive kernel launches (like odd-even sort).

    Design:
    - N worker domains created once, stay alive until program exit
    - Each kernel launch distributes work items to workers
    - Workers wait on a condition variable between kernels
    - Barriers synchronize all workers between kernel launches *)

module ThreadPool = struct
  (** Work item: a range of global thread IDs to execute *)
  type work_item = {
    kernel : thread_state -> shared_mem -> Obj.t array -> unit;
    args : Obj.t array;
    start_tid : int;
    end_tid : int;
    (* Kernel dimensions *)
    bx : int;
    by : int;
    bz : int;
    gx : int;
    gy : int;
    gz : int;
    threads_per_block : int;
    (* Pre-computed int32 tables - shared across workers *)
    bx32 : int32;
    by32 : int32;
    bz32 : int32;
    gx32 : int32;
    gy32 : int32;
    gz32 : int32;
    thread_x_table : int32 array;
    thread_y_table : int32 array;
    thread_z_table : int32 array;
    block_x_table : int32 array;
    block_y_table : int32 array;
    block_z_table : int32 array;
  }

  type t = {
    num_workers : int;
    mutable workers : unit Domain.t array; (* Mutable to allow in-place init *)
    (* Work distribution *)
    work : work_item option Atomic.t array; (* One slot per worker *)
    (* Synchronization *)
    mutex : Mutex.t;
    work_ready : Condition.t;
    work_done : Condition.t;
    mutable pending_workers : int;
    mutable shutdown : bool;
    mutable generation : int; (* Incremented each kernel launch *)
  }

  (** Worker function - runs in each domain *)
  let worker_fn pool worker_id =
    let last_gen = ref 0 in
    (* Start at 0 to match initial pool.generation *)
    let empty_shared = create_shared () in
    let noop_barrier = fun () -> () in
    (* Pre-allocate mutable state *)
    let state =
      {
        thread_idx_x = 0l;
        thread_idx_y = 0l;
        thread_idx_z = 0l;
        block_idx_x = 0l;
        block_idx_y = 0l;
        block_idx_z = 0l;
        block_dim_x = 0l;
        block_dim_y = 0l;
        block_dim_z = 0l;
        grid_dim_x = 0l;
        grid_dim_y = 0l;
        grid_dim_z = 0l;
        barrier = noop_barrier;
      }
    in
    let rec loop () =
      (* Wait for new work - generation must change *)
      Mutex.lock pool.mutex ;
      while pool.generation = !last_gen && not pool.shutdown do
        Condition.wait pool.work_ready pool.mutex
      done ;
      if pool.shutdown then begin
        Mutex.unlock pool.mutex ;
        ()
      end
      else begin
        last_gen := pool.generation ;
        Mutex.unlock pool.mutex ;
        (* Get our work item - only signal completion if we had work *)
        let had_work =
          match Atomic.get pool.work.(worker_id) with
          | None -> false
          | Some w ->
              (* Update dimension fields once per kernel *)
              Obj.set_field (Obj.repr state) 6 (Obj.repr w.bx32) ;
              Obj.set_field (Obj.repr state) 7 (Obj.repr w.by32) ;
              Obj.set_field (Obj.repr state) 8 (Obj.repr w.bz32) ;
              Obj.set_field (Obj.repr state) 9 (Obj.repr w.gx32) ;
              Obj.set_field (Obj.repr state) 10 (Obj.repr w.gy32) ;
              Obj.set_field (Obj.repr state) 11 (Obj.repr w.gz32) ;
              (* Execute our range of threads *)
              for global_tid = w.start_tid to w.end_tid - 1 do
                let block_id = global_tid / w.threads_per_block in
                let local_tid = global_tid - (block_id * w.threads_per_block) in
                let block_x = block_id mod w.gx in
                let block_y = block_id / w.gx mod w.gy in
                let block_z = block_id / (w.gx * w.gy) in
                let thread_x = local_tid mod w.bx in
                let thread_y = local_tid / w.bx mod w.by in
                let thread_z = local_tid / (w.bx * w.by) in
                Obj.set_field
                  (Obj.repr state)
                  0
                  (Obj.repr w.thread_x_table.(thread_x)) ;
                Obj.set_field
                  (Obj.repr state)
                  1
                  (Obj.repr w.thread_y_table.(thread_y)) ;
                Obj.set_field
                  (Obj.repr state)
                  2
                  (Obj.repr w.thread_z_table.(thread_z)) ;
                Obj.set_field
                  (Obj.repr state)
                  3
                  (Obj.repr w.block_x_table.(block_x)) ;
                Obj.set_field
                  (Obj.repr state)
                  4
                  (Obj.repr w.block_y_table.(block_y)) ;
                Obj.set_field
                  (Obj.repr state)
                  5
                  (Obj.repr w.block_z_table.(block_z)) ;
                w.kernel state empty_shared w.args
              done ;
              Atomic.set pool.work.(worker_id) None ;
              true
        in
        (* Signal completion only if we had work *)
        if had_work then begin
          Mutex.lock pool.mutex ;
          pool.pending_workers <- pool.pending_workers - 1 ;
          if pool.pending_workers = 0 then Condition.broadcast pool.work_done ;
          Mutex.unlock pool.mutex
        end ;
        loop ()
      end
    in
    loop ()

  (** Create thread pool with N workers *)
  let create num_workers =
    let pool =
      {
        num_workers;
        workers = [||];
        (* Will be mutated in place *)
        work = Array.init num_workers (fun _ -> Atomic.make None);
        mutex = Mutex.create ();
        work_ready = Condition.create ();
        work_done = Condition.create ();
        pending_workers = 0;
        shutdown = false;
        generation = 0;
      }
    in
    (* Spawn workers - they all reference the same pool record *)
    pool.workers <-
      Array.init num_workers (fun i ->
          Domain.spawn (fun () -> worker_fn pool i)) ;
    pool

  (** Submit kernel to thread pool and wait for completion *)
  let run_kernel pool ~block:(bx, by, bz) ~grid:(gx, gy, gz)
      (kernel : thread_state -> shared_mem -> Obj.t array -> unit)
      (args : Obj.t array) =
    let threads_per_block = bx * by * bz in
    let total_blocks = gx * gy * gz in
    let total_threads = total_blocks * threads_per_block in
    let threads_per_worker =
      (total_threads + pool.num_workers - 1) / pool.num_workers
    in
    (* Pre-compute int32 tables *)
    let bx32 = Int32.of_int bx in
    let by32 = Int32.of_int by in
    let bz32 = Int32.of_int bz in
    let gx32 = Int32.of_int gx in
    let gy32 = Int32.of_int gy in
    let gz32 = Int32.of_int gz in
    let thread_x_table = Array.init bx Int32.of_int in
    let thread_y_table = Array.init by Int32.of_int in
    let thread_z_table = Array.init bz Int32.of_int in
    let block_x_table = Array.init gx Int32.of_int in
    let block_y_table = Array.init gy Int32.of_int in
    let block_z_table = Array.init gz Int32.of_int in
    (* Distribute work *)
    Mutex.lock pool.mutex ;
    let active_workers = ref 0 in
    for i = 0 to pool.num_workers - 1 do
      let start_tid = i * threads_per_worker in
      let end_tid = min ((i + 1) * threads_per_worker) total_threads in
      if start_tid < total_threads then begin
        Atomic.set
          pool.work.(i)
          (Some
             {
               kernel;
               args;
               start_tid;
               end_tid;
               bx;
               by;
               bz;
               gx;
               gy;
               gz;
               threads_per_block;
               bx32;
               by32;
               bz32;
               gx32;
               gy32;
               gz32;
               thread_x_table;
               thread_y_table;
               thread_z_table;
               block_x_table;
               block_y_table;
               block_z_table;
             }) ;
        incr active_workers
      end
    done ;
    pool.pending_workers <- !active_workers ;
    pool.generation <- pool.generation + 1 ;
    Condition.broadcast pool.work_ready ;
    (* Wait for completion *)
    while pool.pending_workers > 0 do
      Condition.wait pool.work_done pool.mutex
    done ;
    Mutex.unlock pool.mutex

  let shutdown pool =
    Mutex.lock pool.mutex ;
    pool.shutdown <- true ;
    Condition.broadcast pool.work_ready ;
    Mutex.unlock pool.mutex ;
    Array.iter Domain.join pool.workers

  let _ = shutdown (* Suppress unused warning *)
end

(** Global fission thread pool - lazily initialized *)
let fission_pool : ThreadPool.t option ref = ref None

let fission_pool_mutex = Mutex.create ()

let get_fission_pool () =
  Mutex.lock fission_pool_mutex ;
  let pool =
    match !fission_pool with
    | Some p -> p
    | None ->
        let num_cores = try Domain.recommended_domain_count () with _ -> 4 in
        let p = ThreadPool.create num_cores in
        fission_pool := Some p ;
        p
  in
  Mutex.unlock fission_pool_mutex ;
  pool

(** Run kernel using the persistent thread pool. This is like run_parallel but
    uses the fission thread pool instead of spawning new domains. For use by the
    wrapper in fission mode. *)
let run_threadpool ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  let pool = get_fission_pool () in
  (* Wrap the typed kernel to take Obj.t array *)
  let wrapped_kernel state shared _obj_args = kernel state shared args in
  ThreadPool.run_kernel
    pool
    ~block:(bx, by, bz)
    ~grid:(gx, gy, gz)
    wrapped_kernel
    [||]

(** {1 Fission Queue with Thread Pool Execution}

    Multiple queues are supported (like CUDA/OpenCL command queues):
    - Same queue_id: kernels execute in order (serialized)
    - Different queue_id: kernels can run in parallel (one dispatcher per queue)

    Each queue has a dispatcher domain that pulls kernels and submits them to
    the shared thread pool. The thread pool workers stay alive across all kernel
    launches. *)

module LaunchQueue = struct
  (** A pending kernel launch - uses wrapper signature for compatibility *)
  type launch = {
    kernel :
      mode:exec_mode ->
      block:int * int * int ->
      grid:int * int * int ->
      Obj.t array ->
      unit;
    args : Obj.t array;
    block : int * int * int;
    grid : int * int * int;
  }

  (** Queue state - one per queue_id *)
  type t = {
    queue : launch Queue.t;
    mutex : Mutex.t;
    cond : Condition.t;  (** Signal: new work available *)
    done_cond : Condition.t;  (** Signal: queue empty and idle *)
    mutable dispatcher : unit Domain.t option;
    mutable shutdown : bool;
    mutable pending : int;  (** Number of launches not yet completed *)
  }

  (** Dispatcher loop - pulls kernels and executes via wrapper The wrapper
      internally uses run_threadpool with mode=Threadpool *)
  let dispatcher_loop (q : t) =
    let rec loop () =
      Mutex.lock q.mutex ;
      (* Wait for work or shutdown *)
      while Queue.is_empty q.queue && not q.shutdown do
        Condition.wait q.cond q.mutex
      done ;
      if q.shutdown && Queue.is_empty q.queue then begin
        Mutex.unlock q.mutex ;
        () (* Exit dispatcher *)
      end
      else begin
        let launch = Queue.pop q.queue in
        Mutex.unlock q.mutex ;
        (* Execute via the wrapper - Threadpool mode uses persistent pool *)
        launch.kernel
          ~mode:Threadpool
          ~block:launch.block
          ~grid:launch.grid
          launch.args ;
        (* Mark completion *)
        Mutex.lock q.mutex ;
        q.pending <- q.pending - 1 ;
        if q.pending = 0 then Condition.broadcast q.done_cond ;
        Mutex.unlock q.mutex ;
        loop ()
      end
    in
    loop ()

  (** Create a new launch queue (dispatcher starts on first enqueue) *)
  let create () : t =
    {
      queue = Queue.create ();
      mutex = Mutex.create ();
      cond = Condition.create ();
      done_cond = Condition.create ();
      dispatcher = None;
      shutdown = false;
      pending = 0;
    }

  (** Ensure dispatcher is running *)
  let ensure_dispatcher (q : t) =
    if q.dispatcher = None then
      q.dispatcher <- Some (Domain.spawn (fun () -> dispatcher_loop q))

  (** Enqueue a kernel launch - dispatcher starts processing immediately *)
  let enqueue (q : t) ~kernel ~args ~block ~grid =
    Mutex.lock q.mutex ;
    ensure_dispatcher q ;
    Queue.add {kernel; args; block; grid} q.queue ;
    q.pending <- q.pending + 1 ;
    Condition.signal q.cond ;
    Mutex.unlock q.mutex

  (** Wait for all pending launches to complete (flush/barrier) *)
  let flush (q : t) =
    Mutex.lock q.mutex ;
    while q.pending > 0 do
      Condition.wait q.done_cond q.mutex
    done ;
    Mutex.unlock q.mutex

  (** Shutdown the dispatcher (for cleanup) *)
  let shutdown (q : t) =
    Mutex.lock q.mutex ;
    q.shutdown <- true ;
    Condition.signal q.cond ;
    Mutex.unlock q.mutex ;
    match q.dispatcher with
    | Some d ->
        Domain.join d ;
        q.dispatcher <- None
    | None -> ()

  let _ = shutdown (* Suppress unused warning *)
end

(** Multiple queues indexed by queue_id - allows concurrent execution *)
let fission_queues : (int, LaunchQueue.t) Hashtbl.t = Hashtbl.create 4

let fission_queues_mutex = Mutex.create ()

let get_fission_queue queue_id =
  Mutex.lock fission_queues_mutex ;
  let q =
    match Hashtbl.find_opt fission_queues queue_id with
    | Some q -> q
    | None ->
        let q = LaunchQueue.create () in
        Hashtbl.add fission_queues queue_id q ;
        q
  in
  Mutex.unlock fission_queues_mutex ;
  q

(** Enqueue a kernel for fission execution on a specific queue. The kernel
    starts executing immediately in the background. Kernels on same queue
    execute in order; different queues run in parallel. The kernel is the
    wrapper function from cpu_kern. *)
let enqueue_fission ?(queue_id = 0) ~kernel ~args ~block ~grid () =
  let q = get_fission_queue queue_id in
  LaunchQueue.enqueue q ~kernel ~args ~block ~grid

(** Wait for a specific queue to complete. *)
let flush_fission_queue queue_id =
  Mutex.lock fission_queues_mutex ;
  let q = Hashtbl.find_opt fission_queues queue_id in
  Mutex.unlock fission_queues_mutex ;
  match q with
  | Some q -> LaunchQueue.flush q
  | None -> () (* Queue doesn't exist = nothing to flush *)

(** Wait for all fission queues to complete. Called by Devices.flush. *)
let flush_fission () =
  (* Get all queues under lock, then flush without lock *)
  Mutex.lock fission_queues_mutex ;
  let queues = Hashtbl.fold (fun _ q acc -> q :: acc) fission_queues [] in
  Mutex.unlock fission_queues_mutex ;
  List.iter LaunchQueue.flush queues
