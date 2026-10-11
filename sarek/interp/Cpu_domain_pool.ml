(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The block pool shared by the parallel interpreter and the Native barrier
    path: worker domains popping tasks from one queue.

    Each launch submits its tasks under its own {!group} and waits for that
    group only, so concurrent launches from several domains neither wait for
    each other's tasks nor receive each other's errors. The pool is created on
    the first launch and retired when idle (see {!Cpu_pool_lifecycle}). *)

type task = unit -> unit

type t = {
  task_queue : task Queue.t;
  mutex : Mutex.t;
  cond : Condition.t;
  mutable shutdown : bool;
  (* [domains] MUST be a mutable field mutated in place (never rebuilt via
     [{pool with domains}]): worker domains close over the exact record
     returned by [create], and a functional update after [Domain.spawn] would
     give them and the submitters two different copies of the mutable fields.
     See sarek/tests/e2e/test_domain_pool_stress.ml. *)
  mutable domains : unit Domain.t array;
}

(** The tasks of one launch. Its fields are protected by the pool's mutex. *)
type group = {
  mutable remaining : int;
  mutable first_error : (exn * Printexc.raw_backtrace) option;
  done_cond : Condition.t;
}

let worker pool =
  let rec loop () =
    Mutex.lock pool.mutex ;
    while Queue.is_empty pool.task_queue && not pool.shutdown do
      Condition.wait pool.cond pool.mutex
    done ;
    if pool.shutdown && Queue.is_empty pool.task_queue then
      Mutex.unlock pool.mutex
    else begin
      let task = Queue.pop pool.task_queue in
      Mutex.unlock pool.mutex ;
      task () ;
      loop ()
    end
  in
  loop ()

let create num_domains =
  let pool =
    {
      task_queue = Queue.create ();
      mutex = Mutex.create ();
      cond = Condition.create ();
      shutdown = false;
      domains = [||];
    }
  in
  let stop () =
    Mutex.protect pool.mutex (fun () ->
        pool.shutdown <- true ;
        Condition.broadcast pool.cond)
  in
  pool.domains <-
    Cpu_pool_lifecycle.spawn_workers
      num_domains
      (fun _ -> worker pool)
      ~abort:stop ;
  pool

(** Make the workers exit once the queue is empty, and join them. *)
let shutdown pool =
  Mutex.protect pool.mutex (fun () ->
      pool.shutdown <- true ;
      Condition.broadcast pool.cond) ;
  Array.iter Domain.join pool.domains

let group () =
  {remaining = 0; first_error = None; done_cond = Condition.create ()}

(* The wrapper never raises: a task's exception is recorded on its group and
   re-raised by [wait], never dropped. *)
let submit pool group task =
  let run () =
    let error =
      try
        task () ;
        None
      with exn -> Some (exn, Printexc.get_raw_backtrace ())
    in
    Mutex.protect pool.mutex (fun () ->
        if Option.is_none group.first_error then group.first_error <- error ;
        group.remaining <- group.remaining - 1 ;
        if group.remaining = 0 then Condition.broadcast group.done_cond)
  in
  Mutex.protect pool.mutex (fun () ->
      group.remaining <- group.remaining + 1 ;
      Queue.add run pool.task_queue ;
      Condition.signal pool.cond)

(** Wait until every task of [group] has run; re-raise the first task error. *)
let wait pool group =
  Mutex.lock pool.mutex ;
  while group.remaining > 0 do
    Condition.wait group.done_cond pool.mutex
  done ;
  let first_error = group.first_error in
  group.first_error <- None ;
  Mutex.unlock pool.mutex ;
  match first_error with
  | Some (exn, bt) -> Printexc.raise_with_backtrace exn bt
  | None -> ()

let slot =
  Cpu_pool_lifecycle.slot
    ~create:(fun () -> create (Cpu_pool_lifecycle.domain_count ~default:4 ()))
    ~retire:shutdown

let run n (f : int -> unit) =
  if n > 0 then
    Cpu_pool_lifecycle.with_pool slot (fun pool ->
        let g = group () in
        for i = 0 to n - 1 do
          submit pool g (fun () -> f i)
        done ;
        wait pool g)
