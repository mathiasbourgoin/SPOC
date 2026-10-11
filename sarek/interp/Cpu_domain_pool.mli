(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The block pool shared by the parallel interpreter and the Native CPU
    runtime: worker domains taking tasks from one queue. Created on the first
    launch and retired when idle (see {!Cpu_pool_lifecycle}).

    Each launch waits for its own tasks only and receives only their errors, so
    concurrent launches from several domains do not interfere.

    The launching domain runs the launch's tasks that no worker has taken yet
    while it waits. A launch from inside a task (a nested launch: an interpreter
    or [Sarek_cpu_runtime.run_parallel] kernel, barrier-free or with barriers,
    that launches again on this pool) therefore completes at any pool size,
    including a single worker. This holds for this pool only: the Native thread
    and parallel pools refuse a launch from their own kernels. *)

type task = unit -> unit

(** [run n f] runs [f 0] .. [f (n - 1)] on the pool, in any order and in
    parallel, and returns when all have run, re-raising the first exception one
    of them raised. [n <= 0] runs nothing and creates no pool. *)
val run : int -> (int -> unit) -> unit
