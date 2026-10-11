(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The block pool shared by the parallel interpreter and the Native CPU
    runtime: worker domains taking tasks from one queue. Created on the first
    launch and retired when idle (see {!Cpu_pool_lifecycle}).

    Each launch waits for its own tasks only and receives only their errors, so
    concurrent launches from several domains do not interfere.

    A launch from inside one of its tasks (a nested launch) needs a free worker:
    it completes while fewer tasks run than there are workers, and deadlocks
    when every worker is waiting on a nested launch (always, with a single
    worker, e.g. under [SAREK_CPU_POOL_MAX_DOMAINS=1]). *)

type task = unit -> unit

(** [run n f] runs [f 0] .. [f (n - 1)] on the pool, in any order and in
    parallel, and returns when all have run, re-raising the first exception one
    of them raised. [n <= 0] runs nothing and creates no pool. *)
val run : int -> (int -> unit) -> unit
