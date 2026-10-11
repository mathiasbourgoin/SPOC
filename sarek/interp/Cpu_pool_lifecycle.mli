(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sizing and lifetime of the process-wide CPU worker pools (the interpreter
    and Native block pool, the Native thread and parallel pools, the fission
    queue dispatchers).

    Every live domain takes part in every minor collection, so an idle worker
    still slows all host allocation in the process. Pools are therefore created
    on their first launch only, never by device enumeration, and retired once no
    launch has used them for the idle timeout: their workers exit and are
    joined. The next launch creates them again. A pool is never retired while a
    launch holds it. Retirement is done by one reaper domain that exists only
    while some pool is live.

    Configuration, read when a pool is created:
    - [SAREK_CPU_POOL_MAX_DOMAINS] (or {!set_max_domains}): upper bound on the
      workers of each pool, so a client can keep cores for itself;
    - [SAREK_CPU_POOL_IDLE_MS] (or {!set_idle_timeout}): idle time before a pool
      is retired, default 1000 ms;
    - [SAREK_DOMAIN_COUNT]: the worker count before the cap (testing aid),
      otherwise [Domain.recommended_domain_count ()]. *)

(** {1 Configuration} *)

(** [set_max_domains n] caps every pool created from now on at [n] workers.
    Raises [Invalid_argument] if [n < 1]. *)
val set_max_domains : int -> unit

(** [set_idle_timeout s] retires pools idle for [s] seconds. Raises
    [Invalid_argument] unless [s > 0.]. *)
val set_idle_timeout : float -> unit

(** [capped n] is [n] bounded by the configured maximum, and at least 1. *)
val capped : int -> int

(** Worker count for a new pool: [SAREK_DOMAIN_COUNT] if set to a positive
    integer, else [Domain.recommended_domain_count ()] (else [default]), then
    {!capped}. *)
val domain_count : default:int -> unit -> int

(** {1 Pool slots} *)

(** A process-wide pool created on demand and retired when idle. *)
type 'p slot

(** [slot ~create ~retire] registers a slot with the reaper. [create ()] makes
    the pool (sizing it with {!domain_count}); [retire p] makes its workers exit
    and joins them, and is called only when no launch holds [p]. *)
val slot : create:(unit -> 'p) -> retire:('p -> unit) -> 'p slot

(** [with_pool s f] runs [f] on the pool of [s], creating it if needed. The pool
    is not retired before [f] returns or raises. *)
val with_pool : 'p slot -> ('p -> 'a) -> 'a

(** {1 Other idle resources} *)

(** [register_reapable f] adds a resource the reaper polls: [f ~now ~idle_after]
    retires it if it has been unused since [now -. idle_after] and nothing is in
    flight, and returns [true] iff the resource is still live afterwards. *)
val register_reapable : (now:float -> idle_after:float -> bool) -> unit

(** Starts the reaper if it is not running. Called by a resource registered with
    {!register_reapable} when it becomes live. *)
val note_live : unit -> unit

(** {1 Explicit retirement} *)

(** Retire every pool and dispatcher that no launch holds now, whatever its idle
    time. Pools in use are left to the reaper. Also run [at_exit]. *)
val retire_idle_now : unit -> unit

(** [true] iff a pool, a dispatcher or the reaper is live. *)
val any_live : unit -> bool
