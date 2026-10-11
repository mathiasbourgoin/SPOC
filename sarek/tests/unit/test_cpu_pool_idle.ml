(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Lifecycle of the process-wide CPU worker pools, counted in OS threads (one
   per domain plus its backup thread):
   - sized under SAREK_CPU_POOL_MAX_DOMAINS while alive;
   - all retired, no domain left, once idle for SAREK_CPU_POOL_IDLE_MS;
   - created again by the next launch;
   - retired at once by [Cpu_pool_lifecycle.retire_idle_now].
   Skipped where /proc/self/status is unavailable. *)

let cap = 3

let idle_ms = 100

(* Four pools of at most [cap] workers, the fission dispatcher and the idle
   reaper; each domain is two OS threads. *)
let max_live_threads = 2 * ((4 * cap) + 2)

(* Every pool, including the Native block pool, which the shared workload
   leaves out. *)
let touch_all () =
  Cpu_pool_workload.touch_every_pool () ;
  Sarek.Sarek_cpu_runtime.run_parallel
    ~has_barriers:true
    ~block:Cpu_pool_workload.block
    ~grid:Cpu_pool_workload.grid
    Cpu_pool_workload.increment
    (Array.make Cpu_pool_workload.elements 0)

let threads_or_skip () =
  match Cpu_pool_workload.os_threads () with
  | Some n -> n
  | None ->
      print_endline "SKIP: /proc/self/status unavailable" ;
      exit 0

let rec wait_for_threads ~target ~deadline =
  let n = threads_or_skip () in
  if n <= target || Unix.gettimeofday () > deadline then n
  else begin
    Unix.sleepf 0.02 ;
    wait_for_threads ~target ~deadline
  end

let failures = ref 0

let check name cond =
  if cond then Printf.printf "  ok   %s\n%!" name
  else begin
    incr failures ;
    Printf.printf "  FAIL %s\n%!" name
  end

let () =
  Unix.putenv "SAREK_CPU_POOL_MAX_DOMAINS" (string_of_int cap) ;
  Unix.putenv "SAREK_CPU_POOL_IDLE_MS" (string_of_int idle_ms) ;
  (* The first spawn also starts the main domain's backup thread, which stays:
     count from after it. *)
  Domain.join (Domain.spawn ignore) ;
  let base = threads_or_skip () in
  touch_all () ;
  let live = threads_or_skip () - base in
  Printf.printf "threads: %d at start, +%d with the pools alive\n%!" base live ;
  check
    (Printf.sprintf
       "at most %d pool threads under a cap of %d"
       max_live_threads
       cap)
    (live <= max_live_threads) ;
  let deadline =
    Unix.gettimeofday () +. (20. *. float_of_int idle_ms /. 1000.)
  in
  let after_idle = wait_for_threads ~target:base ~deadline - base in
  Printf.printf "threads: +%d after %d ms idle\n%!" after_idle idle_ms ;
  check "every pool domain retired after the idle timeout" (after_idle = 0) ;
  touch_all () ;
  check
    "a launch after retirement recreates the pools"
    (threads_or_skip () > base) ;
  Sarek_interp.Cpu_pool_lifecycle.retire_idle_now () ;
  let deadline = Unix.gettimeofday () +. 1. in
  let after_retire = wait_for_threads ~target:base ~deadline - base in
  check
    "retire_idle_now retires every pool at once, the reaper follows"
    (after_retire = 0 && not (Sarek_interp.Cpu_pool_lifecycle.any_live ())) ;
  exit (if !failures = 0 then 0 else 1)
