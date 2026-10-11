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

(* The block, thread and parallel pools of at most [cap] workers each, the
   fission dispatcher and the idle reaper; each domain is two OS threads. *)
let max_live_threads = 2 * ((3 * cap) + 2)

(* Every pool, including the Native block pool (barrier and barrier-free
   paths), which the shared workload leaves out. *)
let touch_all () =
  Cpu_pool_workload.touch_every_pool () ;
  List.iter
    (fun has_barriers ->
      Sarek.Sarek_cpu_runtime.run_parallel
        ~has_barriers
        ~block:Cpu_pool_workload.block
        ~grid:Cpu_pool_workload.grid
        Cpu_pool_workload.increment
        (Array.make Cpu_pool_workload.elements 0))
    [true; false]

let threads_or_skip () =
  match Cpu_pool_workload.os_threads () with
  | Some n -> n
  | None ->
      print_endline "SKIP: /proc/self/status unavailable" ;
      exit 0

(* [Domain.join] can return before the joined domain's OS threads have
   exited, so a count taken right after a join may still fall: wait until it
   holds still for 100 ms (or 5 s have passed). *)
let settled_threads () =
  let deadline = Unix.gettimeofday () +. 5. in
  let rec settle last stable_since =
    Unix.sleepf 0.02 ;
    let n = threads_or_skip () in
    let now = Unix.gettimeofday () in
    if n <> last then settle n now
    else if now -. stable_since >= 0.1 || now > deadline then n
    else settle n stable_since
  in
  let n = threads_or_skip () in
  settle n (Unix.gettimeofday ())

let rec wait_until ~deadline cond =
  if cond () || Unix.gettimeofday () > deadline then cond ()
  else begin
    Unix.sleepf 0.02 ;
    wait_until ~deadline cond
  end

let in_seconds s = Unix.gettimeofday () +. s

let failures = ref 0

let check name cond =
  if cond then Printf.printf "  ok   %s\n%!" name
  else begin
    incr failures ;
    Printf.printf "  FAIL %s\n%!" name
  end

let module_live () = Sarek_interp.Cpu_pool_lifecycle.any_live ()

(* Nothing of the pools is left: no live pool, dispatcher or reaper, and no
   OS thread beyond the baseline (fewer is fine: the baseline may include
   threads that were still exiting). *)
let all_retired ~base =
  wait_until ~deadline:(in_seconds 5.) (fun () -> not (module_live ()))
  && wait_until ~deadline:(in_seconds 5.) (fun () -> threads_or_skip () <= base)

let () =
  Unix.putenv "SAREK_CPU_POOL_MAX_DOMAINS" (string_of_int cap) ;
  Unix.putenv "SAREK_CPU_POOL_IDLE_MS" (string_of_int idle_ms) ;
  (* The first spawn also starts the main domain's backup thread, which stays:
     count from after it. *)
  Domain.join (Domain.spawn ignore) ;
  let base = settled_threads () in
  touch_all () ;
  let live = threads_or_skip () - base in
  Printf.printf "threads: %d at start, +%d with the pools alive\n%!" base live ;
  check
    (Printf.sprintf
       "at most %d pool threads under a cap of %d"
       max_live_threads
       cap)
    (live <= max_live_threads) ;
  check "every pool domain retired after the idle timeout" (all_retired ~base) ;
  touch_all () ;
  check "a launch after retirement recreates the pools" (module_live ()) ;
  Sarek_interp.Cpu_pool_lifecycle.retire_idle_now () ;
  check
    "retire_idle_now retires every pool at once, the reaper follows"
    (all_retired ~base) ;
  exit (if !failures = 0 then 0 else 1)
