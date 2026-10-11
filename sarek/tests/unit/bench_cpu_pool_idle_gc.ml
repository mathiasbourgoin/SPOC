(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Cost of a minor collection in a process that ran CPU kernels once and has
   been idle since. Every live domain takes part in each minor collection, so
   idle pool workers make all later host allocation slower.

   Usage: bench_cpu_pool_idle_gc.exe [--no-workload] [IDLE_SECONDS]
   Runs one launch on every CPU pool (unless --no-workload, the control),
   sleeps IDLE_SECONDS (default 1.5), then times an allocating loop. Prints one
   line: live OS threads, collections, mean us per collection, loop time. *)

let iterations = 20_000_000

let () =
  let args = List.tl (Array.to_list Sys.argv) in
  let workload = not (List.mem "--no-workload" args) in
  let idle =
    List.find_map float_of_string_opt args |> Option.value ~default:1.5
  in
  let threads_before = Cpu_pool_workload.os_threads () in
  if workload then Cpu_pool_workload.touch_every_pool () ;
  Unix.sleepf idle ;
  let threads_after = Cpu_pool_workload.os_threads () in
  let t0 = Unix.gettimeofday () in
  let us_per_gc, collections = Cpu_pool_workload.minor_gc_cost ~iterations in
  let loop_ms = (Unix.gettimeofday () -. t0) *. 1000. in
  let show = function Some n -> string_of_int n | None -> "?" in
  Printf.printf
    "workload=%b idle=%.1fs threads_before=%s threads_after=%s minor_gcs=%d \
     us_per_gc=%.1f loop_ms=%.1f\n\
     %!"
    workload
    idle
    (show threads_before)
    (show threads_after)
    collections
    us_per_gc
    loop_ms
