(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A launch on the shared CPU block pool from inside one of its own tasks (a
   Native [run_parallel] kernel that itself calls [run_parallel]) completes,
   barrier-free or with barriers, whatever the pool size, including a single
   worker. A watchdog exits 124 if a launch does not finish (a deadlock fails
   the test instead of hanging CI). *)

open Sarek.Sarek_cpu_runtime

let watchdog_seconds = 30.

let start_watchdog finished =
  ignore
    (Domain.spawn (fun () ->
         let deadline = Unix.gettimeofday () +. watchdog_seconds in
         while (not (Atomic.get finished)) && Unix.gettimeofday () < deadline do
           Unix.sleepf 0.05
         done ;
         if not (Atomic.get finished) then begin
           prerr_endline "FAIL: nested launch did not finish (deadlock)" ;
           Unix._exit 124
         end))

(* One launch of [blocks] blocks of 64 threads; checks every element. *)
let launch ~barriers blocks =
  let res = Array.make (blocks * 64) 0 in
  let kernel state _ () =
    let i = Int32.to_int (global_idx_x state) in
    res.(i) <- i + 1
  in
  run_parallel
    ~has_barriers:barriers
    ~block:(64, 1, 1)
    ~grid:(blocks, 1, 1)
    kernel
    () ;
  Array.iteri (fun i v -> if v <> i + 1 then failwith "inner launch result") res

(* Outer launch of [blocks] blocks; thread 0 of each block runs an inner
   launch. *)
let nested ~outer ~inner blocks =
  let kernel state _ () =
    if Int32.to_int state.thread_idx_x = 0 then launch ~barriers:inner 2
  in
  run_parallel
    ~has_barriers:outer
    ~block:(64, 1, 1)
    ~grid:(blocks, 1, 1)
    kernel
    ()

let with_cap cap f =
  Sarek_interp.Cpu_pool_lifecycle.retire_idle_now () ;
  Sarek_interp.Cpu_pool_lifecycle.set_max_domains cap ;
  f () ;
  Printf.printf "  ok   cap %d\n%!" cap

let () =
  let finished = Atomic.make false in
  start_watchdog finished ;
  List.iter
    (fun cap ->
      with_cap cap (fun () ->
          List.iter
            (fun (outer, inner) ->
              nested ~outer ~inner 1 ;
              nested ~outer ~inner 64)
            [(false, false); (false, true); (true, false); (true, true)]))
    [1; 2; 4] ;
  Atomic.set finished true ;
  print_endline "nested launches: all completed"
