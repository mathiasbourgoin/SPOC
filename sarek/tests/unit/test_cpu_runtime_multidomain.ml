(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Properties of the process-wide CPU pools:
   - launches issued concurrently from several domains each run to completion
     and each element of a launch's own array is written exactly once per
     launch;
   - a launch from inside a kernel of the same pool is refused with
     [Invalid_argument] instead of blocking;
   - a raising fission kernel surfaces at flush, every queue is still flushed,
     and the dispatcher keeps running.

   A watchdog domain terminates the process with status 124 if the run does not
   finish, so a hang fails the test instead of blocking CI. *)

open Sarek.Sarek_cpu_runtime

let watchdog_seconds = 30.0

let launches_per_domain = 200

let domains = 4

let block = (8, 1, 1)

let grid = (4, 1, 1)

let elements = 32

(* Alcotest redirects fds 1 and 2 while a test runs; the watchdog writes to a
   duplicate of the original stderr taken before that. *)
let start_watchdog ~stderr_copy finished =
  ignore
    (Domain.spawn (fun () ->
         let deadline = Unix.gettimeofday () +. watchdog_seconds in
         while (not (Atomic.get finished)) && Unix.gettimeofday () < deadline do
           Unix.sleepf 0.05
         done ;
         if not (Atomic.get finished) then begin
           let msg = "FAIL: cpu runtime test did not finish (hang)\n" in
           ignore (Unix.write_substring stderr_copy msg 0 (String.length msg)) ;
           Unix._exit 124
         end))

let increment state _shared (arr : int array) =
  let i = Int32.to_int (global_idx_x state) in
  arr.(i) <- arr.(i) + 1

let launch_many ~has_barriers =
  let arr = Array.make elements 0 in
  for _ = 1 to launches_per_domain do
    run_threadpool ~has_barriers ~block ~grid increment arr
  done ;
  arr

let check_mode ~has_barriers () =
  let workers =
    List.init domains (fun _ ->
        Domain.spawn (fun () -> launch_many ~has_barriers))
  in
  List.iter
    (fun d ->
      let arr = Domain.join d in
      Array.iteri
        (fun i v ->
          Alcotest.(check int)
            (Printf.sprintf "element %d incremented once per launch" i)
            launches_per_domain
            v)
        arr)
    workers

(* R-67: a raising kernel on a fission queue must surface at [flush] and leave
   the dispatcher alive for the next launch. *)
exception Kernel_failed

let check_queue_error () =
  let ran = Atomic.make 0 in
  let enqueue kernel =
    enqueue_fission ~queue_id:7 ~kernel ~args:[||] ~block ~grid ()
  in
  enqueue (fun ~mode:_ ~block:_ ~grid:_ _ -> raise Kernel_failed) ;
  Alcotest.check_raises "flush re-raises" Kernel_failed (fun () ->
      flush_fission_queue 7) ;
  enqueue (fun ~mode:_ ~block:_ ~grid:_ _ -> Atomic.incr ran) ;
  flush_fission_queue 7 ;
  Alcotest.(check int) "dispatcher survived" 1 (Atomic.get ran)

(* [flush_fission] must flush every queue even when one of them raises, and
   re-raise the first error only afterwards. *)
let check_flush_all_queues () =
  let finished_slow = Atomic.make 0 in
  let slow_queues = [9; 10; 11; 12; 14; 15; 16; 18; 19] in
  let failing_queues = [8; 13; 17; 20] in
  List.iter
    (fun queue_id ->
      enqueue_fission
        ~queue_id
        ~kernel:(fun ~mode:_ ~block:_ ~grid:_ _ ->
          (* Lower ids finish later, so whichever queue is flushed first, some
             slower queue is still running when a failing one is reached. *)
          Unix.sleepf (0.05 *. float_of_int (21 - queue_id)) ;
          Atomic.incr finished_slow)
        ~args:[||]
        ~block
        ~grid
        ())
    slow_queues ;
  List.iter
    (fun queue_id ->
      enqueue_fission
        ~queue_id
        ~kernel:(fun ~mode:_ ~block:_ ~grid:_ _ -> raise Kernel_failed)
        ~args:[||]
        ~block
        ~grid
        ())
    failing_queues ;
  Alcotest.check_raises "flush_fission re-raises" Kernel_failed flush_fission ;
  Alcotest.(check int)
    "all queues flushed"
    (List.length slow_queues)
    (Atomic.get finished_slow)

(* A kernel that launches on the pool it runs on would wait for itself. *)
let check_reentrant ~has_barriers () =
  let nested _state _shared () =
    run_threadpool ~has_barriers ~block ~grid (fun _ _ () -> ()) ()
  in
  match run_threadpool ~has_barriers ~block ~grid nested () with
  | () -> Alcotest.fail "reentrant launch was accepted"
  | exception Invalid_argument _ -> ()

let () =
  let stderr_copy = Unix.dup Unix.stderr in
  let finished = Atomic.make false in
  start_watchdog ~stderr_copy finished ;
  let code =
    try
      Alcotest.run
        ~and_exit:false
        "cpu_runtime_multidomain"
        [
          ( "concurrent launches",
            [
              Alcotest.test_case
                "barrier-free (ParallelPool)"
                `Quick
                (check_mode ~has_barriers:false);
              Alcotest.test_case
                "with barriers (ThreadPool)"
                `Quick
                (check_mode ~has_barriers:true);
              Alcotest.test_case
                "raising fission kernel"
                `Quick
                check_queue_error;
              Alcotest.test_case
                "flush_fission flushes every queue"
                `Quick
                check_flush_all_queues;
              Alcotest.test_case
                "reentrant launch refused (ParallelPool)"
                `Quick
                (check_reentrant ~has_barriers:false);
              Alcotest.test_case
                "reentrant launch refused (ThreadPool)"
                `Quick
                (check_reentrant ~has_barriers:true);
            ] );
        ] ;
      0
    with Alcotest.Test_error -> 1
  in
  Atomic.set finished true ;
  exit code
