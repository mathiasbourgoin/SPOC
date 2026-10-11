(* Regression for R-64: Native launches issued from several domains at once
   share the process-wide pools and used to hang (counters reset while another
   launch's workers were still running, completion signalled to one waiter).

   Each domain launches many small kernels on its own array; every element must
   have been incremented exactly once per launch. A watchdog domain terminates
   the process with a non-zero status if the run does not finish, so a hang
   fails the test instead of blocking CI. *)

open Sarek.Sarek_cpu_runtime

let watchdog_seconds = 30.0

let launches_per_domain = 200

let domains = 4

let block = (8, 1, 1)

let grid = (4, 1, 1)

let elements = 32

let start_watchdog finished =
  ignore
    (Domain.spawn (fun () ->
         let deadline = Unix.gettimeofday () +. watchdog_seconds in
         while (not (Atomic.get finished)) && Unix.gettimeofday () < deadline do
           Unix.sleepf 0.05
         done ;
         if not (Atomic.get finished) then begin
           prerr_endline "FAIL: multi-domain launches did not finish (hang)" ;
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

let () =
  let finished = Atomic.make false in
  start_watchdog finished ;
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
            ] );
        ] ;
      0
    with Alcotest.Test_error -> 1
  in
  Atomic.set finished true ;
  exit code
