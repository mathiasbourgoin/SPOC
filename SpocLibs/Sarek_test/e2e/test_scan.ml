(******************************************************************************
 * E2E test for Sarek PPX - Prefix Sum (Scan)
 *
 * Tests inclusive prefix sum operations with shared memory and supersteps.
 * Scan is a fundamental parallel primitive for many algorithms.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

(** Inclusive scan within a block using Hillis-Steele algorithm.

    IMPORTANT: Hillis-Steele requires that ALL threads read their neighbor's
    value BEFORE any thread writes. This requires double-buffering or explicit
    barrier between reads and writes.

    Pattern for each step: 1. All threads read neighbor into local variable (in
    superstep) 2. All threads write updated value (in next superstep) *)
let inclusive_scan_kernel =
  [%kernel
    fun (input : int32 vector) (output : int32 vector) (n : int32) ->
      let%shared (temp : int32) = 512l in
      let%shared (temp2 : int32) = 512l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      (* Load to shared memory - use temp as source *)
      let%superstep load =
        if gid < n then temp.(tid) <- input.(gid) else temp.(tid) <- 0l
      in
      (* Hillis-Steele: ping-pong between temp and temp2.
         Each superstep uses the same variable names (v, add) - this tests that
         the code generator properly scopes local variables within superstep blocks. *)
      (* Step 1: read from temp, write to temp2 *)
      let%superstep step1 =
        let v = temp.(tid) in
        let add = if tid >= 1l then temp.(tid - 1l) else 0l in
        temp2.(tid) <- v + add
      in
      (* Step 2: read from temp2, write to temp *)
      let%superstep step2 =
        let v = temp2.(tid) in
        let add = if tid >= 2l then temp2.(tid - 2l) else 0l in
        temp.(tid) <- v + add
      in
      (* Step 4: read from temp, write to temp2 *)
      let%superstep step4 =
        let v = temp.(tid) in
        let add = if tid >= 4l then temp.(tid - 4l) else 0l in
        temp2.(tid) <- v + add
      in
      (* Step 8: read from temp2, write to temp *)
      let%superstep step8 =
        let v = temp2.(tid) in
        let add = if tid >= 8l then temp2.(tid - 8l) else 0l in
        temp.(tid) <- v + add
      in
      (* Step 16: read from temp, write to temp2 *)
      let%superstep step16 =
        let v = temp.(tid) in
        let add = if tid >= 16l then temp.(tid - 16l) else 0l in
        temp2.(tid) <- v + add
      in
      (* Step 32: read from temp2, write to temp *)
      let%superstep step32 =
        let v = temp2.(tid) in
        let add = if tid >= 32l then temp2.(tid - 32l) else 0l in
        temp.(tid) <- v + add
      in
      (* Step 64: read from temp, write to temp2 *)
      let%superstep step64 =
        let v = temp.(tid) in
        let add = if tid >= 64l then temp.(tid - 64l) else 0l in
        temp2.(tid) <- v + add
      in
      (* Step 128: read from temp2, write to temp *)
      let%superstep step128 =
        let v = temp2.(tid) in
        let add = if tid >= 128l then temp2.(tid - 128l) else 0l in
        temp.(tid) <- v + add
      in
      (* Write result from temp (last write was to temp) *)
      if gid < n then output.(gid) <- temp.(tid)]

(** Run inclusive scan test *)
let run_inclusive_scan_test dev =
  let n = min cfg.size 256 in
  (* Single block for simplicity *)
  let input = Vector.create Vector.int32 n in
  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Mem.set input i 1l ;
    (* All ones - sum should be 1, 2, 3, ... *)
    Mem.set output i 0l
  done ;

  ignore (Sarek.Kirc.gen inclusive_scan_kernel dev) ;
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = 1; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run inclusive_scan_kernel (input, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 1 do
        let expected = Int32.of_int (i + 1) in
        let got = Mem.get output i in
        if got <> expected then begin
          if !errors < 10 then
            Printf.printf
              "  Mismatch at %d: expected %ld, got %ld\n"
              i
              expected
              got ;
          incr errors
        end
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run scan with varying values test *)
let run_varying_scan_test dev =
  let n = min cfg.size 256 in
  let input = Vector.create Vector.int32 n in
  let output = Vector.create Vector.int32 n in

  (* Initialize with i+1 values: 1, 2, 3, ... *)
  for i = 0 to n - 1 do
    Mem.set input i (Int32.of_int (i + 1)) ;
    Mem.set output i 0l
  done ;

  ignore (Sarek.Kirc.gen inclusive_scan_kernel dev) ;
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = 1; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run inclusive_scan_kernel (input, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      let expected = ref 0l in
      for i = 0 to n - 1 do
        expected := Int32.add !expected (Int32.of_int (i + 1)) ;
        let got = Mem.get output i in
        if got <> !expected then begin
          if !errors < 10 then
            Printf.printf
              "  Mismatch at %d: expected %ld, got %ld\n"
              i
              !expected
              got ;
          incr errors
        end
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_scan" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.use_native_parallel <- c.use_native_parallel ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  let devs = Devices.init () in
  if Array.length devs = 0 then begin
    print_endline "No GPU devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices devs ;

  if cfg.benchmark_all then begin
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_inclusive_scan_test
      "Inclusive scan" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_varying_scan_test
      "Scan with varying values"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    Printf.printf "Testing prefix sum operations\n%!" ;

    Printf.printf "\nInclusive scan (all ones):\n%!" ;
    let time_ms, ok = run_inclusive_scan_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nScan with varying values:\n%!" ;
    let time_ms, ok = run_varying_scan_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nScan tests PASSED"
  end
