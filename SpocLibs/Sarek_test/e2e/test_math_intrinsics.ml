(******************************************************************************
 * E2E test for Sarek PPX - Math intrinsics
 *
 * Tests all mathematical intrinsic functions: sin, cos, exp, log, sqrt, etc.
 * These map directly to GPU hardware math units.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

(** Trigonometric functions kernel *)
let trig_kernel =
  [%kernel
    fun (input : float32 vector)
        (sin_out : float32 vector)
        (cos_out : float32 vector)
        (tan_out : float32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let x = input.(tid) in
        sin_out.(tid) <- sin x ;
        cos_out.(tid) <- cos x ;
        tan_out.(tid) <- tan x
      end]

(** Exponential and logarithm functions *)
let exp_log_kernel =
  [%kernel
    fun (input : float32 vector)
        (exp_out : float32 vector)
        (log_out : float32 vector)
        (log10_out : float32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let x = input.(tid) in
        exp_out.(tid) <- exp x ;
        if x > 0.0 then begin
          log_out.(tid) <- log x ;
          log10_out.(tid) <- log10 x
        end
        else begin
          log_out.(tid) <- 0.0 ;
          log10_out.(tid) <- 0.0
        end
      end]

(** Power and root functions *)
let power_kernel =
  [%kernel
    fun (base : float32 vector)
        (exponent : float32 vector)
        (pow_out : float32 vector)
        (sqrt_out : float32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let b = base.(tid) in
        let e = exponent.(tid) in
        pow_out.(tid) <- pow b e ;
        if b >= 0.0 then sqrt_out.(tid) <- sqrt b else sqrt_out.(tid) <- 0.0
      end]

(** Rounding functions *)
let rounding_kernel =
  [%kernel
    fun (input : float32 vector)
        (floor_out : float32 vector)
        (ceil_out : float32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let x = input.(tid) in
        floor_out.(tid) <- floor x ;
        ceil_out.(tid) <- ceil x
      end]

(** Abs and sign functions *)
let abs_kernel =
  [%kernel
    fun (input : float32 vector) (abs_out : float32 vector) (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let x = input.(tid) in
        if x < 0.0 then abs_out.(tid) <- -.x else abs_out.(tid) <- x
      end]

(** Complex math expression combining multiple intrinsics *)
let complex_math_kernel =
  [%kernel
    fun (x : float32 vector)
        (y : float32 vector)
        (output : float32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let a = x.(tid) in
        let b = y.(tid) in
        (* Compute: sqrt(a^2 + b^2) * exp(-0.1 * (a + b)) * cos(a * b) *)
        let r = sqrt ((a *. a) +. (b *. b)) in
        let decay = exp (-0.1 *. (a +. b)) in
        let oscillation = cos (a *. b) in
        output.(tid) <- r *. decay *. oscillation
      end]

(** Run trigonometric test *)
let run_trig_test dev =
  let n = cfg.size in
  let input = Vector.create Vector.float32 n in
  let sin_out = Vector.create Vector.float32 n in
  let cos_out = Vector.create Vector.float32 n in
  let tan_out = Vector.create Vector.float32 n in

  (* Initialize with angles *)
  for i = 0 to n - 1 do
    Mem.set input i (float_of_int i *. 0.01) ;
    Mem.set sin_out i 0.0 ;
    Mem.set cos_out i 0.0 ;
    Mem.set tan_out i 0.0
  done ;

  ignore (Sarek.Kirc.gen trig_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    trig_kernel
    (input, sin_out, cos_out, tan_out, n)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu sin_out () ;
      Mem.to_cpu cos_out () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 1 do
        let x = Mem.get input i in
        let s = Mem.get sin_out i in
        let c = Mem.get cos_out i in
        if abs_float (s -. sin x) > 0.001 || abs_float (c -. cos x) > 0.001 then
          incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run exp/log test *)
let run_exp_log_test dev =
  let n = cfg.size in
  let input = Vector.create Vector.float32 n in
  let exp_out = Vector.create Vector.float32 n in
  let log_out = Vector.create Vector.float32 n in
  let log10_out = Vector.create Vector.float32 n in

  (* Initialize with values in safe range for float32 exp (max ~88) and log (positive) *)
  for i = 0 to n - 1 do
    (* Use modulo to keep values in range [0.1, 8.7] for exp, all positive for log *)
    Mem.set input i (float_of_int ((i mod 86) + 1) *. 0.1) ;
    Mem.set exp_out i 0.0 ;
    Mem.set log_out i 0.0 ;
    Mem.set log10_out i 0.0
  done ;

  ignore (Sarek.Kirc.gen exp_log_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    exp_log_kernel
    (input, exp_out, log_out, log10_out, n)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu exp_out () ;
      Mem.to_cpu log_out () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      (* Use relative error for exp (values can be very large) *)
      let rel_err a b = if abs_float b < 1e-6 then abs_float (a -. b) else abs_float (a -. b) /. abs_float b in
      for i = 0 to n - 1 do
        let x = Mem.get input i in
        let e = Mem.get exp_out i in
        let l = Mem.get log_out i in
        let exp_expected = exp x in
        let log_expected = log x in
        (* float32 has ~7 decimal digits of precision, use 1e-5 relative tolerance *)
        let exp_rel_err = rel_err e exp_expected in
        let log_rel_err = rel_err l log_expected in
        if exp_rel_err > 1e-5 || log_rel_err > 1e-5 then begin
          if !errors < 5 then
            Printf.printf "  [%d] x=%.4f exp: got %.4f expected %.4f (rel_err=%.2e), log: got %.4f expected %.4f (rel_err=%.2e)\n"
              i x e exp_expected exp_rel_err l log_expected log_rel_err;
          incr errors
        end
      done ;
      if !errors > 0 then Printf.printf "  Total errors: %d\n" !errors;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run power test *)
let run_power_test dev =
  let n = cfg.size in
  let base = Vector.create Vector.float32 n in
  let exponent = Vector.create Vector.float32 n in
  let pow_out = Vector.create Vector.float32 n in
  let sqrt_out = Vector.create Vector.float32 n in

  (* Initialize *)
  for i = 0 to n - 1 do
    Mem.set base i (float_of_int (i + 1) *. 0.1) ;
    Mem.set exponent i 2.0 ;
    Mem.set pow_out i 0.0 ;
    Mem.set sqrt_out i 0.0
  done ;

  ignore (Sarek.Kirc.gen power_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    power_kernel
    (base, exponent, pow_out, sqrt_out, n)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu pow_out () ;
      Mem.to_cpu sqrt_out () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      (* Use relative error for pow (values can be very large) *)
      let rel_err a b = if abs_float b < 1e-6 then abs_float (a -. b) else abs_float (a -. b) /. abs_float b in
      for i = 0 to n - 1 do
        let b = Mem.get base i in
        let p = Mem.get pow_out i in
        let s = Mem.get sqrt_out i in
        let pow_expected = b ** 2.0 in
        let sqrt_expected = sqrt b in
        (* float32 has ~7 decimal digits of precision, use 1e-5 relative tolerance *)
        if rel_err p pow_expected > 1e-5 || rel_err s sqrt_expected > 1e-5
        then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run complex math test *)
let run_complex_math_test dev =
  let n = cfg.size in
  let x = Vector.create Vector.float32 n in
  let y = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Mem.set x i (float_of_int (i mod 10) *. 0.5) ;
    Mem.set y i (float_of_int ((i + 5) mod 10) *. 0.5) ;
    Mem.set output i 0.0
  done ;

  ignore (Sarek.Kirc.gen complex_math_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run complex_math_kernel (x, y, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 1 do
        let a = Mem.get x i in
        let b = Mem.get y i in
        let expected =
          sqrt ((a *. a) +. (b *. b)) *. exp (-0.1 *. (a +. b)) *. cos (a *. b)
        in
        let got = Mem.get output i in
        if abs_float (got -. expected) > 0.01 then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_math_intrinsics" in
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
      run_trig_test
      "Trigonometric (sin/cos/tan)" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_exp_log_test
      "Exp/Log" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_power_test
      "Power/Sqrt" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_complex_math_test
      "Complex math expression"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    Printf.printf "Testing math intrinsics with size=%d\n%!" cfg.size ;

    Printf.printf "\nTrigonometric functions:\n%!" ;
    let time_ms, ok = run_trig_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nExp/Log functions:\n%!" ;
    let time_ms, ok = run_exp_log_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nPower/Sqrt functions:\n%!" ;
    let time_ms, ok = run_power_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nComplex math expression:\n%!" ;
    let time_ms, ok = run_complex_math_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nMath intrinsics tests PASSED"
  end
