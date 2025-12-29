(******************************************************************************
 * E2E test for Sarek PPX - Math intrinsics
 *
 * Tests all mathematical intrinsic functions: sin, cos, exp, log, sqrt, etc.
 * These map directly to GPU hardware math units.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

(* ========== Pure OCaml baselines ========== *)

let ocaml_trig input sin_out cos_out tan_out n =
  for i = 0 to n - 1 do
    let x = input.(i) in
    sin_out.(i) <- sin x ;
    cos_out.(i) <- cos x ;
    tan_out.(i) <- tan x
  done

let ocaml_exp_log input exp_out log_out log10_out n =
  for i = 0 to n - 1 do
    let x = input.(i) in
    exp_out.(i) <- exp x ;
    if x > 0.0 then begin
      log_out.(i) <- log x ;
      log10_out.(i) <- log10 x
    end
    else begin
      log_out.(i) <- 0.0 ;
      log10_out.(i) <- 0.0
    end
  done

let ocaml_power base exponent pow_out sqrt_out n =
  for i = 0 to n - 1 do
    let b = base.(i) in
    let e = exponent.(i) in
    pow_out.(i) <- b ** e ;
    if b >= 0.0 then sqrt_out.(i) <- sqrt b else sqrt_out.(i) <- 0.0
  done

let ocaml_complex_math x y output n =
  for i = 0 to n - 1 do
    let a = x.(i) in
    let b = y.(i) in
    let r = sqrt ((a *. a) +. (b *. b)) in
    let decay = exp (-0.1 *. (a +. b)) in
    let oscillation = cos (a *. b) in
    output.(i) <- r *. decay *. oscillation
  done

(* ========== Shared test data ========== *)

let input_trig = ref [||]
let expected_sin = ref [||]
let expected_cos = ref [||]
let expected_tan = ref [||]

let input_exp = ref [||]
let expected_exp = ref [||]
let expected_log = ref [||]
let expected_log10 = ref [||]

let input_base = ref [||]
let input_exponent = ref [||]
let expected_pow = ref [||]
let expected_sqrt = ref [||]

let input_x = ref [||]
let input_y = ref [||]
let expected_complex = ref [||]

let init_trig_data () =
  let n = cfg.size in
  let inp = Array.init n (fun i -> float_of_int i *. 0.01) in
  let s = Array.make n 0.0 in
  let c = Array.make n 0.0 in
  let t = Array.make n 0.0 in
  input_trig := inp ;
  expected_sin := s ;
  expected_cos := c ;
  expected_tan := t ;
  let t0 = Unix.gettimeofday () in
  ocaml_trig inp s c t n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_exp_log_data () =
  let n = cfg.size in
  let inp = Array.init n (fun i -> float_of_int ((i mod 86) + 1) *. 0.1) in
  let e = Array.make n 0.0 in
  let l = Array.make n 0.0 in
  let l10 = Array.make n 0.0 in
  input_exp := inp ;
  expected_exp := e ;
  expected_log := l ;
  expected_log10 := l10 ;
  let t0 = Unix.gettimeofday () in
  ocaml_exp_log inp e l l10 n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_power_data () =
  let n = cfg.size in
  let b = Array.init n (fun i -> float_of_int (i + 1) *. 0.1) in
  let e = Array.make n 2.0 in
  let p = Array.make n 0.0 in
  let s = Array.make n 0.0 in
  input_base := b ;
  input_exponent := e ;
  expected_pow := p ;
  expected_sqrt := s ;
  let t0 = Unix.gettimeofday () in
  ocaml_power b e p s n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_complex_data () =
  let n = cfg.size in
  let x = Array.init n (fun i -> float_of_int (i mod 10) *. 0.5) in
  let y = Array.init n (fun i -> float_of_int ((i + 5) mod 10) *. 0.5) in
  let o = Array.make n 0.0 in
  input_x := x ;
  input_y := y ;
  expected_complex := o ;
  let t0 = Unix.gettimeofday () in
  ocaml_complex_math x y o n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Sarek kernels ========== *)

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

(* ========== Device test runners ========== *)

(** Run trigonometric test *)
let run_trig_test dev =
  let n = cfg.size in
  let inp = !input_trig in
  let exp_sin = !expected_sin in
  let exp_cos = !expected_cos in

  let input = Vector.create Vector.float32 n in
  let sin_out = Vector.create Vector.float32 n in
  let cos_out = Vector.create Vector.float32 n in
  let tan_out = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Mem.set input i inp.(i) ;
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
        let s = Mem.get sin_out i in
        let c = Mem.get cos_out i in
        if abs_float (s -. exp_sin.(i)) > 0.001 || abs_float (c -. exp_cos.(i)) > 0.001
        then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run exp/log test *)
let run_exp_log_test dev =
  let n = cfg.size in
  let inp = !input_exp in
  let exp_expected = !expected_exp in
  let log_expected = !expected_log in

  let input = Vector.create Vector.float32 n in
  let exp_out = Vector.create Vector.float32 n in
  let log_out = Vector.create Vector.float32 n in
  let log10_out = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Mem.set input i inp.(i) ;
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
      let rel_err a b =
        if abs_float b < 1e-6 then abs_float (a -. b)
        else abs_float (a -. b) /. abs_float b
      in
      for i = 0 to n - 1 do
        let e = Mem.get exp_out i in
        let l = Mem.get log_out i in
        if rel_err e exp_expected.(i) > 1e-5 || rel_err l log_expected.(i) > 1e-5
        then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run power test *)
let run_power_test dev =
  let n = cfg.size in
  let inp_base = !input_base in
  let inp_exp = !input_exponent in
  let pow_expected = !expected_pow in
  let sqrt_expected = !expected_sqrt in

  let base = Vector.create Vector.float32 n in
  let exponent = Vector.create Vector.float32 n in
  let pow_out = Vector.create Vector.float32 n in
  let sqrt_out = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Mem.set base i inp_base.(i) ;
    Mem.set exponent i inp_exp.(i) ;
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
      let rel_err a b =
        if abs_float b < 1e-6 then abs_float (a -. b)
        else abs_float (a -. b) /. abs_float b
      in
      for i = 0 to n - 1 do
        let p = Mem.get pow_out i in
        let s = Mem.get sqrt_out i in
        if rel_err p pow_expected.(i) > 1e-5 || rel_err s sqrt_expected.(i) > 1e-5
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
  let inp_x = !input_x in
  let inp_y = !input_y in
  let exp_out = !expected_complex in

  let x = Vector.create Vector.float32 n in
  let y = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Mem.set x i inp_x.(i) ;
    Mem.set y i inp_y.(i) ;
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
        let got = Mem.get output i in
        if abs_float (got -. exp_out.(i)) > 0.01 then incr errors
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
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_trig_data
      run_trig_test
      "Trigonometric (sin/cos/tan)" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_exp_log_data
      run_exp_log_test
      "Exp/Log" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_power_data
      run_power_test
      "Power/Sqrt" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_complex_data
      run_complex_math_test
      "Complex math expression"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    Printf.printf "Testing math intrinsics with size=%d\n%!" cfg.size ;

    let baseline_ms, _ = init_trig_data () in
    Printf.printf "\nOCaml baseline (trig): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nTrigonometric functions:\n%!" ;
    let time_ms, ok = run_trig_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    let baseline_ms, _ = init_exp_log_data () in
    Printf.printf "\nOCaml baseline (exp/log): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nExp/Log functions:\n%!" ;
    let time_ms, ok = run_exp_log_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    let baseline_ms, _ = init_power_data () in
    Printf.printf "\nOCaml baseline (power): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nPower/Sqrt functions:\n%!" ;
    let time_ms, ok = run_power_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    let baseline_ms, _ = init_complex_data () in
    Printf.printf "\nOCaml baseline (complex): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nComplex math expression:\n%!" ;
    let time_ms, ok = run_complex_math_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nMath intrinsics tests PASSED"
  end
