(******************************************************************************
 * E2E test for Sarek PPX - Parallel Reduction
 *
 * Tests tree-based parallel reduction with shared memory and barriers.
 * Reduction is a fundamental parallel primitive for computing sums, min, max.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

(* ========== Pure OCaml baselines ========== *)

let ocaml_sum arr n =
  let sum = ref 0.0 in
  for i = 0 to n - 1 do
    sum := !sum +. arr.(i)
  done ;
  !sum

let ocaml_max arr n =
  let m = ref arr.(0) in
  for i = 1 to n - 1 do
    if arr.(i) > !m then m := arr.(i)
  done ;
  !m

let ocaml_dot a b n =
  let sum = ref 0.0 in
  for i = 0 to n - 1 do
    sum := !sum +. (a.(i) *. b.(i))
  done ;
  !sum

(* ========== Shared test data ========== *)

let input_sum = ref [||]
let expected_sum = ref 0.0
let input_max = ref [||]
let expected_max = ref 0.0
let input_a = ref [||]
let input_b = ref [||]
let expected_dot = ref 0.0

let init_sum_data () =
  let n = cfg.size in
  let arr = Array.make n 1.0 in
  input_sum := arr ;
  let t0 = Unix.gettimeofday () in
  expected_sum := ocaml_sum arr n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_max_data () =
  let n = cfg.size in
  let arr = Array.init n (fun i -> float_of_int i) in
  input_max := arr ;
  let t0 = Unix.gettimeofday () in
  expected_max := ocaml_max arr n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_dot_data () =
  let n = cfg.size in
  let a = Array.init n (fun i -> float_of_int i) in
  let b = Array.make n 1.0 in
  input_a := a ;
  input_b := b ;
  let t0 = Unix.gettimeofday () in
  expected_dot := ocaml_dot a b n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Sarek kernels ========== *)

let reduce_sum_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid) else sdata.(tid) <- 0.0
      in
      let%superstep reduce128 =
        if tid < 128l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 128l)
      in
      let%superstep reduce64 =
        if tid < 64l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 64l)
      in
      let%superstep reduce32 =
        if tid < 32l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 32l)
      in
      let%superstep reduce16 =
        if tid < 16l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 16l)
      in
      let%superstep reduce8 =
        if tid < 8l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 8l)
      in
      let%superstep reduce4 =
        if tid < 4l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 4l)
      in
      let%superstep reduce2 =
        if tid < 2l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 2l)
      in
      let%superstep reduce1 =
        if tid < 1l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 1l)
      in
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]

let reduce_max_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid)
        else sdata.(tid) <- -1000000.0
      in
      let%superstep reduce128 =
        if tid < 128l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 128l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce64 =
        if tid < 64l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 64l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce32 =
        if tid < 32l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 32l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce16 =
        if tid < 16l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 16l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce8 =
        if tid < 8l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 8l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce4 =
        if tid < 4l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 4l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce2 =
        if tid < 2l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 2l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce1 =
        if tid < 1l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 1l) in
          if b > a then sdata.(tid) <- b
        end
      in
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]

let dot_product_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (output : float32 vector)
        (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- a.(gid) *. b.(gid)
        else sdata.(tid) <- 0.0
      in
      let%superstep reduce128 =
        if tid < 128l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 128l)
      in
      let%superstep reduce64 =
        if tid < 64l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 64l)
      in
      let%superstep reduce32 =
        if tid < 32l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 32l)
      in
      let%superstep reduce16 =
        if tid < 16l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 16l)
      in
      let%superstep reduce8 =
        if tid < 8l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 8l)
      in
      let%superstep reduce4 =
        if tid < 4l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 4l)
      in
      let%superstep reduce2 =
        if tid < 2l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 2l)
      in
      let%superstep reduce1 =
        if tid < 1l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 1l)
      in
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]

(* ========== Device test runners ========== *)

let run_reduce_sum dev =
  let n = cfg.size in
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = (n + block_size - 1) / block_size in
  let inp = !input_sum in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 num_blocks in

  for i = 0 to n - 1 do
    Mem.set input i inp.(i)
  done ;
  for i = 0 to num_blocks - 1 do
    Mem.set output i 0.0
  done ;

  ignore (Sarek.Kirc.gen reduce_sum_kernel dev) ;
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run reduce_sum_kernel (input, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let total = ref 0.0 in
      for i = 0 to num_blocks - 1 do
        total := !total +. Mem.get output i
      done ;
      abs_float (!total -. !expected_sum) < 0.1
    end
    else true
  in
  (time_ms, ok)

let run_reduce_max dev =
  let n = cfg.size in
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = (n + block_size - 1) / block_size in
  let inp = !input_max in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 num_blocks in

  for i = 0 to n - 1 do
    Mem.set input i inp.(i)
  done ;
  for i = 0 to num_blocks - 1 do
    Mem.set output i (-1000000.0)
  done ;

  ignore (Sarek.Kirc.gen reduce_max_kernel dev) ;
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run reduce_max_kernel (input, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let max_val = ref (-1000000.0) in
      for i = 0 to num_blocks - 1 do
        let v = Mem.get output i in
        if v > !max_val then max_val := v
      done ;
      abs_float (!max_val -. !expected_max) < 0.1
    end
    else true
  in
  (time_ms, ok)

let run_dot_product dev =
  let n = cfg.size in
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = (n + block_size - 1) / block_size in
  let inp_a = !input_a in
  let inp_b = !input_b in

  let a = Vector.create Vector.float32 n in
  let b = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 num_blocks in

  for i = 0 to n - 1 do
    Mem.set a i inp_a.(i) ;
    Mem.set b i inp_b.(i)
  done ;
  for i = 0 to num_blocks - 1 do
    Mem.set output i 0.0
  done ;

  ignore (Sarek.Kirc.gen dot_product_kernel dev) ;
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run dot_product_kernel (a, b, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let total = ref 0.0 in
      for i = 0 to num_blocks - 1 do
        total := !total +. Mem.get output i
      done ;
      abs_float (!total -. !expected_dot) < float_of_int n *. 0.01
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_reduce" in
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
      ~baseline:init_sum_data
      run_reduce_sum
      "Reduction (sum)" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_max_data
      run_reduce_max
      "Reduction (max)" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_dot_data
      run_dot_product
      "Dot product"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    Printf.printf "Testing reduction operations with size=%d\n%!" cfg.size ;

    let baseline_ms, _ = init_sum_data () in
    Printf.printf "\nOCaml baseline (sum): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nSum reduction:\n%!" ;
    let time_ms, ok1 = run_reduce_sum dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok1 then "PASSED" else "FAILED") ;

    let baseline_ms, _ = init_max_data () in
    Printf.printf "\nOCaml baseline (max): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nMax reduction:\n%!" ;
    let time_ms, ok2 = run_reduce_max dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok2 then "PASSED" else "FAILED") ;

    let baseline_ms, _ = init_dot_data () in
    Printf.printf "\nOCaml baseline (dot): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nDot product:\n%!" ;
    let time_ms, ok3 = run_dot_product dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok3 then "PASSED" else "FAILED") ;

    if ok1 && ok2 && ok3 then print_endline "\nReduction tests PASSED"
    else begin
      print_endline "\nReduction tests FAILED" ;
      exit 1
    end
  end
