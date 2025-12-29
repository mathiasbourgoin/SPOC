(******************************************************************************
 * E2E test for Sarek PPX - Parallel Reduction
 *
 * Tests tree-based parallel reduction with shared memory and barriers.
 * Reduction is a fundamental parallel primitive for computing sums, min, max.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

(** Block-level sum reduction kernel with supersteps *)
let reduce_sum_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      (* Load to shared memory *)
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid) else sdata.(tid) <- 0.0
      in
      (* Tree-based reduction - stride 128 *)
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
      (* Write result *)
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]

(** Block-level max reduction with supersteps *)
let reduce_max_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      (* Load to shared memory *)
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid)
        else sdata.(tid) <- -1000000.0
      in
      (* Tree-based reduction *)
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
      (* Write result *)
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]

(** Dot product with shared memory reduction *)
let dot_product_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (output : float32 vector)
        (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      (* Load and multiply *)
      let%superstep load =
        if gid < n then sdata.(tid) <- a.(gid) *. b.(gid)
        else sdata.(tid) <- 0.0
      in
      (* Tree-based reduction *)
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
      (* Write result *)
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]

(** Run sum reduction test *)
let run_reduce_sum dev =
  let n = cfg.size in
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = (n + block_size - 1) / block_size in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 num_blocks in

  (* Initialize with values 1..n *)
  for i = 0 to n - 1 do
    Mem.set input i 1.0
  done ;
  for i = 0 to num_blocks - 1 do
    Mem.set output i 0.0
  done ;

  (* Generate and run kernel *)
  ignore (Sarek.Kirc.gen reduce_sum_kernel dev) ;
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run reduce_sum_kernel (input, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Verify - sum of partial results *)
  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let total = ref 0.0 in
      for i = 0 to num_blocks - 1 do
        total := !total +. Mem.get output i
      done ;
      let expected = float_of_int n in
      abs_float (!total -. expected) < 0.1
    end
    else true
  in
  (time_ms, ok)

(** Run max reduction test *)
let run_reduce_max dev =
  let n = cfg.size in
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = (n + block_size - 1) / block_size in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 num_blocks in

  (* Initialize with increasing values, max at n-1 *)
  for i = 0 to n - 1 do
    Mem.set input i (float_of_int i)
  done ;
  for i = 0 to num_blocks - 1 do
    Mem.set output i (-1000000.0)
  done ;

  (* Generate and run kernel *)
  ignore (Sarek.Kirc.gen reduce_max_kernel dev) ;
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run reduce_max_kernel (input, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Verify - max of partial results *)
  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let max_val = ref (-1000000.0) in
      for i = 0 to num_blocks - 1 do
        let v = Mem.get output i in
        if v > !max_val then max_val := v
      done ;
      let expected = float_of_int (n - 1) in
      abs_float (!max_val -. expected) < 0.1
    end
    else true
  in
  (time_ms, ok)

(** Run dot product test *)
let run_dot_product dev =
  let n = cfg.size in
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = (n + block_size - 1) / block_size in

  let a = Vector.create Vector.float32 n in
  let b = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 num_blocks in

  (* Initialize: a[i] = i, b[i] = 1 => dot = sum of 0..n-1 *)
  for i = 0 to n - 1 do
    Mem.set a i (float_of_int i) ;
    Mem.set b i 1.0
  done ;
  for i = 0 to num_blocks - 1 do
    Mem.set output i 0.0
  done ;

  (* Generate and run kernel *)
  ignore (Sarek.Kirc.gen dot_product_kernel dev) ;
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run dot_product_kernel (a, b, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Verify - sum of partial results *)
  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let total = ref 0.0 in
      for i = 0 to num_blocks - 1 do
        total := !total +. Mem.get output i
      done ;
      let expected = float_of_int (n * (n - 1) / 2) in
      abs_float (!total -. expected) < float_of_int n *. 0.01
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_reduce" in
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
      run_reduce_sum
      "Reduction (sum)" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_reduce_max
      "Reduction (max)" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_dot_product
      "Dot product"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    Printf.printf "Testing reduction operations with size=%d\n%!" cfg.size ;

    Printf.printf "\nSum reduction:\n%!" ;
    let time_ms, ok1 = run_reduce_sum dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok1 then "PASSED" else "FAILED") ;

    Printf.printf "\nMax reduction:\n%!" ;
    let time_ms, ok2 = run_reduce_max dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok2 then "PASSED" else "FAILED") ;

    Printf.printf "\nDot product:\n%!" ;
    let time_ms, ok3 = run_dot_product dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok3 then "PASSED" else "FAILED") ;

    if ok1 && ok2 && ok3 then print_endline "\nReduction tests PASSED"
    else begin
      print_endline "\nReduction tests FAILED" ;
      exit 1
    end
  end
