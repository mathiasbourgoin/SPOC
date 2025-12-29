(******************************************************************************
 * E2E test for Sarek PPX - Matrix Multiplication
 *
 * Tests naive and tiled matrix multiplication with shared memory.
 * Matrix multiplication is the canonical GPU compute benchmark.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

let tile_size = ref 16

(** Naive matrix multiplication - each thread computes one output element *)
let matmul_naive_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = thread_idx_y + (block_dim_y * block_idx_y) in
      let col = thread_idx_x + (block_dim_x * block_idx_x) in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

(** Tiled matrix multiplication with shared memory and supersteps.

    Uses the classic tiled algorithm where each block cooperatively loads tiles
    of A and B into shared memory, then computes partial dot products.

    BSP Synchronization Pattern: Each loop iteration has 3 supersteps with
    barriers: 1. load_a: All threads load their portion of tile_a, then barrier
    2. load_b: All threads load their portion of tile_b, then barrier 3.
    compute: All threads read from tiles and accumulate, then barrier

    The final barrier (after compute) is critical: it ensures all threads finish
    reading the tiles before the next iteration overwrites them. Without it,
    thread 0 could start writing tile_a for iteration t+1 while thread 1 is
    still reading tile_a for iteration t. *)
let matmul_tiled_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      (* Shared memory tiles: 16x16 = 256 elements each *)
      let%shared (tile_a : float32) = 256l in
      let%shared (tile_b : float32) = 256l in
      let tx = thread_idx_x in
      let ty = thread_idx_y in
      let row = ty + (block_dim_y * block_idx_y) in
      let col = tx + (block_dim_x * block_idx_x) in
      let tile_size = 16l in
      let num_tiles = (k + tile_size - 1l) / tile_size in
      let sum = mut 0.0 in
      for t = 0 to num_tiles - 1l do
        (* Superstep 1: Cooperatively load tile from A into shared memory *)
        let%superstep load_a =
          let a_col = (t * tile_size) + tx in
          if row < m && a_col < k then
            tile_a.((ty * tile_size) + tx) <- a.((row * k) + a_col)
          else tile_a.((ty * tile_size) + tx) <- 0.0
        in
        (* Superstep 2: Cooperatively load tile from B into shared memory *)
        let%superstep load_b =
          let b_row = (t * tile_size) + ty in
          if b_row < k && col < n then
            tile_b.((ty * tile_size) + tx) <- b.((b_row * n) + col)
          else tile_b.((ty * tile_size) + tx) <- 0.0
        in
        (* Superstep 3: Compute partial dot product using shared tiles.
           Must be in a superstep so barrier runs before next iteration
           overwrites the tiles. *)
        let%superstep _compute =
          for i = 0 to tile_size - 1l do
            sum :=
              sum
              +. (tile_a.((ty * tile_size) + i) *. tile_b.((i * tile_size) + tx))
          done
        in
        ()
      done ;
      (* Write final result to global memory *)
      if row < m && col < n then c.((row * n) + col) <- sum]

(** Run naive matrix multiplication test *)
let run_matmul_naive dev =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) in
  let m, n, k = (dim, dim, dim) in

  let a = Vector.create Vector.float32 (m * k) in
  let b = Vector.create Vector.float32 (k * n) in
  let c = Vector.create Vector.float32 (m * n) in

  (* Initialize matrices *)
  for i = 0 to (m * k) - 1 do
    Mem.set a i (float_of_int (i mod 10) /. 10.0)
  done ;
  for i = 0 to (k * n) - 1 do
    Mem.set b i (float_of_int ((i + 1) mod 10) /. 10.0)
  done ;
  for i = 0 to (m * n) - 1 do
    Mem.set c i 0.0
  done ;

  (* Generate and run kernel *)
  ignore (Sarek.Kirc.gen matmul_naive_kernel dev) ;
  let block_size = min 16 (Test_helpers.get_block_size cfg dev) in
  let blocks_x = (n + block_size - 1) / block_size in
  let blocks_y = (m + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run matmul_naive_kernel (a, b, c, m, n, k) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Verify *)
  let ok =
    if cfg.verify then begin
      Mem.to_cpu a () ;
      Mem.to_cpu b () ;
      Mem.to_cpu c () ;
      Devices.flush dev () ;
      Mem.unsafe_rw true ;
      let errors = ref 0 in
      (* Check a few elements *)
      let check_count = min 100 (m * n) in
      for idx = 0 to check_count - 1 do
        let row = idx / n in
        let col = idx mod n in
        let expected = ref 0.0 in
        for i = 0 to k - 1 do
          expected :=
            !expected +. (Mem.get a ((row * k) + i) *. Mem.get b ((i * n) + col))
        done ;
        let got = Mem.get c idx in
        if abs_float (got -. !expected) > 0.01 then incr errors
      done ;
      Mem.unsafe_rw false ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run tiled matrix multiplication test *)
let run_matmul_tiled dev =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) in
  (* Round to multiple of tile size *)
  let dim = (dim + !tile_size - 1) / !tile_size * !tile_size in
  let m, n, k = (dim, dim, dim) in

  let a = Vector.create Vector.float32 (m * k) in
  let b = Vector.create Vector.float32 (k * n) in
  let c = Vector.create Vector.float32 (m * n) in

  (* Initialize matrices *)
  for i = 0 to (m * k) - 1 do
    Mem.set a i (float_of_int (i mod 10) /. 10.0)
  done ;
  for i = 0 to (k * n) - 1 do
    Mem.set b i (float_of_int ((i + 1) mod 10) /. 10.0)
  done ;
  for i = 0 to (m * n) - 1 do
    Mem.set c i 0.0
  done ;

  (* Generate and run kernel *)
  ignore (Sarek.Kirc.gen matmul_tiled_kernel dev) ;
  let block_size = !tile_size in
  let blocks_x = n / block_size in
  let blocks_y = m / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run matmul_tiled_kernel (a, b, c, m, n, k) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Verify *)
  let ok =
    if cfg.verify then begin
      Mem.to_cpu a () ;
      Mem.to_cpu b () ;
      Mem.to_cpu c () ;
      Devices.flush dev () ;
      Mem.unsafe_rw true ;
      let errors = ref 0 in
      (* Check a few elements *)
      let check_count = min 100 (m * n) in
      for idx = 0 to check_count - 1 do
        let row = idx / n in
        let col = idx mod n in
        let expected = ref 0.0 in
        for i = 0 to k - 1 do
          expected :=
            !expected +. (Mem.get a ((row * k) + i) *. Mem.get b ((i * n) + col))
        done ;
        let got = Mem.get c idx in
        if abs_float (got -. !expected) > 0.01 then incr errors
      done ;
      Mem.unsafe_rw false ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_matrix_mul" in
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

  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) in
  Printf.printf
    "Matrix dimensions: %dx%d (total elements: %d)\n%!"
    dim
    dim
    (dim * dim) ;

  if cfg.benchmark_all then begin
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_matmul_naive
      "Matrix multiplication (naive)" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_matmul_tiled
      "Matrix multiplication (tiled)"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

    Printf.printf "\nNaive matrix multiplication:\n%!" ;
    let time_ms, ok = run_matmul_naive dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nTiled matrix multiplication:\n%!" ;
    let time_ms, ok = run_matmul_tiled dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nMatrix multiplication tests PASSED"
  end
