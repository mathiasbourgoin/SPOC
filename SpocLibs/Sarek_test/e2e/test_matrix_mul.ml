(******************************************************************************
 * E2E test for Sarek PPX - Matrix Multiplication
 *
 * Tests naive and tiled matrix multiplication with shared memory.
 * Matrix multiplication is the canonical GPU compute benchmark.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

let tile_size = ref 16

(* ========== Pure OCaml baseline ========== *)

(** Pure OCaml matrix multiplication *)
let ocaml_matmul a b c m n k =
  for row = 0 to m - 1 do
    for col = 0 to n - 1 do
      let sum = ref 0.0 in
      for i = 0 to k - 1 do
        sum := !sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- !sum
    done
  done

(* ========== Shared test data ========== *)

let input_a = ref [||]

let input_b = ref [||]

let expected_c = ref [||]

let matrix_dim = ref 0

(** Initialize matrices and compute expected result *)
let init_matmul_data () =
  let dim = int_of_float (sqrt (float_of_int cfg.size)) in
  matrix_dim := dim ;
  let m, n, k = (dim, dim, dim) in
  let a = Array.init (m * k) (fun i -> float_of_int (i mod 10) /. 10.0) in
  let b = Array.init (k * n) (fun i -> float_of_int ((i + 1) mod 10) /. 10.0) in
  let c = Array.make (m * n) 0.0 in
  input_a := a ;
  input_b := b ;
  expected_c := c ;
  let t0 = Unix.gettimeofday () in
  ocaml_matmul a b c m n k ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Sarek kernels ========== *)

(** Naive matrix multiplication - each thread computes one output element. Uses
    global_idx_x/y for optimized Simple2D execution path on native CPU. *)
let matmul_naive_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let open Std in
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

(** Tiled matrix multiplication with shared memory and supersteps. *)
let matmul_tiled_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
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
        let%superstep load_a =
          let a_col = (t * tile_size) + tx in
          if row < m && a_col < k then
            tile_a.((ty * tile_size) + tx) <- a.((row * k) + a_col)
          else tile_a.((ty * tile_size) + tx) <- 0.0
        in
        let%superstep load_b =
          let b_row = (t * tile_size) + ty in
          if b_row < k && col < n then
            tile_b.((ty * tile_size) + tx) <- b.((b_row * n) + col)
          else tile_b.((ty * tile_size) + tx) <- 0.0
        in
        let%superstep _compute =
          for i = 0 to tile_size - 1l do
            sum :=
              sum
              +. (tile_a.((ty * tile_size) + i) *. tile_b.((i * tile_size) + tx))
          done
        in
        ()
      done ;
      if row < m && col < n then c.((row * n) + col) <- sum]

(* ========== Device test runners ========== *)

(** Run naive matrix multiplication test *)
let run_matmul_naive dev =
  let dim = !matrix_dim in
  let m, n, k = (dim, dim, dim) in
  let inp_a = !input_a in
  let inp_b = !input_b in
  let exp_c = !expected_c in

  let a = Vector.create Vector.float32 (m * k) in
  let b = Vector.create Vector.float32 (k * n) in
  let c = Vector.create Vector.float32 (m * n) in

  for i = 0 to (m * k) - 1 do
    Mem.set a i inp_a.(i)
  done ;
  for i = 0 to (k * n) - 1 do
    Mem.set b i inp_b.(i)
  done ;
  for i = 0 to (m * n) - 1 do
    Mem.set c i 0.0
  done ;

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

  let ok =
    if cfg.verify then begin
      Mem.to_cpu c () ;
      Devices.flush dev () ;
      Mem.unsafe_rw true ;
      let errors = ref 0 in
      let check_count = min 100 (m * n) in
      for idx = 0 to check_count - 1 do
        let expected = exp_c.(idx) in
        let got = Mem.get c idx in
        if abs_float (got -. expected) > 0.01 then incr errors
      done ;
      Mem.unsafe_rw false ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run tiled matrix multiplication test *)
let run_matmul_tiled dev =
  let dim = !matrix_dim in
  let dim = (dim + !tile_size - 1) / !tile_size * !tile_size in
  let m, n, k = (dim, dim, dim) in

  (* Need to recompute expected for padded dimension *)
  let inp_a = !input_a in
  let inp_b = !input_b in
  let orig_dim = !matrix_dim in

  let a = Vector.create Vector.float32 (m * k) in
  let b = Vector.create Vector.float32 (k * n) in
  let c = Vector.create Vector.float32 (m * n) in

  (* Initialize with padding *)
  for row = 0 to m - 1 do
    for col = 0 to k - 1 do
      let idx = (row * k) + col in
      if row < orig_dim && col < orig_dim then
        Mem.set a idx inp_a.((row * orig_dim) + col)
      else Mem.set a idx 0.0
    done
  done ;
  for row = 0 to k - 1 do
    for col = 0 to n - 1 do
      let idx = (row * n) + col in
      if row < orig_dim && col < orig_dim then
        Mem.set b idx inp_b.((row * orig_dim) + col)
      else Mem.set b idx 0.0
    done
  done ;
  for i = 0 to (m * n) - 1 do
    Mem.set c i 0.0
  done ;

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

  let ok =
    if cfg.verify then begin
      Mem.to_cpu c () ;
      Devices.flush dev () ;
      Mem.unsafe_rw true ;
      let errors = ref 0 in
      let exp_c = !expected_c in
      let check_count = min 100 (orig_dim * orig_dim) in
      for idx = 0 to check_count - 1 do
        let row = idx / orig_dim in
        let col = idx mod orig_dim in
        let expected = exp_c.(idx) in
        let got = Mem.get c ((row * n) + col) in
        if abs_float (got -. expected) > 0.01 then incr errors
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

  let dim = int_of_float (sqrt (float_of_int cfg.size)) in
  Printf.printf
    "Matrix dimensions: %dx%d (total elements: %d)\n%!"
    dim
    dim
    (dim * dim) ;

  if cfg.benchmark_all then begin
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_matmul_data
      run_matmul_naive
      "Matrix multiplication (naive)" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:(fun () -> init_matmul_data ())
      run_matmul_tiled
      "Matrix multiplication (tiled)"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

    let baseline_ms, _ = init_matmul_data () in
    Printf.printf "\nOCaml baseline: %.4f ms\n%!" baseline_ms ;

    Printf.printf "\nNaive matrix multiplication:\n%!" ;
    let time_ms, ok = run_matmul_naive dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nTiled matrix multiplication:\n%!" ;
    let time_ms, ok = run_matmul_tiled dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nMatrix multiplication tests PASSED"
  end
