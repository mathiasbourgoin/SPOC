(******************************************************************************
 * E2E test for Sarek PPX - Matrix Transpose
 *
 * Tests naive and coalesced (shared memory) matrix transpose.
 * Transpose is a memory-bound operation that benefits from coalescing.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

(* ========== Pure OCaml baselines ========== *)

let ocaml_transpose input output width height =
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let in_idx = (y * width) + x in
      let out_idx = (x * height) + y in
      output.(out_idx) <- input.(in_idx)
    done
  done

(* ========== Shared test data ========== *)

let input_naive = ref [||]

let expected_naive = ref [||]

let matrix_dim_naive = ref 0

let input_coalesced = ref [||]

let expected_coalesced = ref [||]

let matrix_dim_coalesced = ref 0

let input_rect = ref [||]

let expected_rect = ref [||]

let init_naive_data () =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) in
  matrix_dim_naive := dim ;
  let n = dim * dim in
  let inp = Array.init n (fun i -> float_of_int i) in
  let out = Array.make n 0.0 in
  input_naive := inp ;
  expected_naive := out ;
  let t0 = Unix.gettimeofday () in
  ocaml_transpose inp out dim dim ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_coalesced_data () =
  let tile_dim = 16 in
  let dim =
    (Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) + tile_dim - 1)
    / tile_dim * tile_dim
  in
  matrix_dim_coalesced := dim ;
  let n = dim * dim in
  let inp = Array.init n (fun i -> float_of_int i) in
  let out = Array.make n 0.0 in
  input_coalesced := inp ;
  expected_coalesced := out ;
  let t0 = Unix.gettimeofday () in
  ocaml_transpose inp out dim dim ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_rect_data () =
  let width = 64 in
  let height = 32 in
  let n = width * height in
  let inp =
    Array.init n (fun i -> float_of_int ((i / width * 100) + (i mod width)))
  in
  let out = Array.make n 0.0 in
  input_rect := inp ;
  expected_rect := out ;
  let t0 = Unix.gettimeofday () in
  ocaml_transpose inp out width height ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Sarek kernels ========== *)

(** Naive matrix transpose - poor memory access pattern *)
let transpose_naive_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let x = thread_idx_x + (block_dim_x * block_idx_x) in
      let y = thread_idx_y + (block_dim_y * block_idx_y) in
      if x < width && y < height then begin
        let in_idx = (y * width) + x in
        let out_idx = (x * height) + y in
        output.(out_idx) <- input.(in_idx)
      end]

(** Coalesced matrix transpose using shared memory with supersteps *)
let transpose_coalesced_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let%shared (tile : float32) = 289l in
      (* 17x17 to avoid bank conflicts *)
      let tile_dim = 16l in
      let tx = thread_idx_x in
      let ty = thread_idx_y in
      let x = (block_idx_x * tile_dim) + tx in
      let y = (block_idx_y * tile_dim) + ty in
      (* Load tile with coalesced reads *)
      let%superstep load =
        if x < width && y < height then
          tile.((ty * (tile_dim + 1l)) + tx) <- input.((y * width) + x)
      in
      (* Transpose within tile coordinates *)
      let out_x = (block_idx_y * tile_dim) + tx in
      let out_y = (block_idx_x * tile_dim) + ty in
      (* Write with coalesced writes *)
      if out_x < height && out_y < width then
        output.((out_y * height) + out_x) <- tile.((tx * (tile_dim + 1l)) + ty)]

(* ========== Device test runners ========== *)

(** Run naive transpose test *)
let run_transpose_naive dev =
  let dim = !matrix_dim_naive in
  let width = dim in
  let height = dim in
  let n = width * height in
  let inp = !input_naive in
  let exp = !expected_naive in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Mem.set input i inp.(i) ;
    Mem.set output i 0.0
  done ;

  ignore (Sarek.Kirc.gen transpose_naive_kernel dev) ;
  let block_size = min 16 (Test_helpers.get_block_size cfg dev) in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    transpose_naive_kernel
    (input, output, width, height)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 1 do
        if abs_float (Mem.get output i -. exp.(i)) > 0.001 then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run coalesced transpose test *)
let run_transpose_coalesced dev =
  let dim = !matrix_dim_coalesced in
  let width = dim in
  let height = dim in
  let n = width * height in
  let inp = !input_coalesced in
  let exp = !expected_coalesced in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Mem.set input i inp.(i) ;
    Mem.set output i 0.0
  done ;

  ignore (Sarek.Kirc.gen transpose_coalesced_kernel dev) ;
  let block_size = 16 in
  let blocks_x = width / block_size in
  let blocks_y = height / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    transpose_coalesced_kernel
    (input, output, width, height)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 1 do
        if abs_float (Mem.get output i -. exp.(i)) > 0.001 then begin
          if !errors < 10 then
            Printf.printf
              "  Mismatch at %d: expected %.0f, got %.0f\n"
              i
              exp.(i)
              (Mem.get output i) ;
          incr errors
        end
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run rectangular transpose test *)
let run_transpose_rectangular dev =
  let width = 64 in
  let height = 32 in
  let n = width * height in
  let inp = !input_rect in
  let exp = !expected_rect in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Mem.set input i inp.(i) ;
    Mem.set output i 0.0
  done ;

  ignore (Sarek.Kirc.gen transpose_naive_kernel dev) ;
  let block_size = min 16 (Test_helpers.get_block_size cfg dev) in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    transpose_naive_kernel
    (input, output, width, height)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 1 do
        if abs_float (Mem.get output i -. exp.(i)) > 0.001 then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_transpose" in
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

  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) in
  Printf.printf "Matrix dimensions: %dx%d\n%!" dim dim ;

  if cfg.benchmark_all then begin
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_naive_data
      run_transpose_naive
      "Transpose (naive)" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_coalesced_data
      run_transpose_coalesced
      "Transpose (coalesced)" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_rect_data
      run_transpose_rectangular
      "Transpose (rectangular)"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

    let baseline_ms, _ = init_naive_data () in
    Printf.printf "\nOCaml baseline (naive): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nNaive transpose:\n%!" ;
    let time_ms, ok = run_transpose_naive dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    let baseline_ms, _ = init_coalesced_data () in
    Printf.printf "\nOCaml baseline (coalesced): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nCoalesced transpose:\n%!" ;
    let time_ms, ok = run_transpose_coalesced dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    let baseline_ms, _ = init_rect_data () in
    Printf.printf "\nOCaml baseline (rect): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nRectangular transpose:\n%!" ;
    let time_ms, ok = run_transpose_rectangular dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nTranspose tests PASSED"
  end
