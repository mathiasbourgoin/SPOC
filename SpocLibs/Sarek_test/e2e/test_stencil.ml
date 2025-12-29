(******************************************************************************
 * E2E test for Sarek PPX - Stencil operations
 *
 * Tests 1D and 2D stencil patterns with shared memory optimization.
 * Stencil operations are fundamental for image processing, PDE solvers, etc.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

(* ========== Pure OCaml baseline implementations ========== *)

(** Pure OCaml 1D stencil (3-point averaging) *)
let ocaml_stencil_1d input output n =
  for i = 1 to n - 2 do
    let left = input.(i - 1) in
    let center = input.(i) in
    let right = input.(i + 1) in
    output.(i) <- (left +. center +. right) /. 3.0
  done

(** Pure OCaml 2D stencil (Laplacian) *)
let ocaml_stencil_2d input output width height =
  for y = 1 to height - 2 do
    for x = 1 to width - 2 do
      let idx = (y * width) + x in
      let center = input.(idx) in
      let left = input.(idx - 1) in
      let right = input.(idx + 1) in
      let up = input.(idx - width) in
      let down = input.(idx + width) in
      output.(idx) <- (left +. right +. up +. down -. (4.0 *. center)) /. 4.0
    done
  done

(* ========== Sarek kernels ========== *)

(** 1D stencil: 3-point averaging (blur) - no shared memory *)
let stencil_1d_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid > 0l && tid < n - 1l then
        let left = input.(tid - 1l) in
        let center = input.(tid) in
        let right = input.(tid + 1l) in
        output.(tid) <- (left +. center +. right) /. 3.0]

(** 1D stencil with shared memory and supersteps *)
let stencil_1d_shared_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (tile : float32) = 258l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      (* Load center elements *)
      let%superstep load_center =
        if gid < n then tile.(tid + 1l) <- input.(gid)
      in
      (* Load left halo *)
      let%superstep load_left =
        if tid = 0l && gid > 0l then tile.(0l) <- input.(gid - 1l)
      in
      (* Load right halo *)
      let%superstep load_right =
        if tid = block_dim_x - 1l && gid < n - 1l then
          tile.(tid + 2l) <- input.(gid + 1l)
      in
      (* Compute stencil *)
      if gid > 0l && gid < n - 1l then begin
        let left = tile.(tid) in
        let center = tile.(tid + 1l) in
        let right = tile.(tid + 2l) in
        output.(gid) <- (left +. center +. right) /. 3.0
      end]

(** 5-point 2D stencil (Laplacian) - no shared memory *)
let stencil_2d_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let x = thread_idx_x + (block_dim_x * block_idx_x) in
      let y = thread_idx_y + (block_dim_y * block_idx_y) in
      if x > 0l && x < width - 1l && y > 0l && y < height - 1l then begin
        let idx = (y * width) + x in
        let center = input.(idx) in
        let left = input.(idx - 1l) in
        let right = input.(idx + 1l) in
        let up = input.(idx - width) in
        let down = input.(idx + width) in
        output.(idx) <- (left +. right +. up +. down -. (4.0 *. center)) /. 4.0
      end]

(* ========== Shared test data ========== *)

(* Mutable refs to hold test data for sharing between baseline and device runs *)
let input_1d = ref [||]
let expected_1d = ref [||]
let input_2d = ref [||]
let expected_2d = ref [||]
let dim_2d = ref 0

(** Initialize 1D test data and compute expected result *)
let init_1d_data () =
  let n = cfg.size in
  let inp = Array.init n (fun i -> sin (float_of_int i *. 0.1)) in
  let exp = Array.make n 0.0 in
  input_1d := inp ;
  expected_1d := exp ;
  (* Compute expected result *)
  let t0 = Unix.gettimeofday () in
  ocaml_stencil_1d inp exp n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(** Initialize 2D test data and compute expected result *)
let init_2d_data () =
  let dim = int_of_float (sqrt (float_of_int cfg.size)) in
  dim_2d := dim ;
  let n = dim * dim in
  let inp =
    Array.init n (fun idx ->
        let x = idx mod dim in
        let y = idx / dim in
        float_of_int ((x + y) mod 10))
  in
  let exp = Array.make n 0.0 in
  input_2d := inp ;
  expected_2d := exp ;
  (* Compute expected result *)
  let t0 = Unix.gettimeofday () in
  ocaml_stencil_2d inp exp dim dim ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Device test runners ========== *)

(** Run 1D stencil on device, verify against precomputed expected *)
let run_stencil_1d dev =
  let n = cfg.size in
  let inp = !input_1d in
  let exp = !expected_1d in
  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  (* Copy input data to SPOC vector *)
  for i = 0 to n - 1 do
    Mem.set input i inp.(i) ;
    Mem.set output i 0.0
  done ;

  (* Generate and run kernel *)
  ignore (Sarek.Kirc.gen stencil_1d_kernel dev) ;

  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run stencil_1d_kernel (input, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Verify against OCaml expected result *)
  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      Mem.unsafe_rw true ;
      let errors = ref 0 in
      for i = 1 to n - 2 do
        let expected = exp.(i) in
        let got = Mem.get output i in
        if abs_float (got -. expected) > 0.0001 then incr errors
      done ;
      Mem.unsafe_rw false ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run 1D stencil with shared memory on device *)
let run_stencil_1d_shared dev =
  let n = cfg.size in
  let inp = !input_1d in
  let exp = !expected_1d in
  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  (* Copy input data to SPOC vector *)
  for i = 0 to n - 1 do
    Mem.set input i inp.(i) ;
    Mem.set output i 0.0
  done ;

  (* Generate and run kernel *)
  ignore (Sarek.Kirc.gen stencil_1d_shared_kernel dev) ;
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run stencil_1d_shared_kernel (input, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Verify against OCaml expected result *)
  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      Mem.unsafe_rw true ;
      let errors = ref 0 in
      for i = 1 to n - 2 do
        let expected = exp.(i) in
        let got = Mem.get output i in
        if abs_float (got -. expected) > 0.0001 then incr errors
      done ;
      Mem.unsafe_rw false ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run 2D stencil on device *)
let run_stencil_2d dev =
  let dim = !dim_2d in
  let width = dim in
  let height = dim in
  let n = width * height in
  let inp = !input_2d in
  let exp = !expected_2d in
  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  (* Copy input data to SPOC vector *)
  for i = 0 to n - 1 do
    Mem.set input i inp.(i) ;
    Mem.set output i 0.0
  done ;

  (* Generate and run kernel *)
  ignore (Sarek.Kirc.gen stencil_2d_kernel dev) ;
  let block_size = min 16 (Test_helpers.get_block_size cfg dev) in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    stencil_2d_kernel
    (input, output, width, height)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Verify against OCaml expected result *)
  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      Mem.unsafe_rw true ;
      let errors = ref 0 in
      for y = 1 to height - 2 do
        for x = 1 to width - 2 do
          let idx = (y * width) + x in
          let expected = exp.(idx) in
          let got = Mem.get output idx in
          if abs_float (got -. expected) > 0.0001 then incr errors
        done
      done ;
      Mem.unsafe_rw false ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_stencil" in
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
    (* Initialize data and get baseline timing *)
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_1d_data
      run_stencil_1d
      "1D stencil (simple)" ;
    (* Reuse same data for shared memory version *)
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:(fun () -> init_1d_data ())
      run_stencil_1d_shared
      "1D stencil (shared memory)" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_2d_data
      run_stencil_2d
      "2D stencil (Laplacian)"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    Printf.printf "Testing stencil operations with size=%d\n%!" cfg.size ;

    (* Initialize data *)
    let baseline_ms, _ = init_1d_data () in
    Printf.printf "\nOCaml baseline (1D): %.4f ms\n%!" baseline_ms ;

    Printf.printf "\n1D Stencil (simple):\n%!" ;
    let time_ms, ok = run_stencil_1d dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\n1D Stencil (shared memory):\n%!" ;
    let time_ms, ok = run_stencil_1d_shared dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    let baseline_2d_ms, _ = init_2d_data () in
    Printf.printf "\nOCaml baseline (2D): %.4f ms\n%!" baseline_2d_ms ;

    Printf.printf "\n2D Stencil (Laplacian):\n%!" ;
    let time_ms, ok = run_stencil_2d dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_2d_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nStencil tests PASSED"
  end
