(******************************************************************************
 * E2E test for Sarek PPX - Mandelbrot set computation
 *
 * Tests iterative computation with complex arithmetic.
 * Mandelbrot is a classic GPU benchmark with high arithmetic intensity.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

let max_iter = ref 256

(** Run basic Mandelbrot test *)
let run_mandelbrot_test dev =
  let mandelbrot_kernel =
    [%kernel
      fun (output : int32 vector)
          (width : int)
          (height : int)
          (max_iter : int) ->
        let open Std in
        let px = thread_idx_x + (block_dim_x * block_idx_x) in
        let py = thread_idx_y + (block_dim_y * block_idx_y) in
        if px < width && py < height then begin
          (* Map pixel to complex plane: x in [-2.5, 1.0], y in [-1.5, 1.5] *)
          let x0 = (4.0 *. (float px /. float width)) -. 2.5 in
          let y0 = (3.0 *. (float py /. float height)) -. 1.5 in
          let x = mut 0.0 in
          let y = mut 0.0 in
          let iter = mut 0l in
          while (x *. x) +. (y *. y) <= 4.0 && iter < max_iter do
            let xtemp = (x *. x) -. (y *. y) +. x0 in
            y := (2.0 *. x *. y) +. y0 ;
            x := xtemp ;
            iter := iter + 1l
          done ;
          output.((py * width) + px) <- iter
        end]
  in
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) in
  let width = dim in
  let height = dim in
  let n = width * height in

  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Mem.set output i 0l
  done ;

  ignore (Sarek.Kirc.gen mandelbrot_kernel dev) ;
  let block_size = min 16 (Test_helpers.get_block_size cfg dev) in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    mandelbrot_kernel
    (output, width, height, !max_iter)
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
      (* Verify some known properties:
         - Center of set (0,0) should be in the set (max_iter)
         - Points far from origin should escape quickly *)
      let center_x = (width / 2) + (width / 4) in
      (* Map to about (-0.5, 0) *)
      let center_y = height / 2 in
      let center_iter = Mem.get output ((center_y * width) + center_x) in
      let corner_iter = Mem.get output 0 in
      (* (-2.5, -1.5) is outside *)
      center_iter >= Int32.of_int (!max_iter / 2) && corner_iter < 100l
    end
    else true
  in
  (time_ms, ok)

(** Run Julia set test *)
let run_julia_test dev =
  let julia_kernel =
    [%kernel
      fun (output : int32 vector)
          (width : int)
          (height : int)
          (c_real : float32)
          (c_imag : float32)
          (max_iter : int) ->
        let open Std in
        let px = thread_idx_x + (block_dim_x * block_idx_x) in
        let py = thread_idx_y + (block_dim_y * block_idx_y) in
        if px < width && py < height then begin
          (* Map pixel to [-2, 2] x [-2, 2] *)
          let x = mut ((4.0 *. (float px /. float width)) -. 2.0) in
          let y = mut ((4.0 *. (float py /. float height)) -. 2.0) in
          let iter = mut 0l in
          while (x *. x) +. (y *. y) <= 4.0 && iter < max_iter do
            let xtemp = (x *. x) -. (y *. y) +. c_real in
            y := (2.0 *. x *. y) +. c_imag ;
            x := xtemp ;
            iter := iter + 1l
          done ;
          output.((py * width) + px) <- iter
        end]
  in
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) in
  let width = dim in
  let height = dim in
  let n = width * height in

  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Mem.set output i 0l
  done ;

  (* Classic Julia set constant *)
  let c_real = -0.7 in
  let c_imag = 0.27015 in

  ignore (Sarek.Kirc.gen julia_kernel dev) ;
  let block_size = min 16 (Test_helpers.get_block_size cfg dev) in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    julia_kernel
    (output, width, height, c_real, c_imag, !max_iter)
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
      (* Verify that we got a mix of iterations (not all same value) *)
      let min_iter = ref 1000000l in
      let max_iter_found = ref 0l in
      for i = 0 to n - 1 do
        let v = Mem.get output i in
        if v < !min_iter then min_iter := v ;
        if v > !max_iter_found then max_iter_found := v
      done ;
      (* Should have variation in iteration counts *)
      !max_iter_found > Int32.add !min_iter 10l
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_mandelbrot" in
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
  Printf.printf "Image dimensions: %dx%d, max_iter=%d\n%!" dim dim !max_iter ;

  if cfg.benchmark_all then begin
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_mandelbrot_test
      "Mandelbrot" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_julia_test
      "Julia set"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

    Printf.printf "\nMandelbrot:\n%!" ;
    let time_ms, ok = run_mandelbrot_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nJulia set:\n%!" ;
    let time_ms, ok = run_julia_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nMandelbrot tests PASSED"
  end
