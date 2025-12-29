(******************************************************************************
 * E2E test for Sarek PPX - Mandelbrot set computation
 *
 * Tests iterative computation with complex arithmetic.
 * Mandelbrot is a classic GPU benchmark with high arithmetic intensity.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

let max_iter = ref 256

(* ========== Pure OCaml baselines ========== *)

let ocaml_mandelbrot output width height max_iter =
  for py = 0 to height - 1 do
    for px = 0 to width - 1 do
      let x0 = (4.0 *. float_of_int px /. float_of_int width) -. 2.5 in
      let y0 = (3.0 *. float_of_int py /. float_of_int height) -. 1.5 in
      let x = ref 0.0 in
      let y = ref 0.0 in
      let iter = ref 0 in
      while (!x *. !x) +. (!y *. !y) <= 4.0 && !iter < max_iter do
        let xtemp = (!x *. !x) -. (!y *. !y) +. x0 in
        y := (2.0 *. !x *. !y) +. y0 ;
        x := xtemp ;
        incr iter
      done ;
      output.((py * width) + px) <- Int32.of_int !iter
    done
  done

let ocaml_julia output width height c_real c_imag max_iter =
  for py = 0 to height - 1 do
    for px = 0 to width - 1 do
      let x = ref ((4.0 *. float_of_int px /. float_of_int width) -. 2.0) in
      let y = ref ((4.0 *. float_of_int py /. float_of_int height) -. 2.0) in
      let iter = ref 0 in
      while (!x *. !x) +. (!y *. !y) <= 4.0 && !iter < max_iter do
        let xtemp = (!x *. !x) -. (!y *. !y) +. c_real in
        y := (2.0 *. !x *. !y) +. c_imag ;
        x := xtemp ;
        incr iter
      done ;
      output.((py * width) + px) <- Int32.of_int !iter
    done
  done

(* ========== Shared test data ========== *)

let expected_mandelbrot = ref [||]
let expected_julia = ref [||]
let image_dim = ref 0

let init_mandelbrot_data () =
  let dim = int_of_float (sqrt (float_of_int cfg.size)) in
  image_dim := dim ;
  let n = dim * dim in
  let output = Array.make n 0l in
  expected_mandelbrot := output ;
  let t0 = Unix.gettimeofday () in
  ocaml_mandelbrot output dim dim !max_iter ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_julia_data () =
  let dim = !image_dim in
  let n = dim * dim in
  let output = Array.make n 0l in
  expected_julia := output ;
  let c_real = -0.7 in
  let c_imag = 0.27015 in
  let t0 = Unix.gettimeofday () in
  ocaml_julia output dim dim c_real c_imag !max_iter ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Device test runners ========== *)

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
  let dim = !image_dim in
  let width = dim in
  let height = dim in
  let n = width * height in
  let exp = !expected_mandelbrot in

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
      let errors = ref 0 in
      for i = 0 to min 1000 (n - 1) do
        let got = Mem.get output i in
        if got <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

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
  let dim = !image_dim in
  let width = dim in
  let height = dim in
  let n = width * height in
  let exp = !expected_julia in

  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Mem.set output i 0l
  done ;

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
      let errors = ref 0 in
      for i = 0 to min 1000 (n - 1) do
        let got = Mem.get output i in
        if got <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_mandelbrot" in
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
  Printf.printf "Image dimensions: %dx%d, max_iter=%d\n%!" dim dim !max_iter ;

  if cfg.benchmark_all then begin
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_mandelbrot_data
      run_mandelbrot_test
      "Mandelbrot" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_julia_data
      run_julia_test
      "Julia set"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

    let baseline_ms, _ = init_mandelbrot_data () in
    Printf.printf "\nOCaml baseline (Mandelbrot): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nMandelbrot:\n%!" ;
    let time_ms, ok = run_mandelbrot_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    let baseline_ms, _ = init_julia_data () in
    Printf.printf "\nOCaml baseline (Julia): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nJulia set:\n%!" ;
    let time_ms, ok = run_julia_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nMandelbrot tests PASSED"
  end
