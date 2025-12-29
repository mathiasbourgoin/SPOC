(******************************************************************************
 * E2E test for Sarek PPX - Convolution
 *
 * Tests 1D and 2D convolution operations with various filter sizes.
 * Convolution is fundamental for signal processing and image filtering.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

(** 1D convolution with 3-point filter (blur) *)
let conv1d_3point_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid > 0 && tid < n - 1 then begin
        (* Simple box filter: [0.25, 0.5, 0.25] *)
        let left = input.(tid - 1) in
        let center = input.(tid) in
        let right = input.(tid + 1) in
        output.(tid) <- (0.25 *. left) +. (0.5 *. center) +. (0.25 *. right)
      end
      else if tid = 0 || tid = n - 1 then output.(tid) <- input.(tid)]

(** 1D convolution with 5-point filter *)
let conv1d_5point_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid >= 2 && tid < n - 2 then begin
        (* Gaussian-like filter *)
        let v0 = input.(tid - 2) in
        let v1 = input.(tid - 1) in
        let v2 = input.(tid) in
        let v3 = input.(tid + 1) in
        let v4 = input.(tid + 2) in
        output.(tid) <-
          (0.1 *. v0) +. (0.2 *. v1) +. (0.4 *. v2) +. (0.2 *. v3) +. (0.1 *. v4)
      end
      else output.(tid) <- input.(tid)]

(** 2D convolution with 3x3 filter (blur/smoothing) *)
let conv2d_3x3_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let x = thread_idx_x + (block_dim_x * block_idx_x) in
      let y = thread_idx_y + (block_dim_y * block_idx_y) in
      if x > 0 && x < width - 1 && y > 0 && y < height - 1 then begin
        let idx = (y * width) + x in
        (* 3x3 box filter *)
        let sum = mut 0.0 in
        sum := sum +. input.(idx - width - 1) ;
        sum := sum +. input.(idx - width) ;
        sum := sum +. input.(idx - width + 1) ;
        sum := sum +. input.(idx - 1) ;
        sum := sum +. input.(idx) ;
        sum := sum +. input.(idx + 1) ;
        sum := sum +. input.(idx + width - 1) ;
        sum := sum +. input.(idx + width) ;
        sum := sum +. input.(idx + width + 1) ;
        output.(idx) <- sum /. 9.0
      end
      else if x < width && y < height then begin
        let idx = (y * width) + x in
        output.(idx) <- input.(idx)
      end]

(** 2D Sobel edge detection (gradient magnitude) *)
let sobel_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let x = thread_idx_x + (block_dim_x * block_idx_x) in
      let y = thread_idx_y + (block_dim_y * block_idx_y) in
      if x > 0 && x < width - 1 && y > 0 && y < height - 1 then begin
        let idx = (y * width) + x in
        (* Load 3x3 neighborhood *)
        let p00 = input.(idx - width - 1) in
        let p01 = input.(idx - width) in
        let p02 = input.(idx - width + 1) in
        let p10 = input.(idx - 1) in
        let p12 = input.(idx + 1) in
        let p20 = input.(idx + width - 1) in
        let p21 = input.(idx + width) in
        let p22 = input.(idx + width + 1) in
        (* Sobel X: [-1 0 1; -2 0 2; -1 0 1] *)
        let gx =
          -.p00 +. p02 +. (-2.0 *. p10) +. (2.0 *. p12) +. -.p20 +. p22
        in
        (* Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1] *)
        let gy =
          -.p00 +. (-2.0 *. p01) +. -.p02 +. p20 +. (2.0 *. p21) +. p22
        in
        (* Gradient magnitude *)
        output.(idx) <- sqrt ((gx *. gx) +. (gy *. gy))
      end
      else if x < width && y < height then begin
        let idx = (y * width) + x in
        output.(idx) <- 0.0
      end]

(** 2D convolution with shared memory and supersteps. Uses 18x18 tile for 16x16
    block with 1-pixel halo on each side. Must load: center, 4 edges
    (left/right/top/bottom), and 4 corners. *)
let conv2d_shared_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let%shared (tile : float32) = 324l in
      (* 18x18 for 16x16 block + 1 halo *)
      let tile_width = 18l in
      let tx = thread_idx_x in
      let ty = thread_idx_y in
      let gx = thread_idx_x + (block_dim_x * block_idx_x) in
      let gy = thread_idx_y + (block_dim_y * block_idx_y) in
      (* Global block origin for corner calculations *)
      let block_origin_x = block_dim_x * block_idx_x in
      let block_origin_y = block_dim_y * block_idx_y in
      (* Load center elements *)
      let%superstep load_center =
        if gx < width && gy < height then
          tile.(((ty + 1l) * tile_width) + tx + 1l) <- input.((gy * width) + gx)
        else tile.(((ty + 1l) * tile_width) + tx + 1l) <- 0.0
      in
      (* Load left halo *)
      let%superstep load_left =
        if tx = 0l && gx > 0l && gy < height then
          tile.(((ty + 1l) * tile_width) + 0l) <- input.((gy * width) + gx - 1l)
        else if tx = 0l then tile.(((ty + 1l) * tile_width) + 0l) <- 0.0
      in
      (* Load right halo *)
      let%superstep load_right =
        if tx = block_dim_x - 1l && gx < width - 1l && gy < height then
          tile.(((ty + 1l) * tile_width) + tx + 2l) <-
            input.((gy * width) + gx + 1l)
        else if tx = block_dim_x - 1l then
          tile.(((ty + 1l) * tile_width) + tx + 2l) <- 0.0
      in
      (* Load top halo *)
      let%superstep load_top =
        if ty = 0l && gy > 0l && gx < width then
          tile.((0l * tile_width) + tx + 1l) <- input.(((gy - 1l) * width) + gx)
        else if ty = 0l then tile.((0l * tile_width) + tx + 1l) <- 0.0
      in
      (* Load bottom halo *)
      let%superstep load_bottom =
        if ty = block_dim_y - 1l && gy < height - 1l && gx < width then
          tile.(((ty + 2l) * tile_width) + tx + 1l) <-
            input.(((gy + 1l) * width) + gx)
        else if ty = block_dim_y - 1l then
          tile.(((ty + 2l) * tile_width) + tx + 1l) <- 0.0
      in
      (* Load corner halos - only thread (0,0) loads all 4 corners *)
      let%superstep load_corners =
        if tx = 0l && ty = 0l then begin
          (* Top-left corner *)
          if block_origin_x > 0l && block_origin_y > 0l then
            tile.(0l) <-
              input.(((block_origin_y - 1l) * width) + block_origin_x - 1l)
          else tile.(0l) <- 0.0 ;
          (* Top-right corner *)
          if block_origin_x + block_dim_x < width && block_origin_y > 0l then
            tile.(block_dim_x + 1l) <-
              input.(((block_origin_y - 1l) * width)
                     + block_origin_x + block_dim_x)
          else tile.(block_dim_x + 1l) <- 0.0 ;
          (* Bottom-left corner *)
          if block_origin_x > 0l && block_origin_y + block_dim_y < height then
            tile.(((block_dim_y + 1l) * tile_width) + 0l) <-
              input.(((block_origin_y + block_dim_y) * width)
                     + block_origin_x - 1l)
          else tile.(((block_dim_y + 1l) * tile_width) + 0l) <- 0.0 ;
          (* Bottom-right corner *)
          if
            block_origin_x + block_dim_x < width
            && block_origin_y + block_dim_y < height
          then
            tile.(((block_dim_y + 1l) * tile_width) + block_dim_x + 1l) <-
              input.(((block_origin_y + block_dim_y) * width)
                     + block_origin_x + block_dim_x)
          else
            tile.(((block_dim_y + 1l) * tile_width) + block_dim_x + 1l) <- 0.0
        end
      in
      (* Compute convolution from shared memory *)
      if gx > 0l && gx < width - 1l && gy > 0l && gy < height - 1l then begin
        let idx = ((ty + 1l) * tile_width) + tx + 1l in
        let sum =
          tile.(idx - tile_width - 1l)
          +. tile.(idx - tile_width)
          +. tile.(idx - tile_width + 1l)
          +. tile.(idx - 1l)
          +. tile.(idx)
          +. tile.(idx + 1l)
          +. tile.(idx + tile_width - 1l)
          +. tile.(idx + tile_width)
          +. tile.(idx + tile_width + 1l)
        in
        output.((gy * width) + gx) <- sum /. 9.0
      end
      else if gx < width && gy < height then
        output.((gy * width) + gx) <- tile.(((ty + 1l) * tile_width) + tx + 1l)]

(** Run 1D convolution test *)
let run_conv1d_test dev =
  let n = cfg.size in
  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  (* Initialize with sine wave *)
  for i = 0 to n - 1 do
    Mem.set input i (sin (float_of_int i *. 0.1)) ;
    Mem.set output i 0.0
  done ;

  ignore (Sarek.Kirc.gen conv1d_3point_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run conv1d_3point_kernel (input, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 1 to n - 2 do
        let expected =
          (0.25 *. Mem.get input (i - 1))
          +. (0.5 *. Mem.get input i)
          +. (0.25 *. Mem.get input (i + 1))
        in
        let got = Mem.get output i in
        if abs_float (got -. expected) > 0.0001 then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run 2D convolution test *)
let run_conv2d_test dev =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) in
  let width = dim in
  let height = dim in
  let n = width * height in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  (* Initialize with checkerboard pattern *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let idx = (y * width) + x in
      Mem.set input idx (float_of_int ((x + y) mod 2 * 100)) ;
      Mem.set output idx 0.0
    done
  done ;

  ignore (Sarek.Kirc.gen conv2d_3x3_kernel dev) ;
  let block_size = min 16 (Test_helpers.get_block_size cfg dev) in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    conv2d_3x3_kernel
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
      for y = 1 to height - 2 do
        for x = 1 to width - 2 do
          let idx = (y * width) + x in
          let sum = ref 0.0 in
          for dy = -1 to 1 do
            for dx = -1 to 1 do
              let nidx = ((y + dy) * width) + x + dx in
              sum := !sum +. Mem.get input nidx
            done
          done ;
          let expected = !sum /. 9.0 in
          let got = Mem.get output idx in
          if abs_float (got -. expected) > 0.01 then incr errors
        done
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run Sobel edge detection test *)
let run_sobel_test dev =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) in
  let width = dim in
  let height = dim in
  let n = width * height in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  (* Create an image with a vertical edge in the middle *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let idx = (y * width) + x in
      if x < width / 2 then Mem.set input idx 0.0 else Mem.set input idx 100.0 ;
      Mem.set output idx 0.0
    done
  done ;

  ignore (Sarek.Kirc.gen sobel_kernel dev) ;
  let block_size = min 16 (Test_helpers.get_block_size cfg dev) in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run sobel_kernel (input, output, width, height) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      (* Check that edges are detected near the middle *)
      let edge_x = width / 2 in
      let edge_sum = ref 0.0 in
      let non_edge_sum = ref 0.0 in
      for y = 2 to height - 3 do
        edge_sum := !edge_sum +. Mem.get output ((y * width) + edge_x) ;
        non_edge_sum := !non_edge_sum +. Mem.get output ((y * width) + 2)
      done ;
      (* Edges should have higher values than non-edges *)
      !edge_sum > !non_edge_sum *. 2.0
    end
    else true
  in
  (time_ms, ok)

(** Run 2D convolution with shared memory test *)
let run_conv2d_shared_test dev =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) in
  (* Round to multiple of 16 *)
  let dim = (dim + 15) / 16 * 16 in
  let width = dim in
  let height = dim in
  let n = width * height in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  (* Initialize with gradient *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let idx = (y * width) + x in
      Mem.set input idx (float_of_int ((x + y) mod 256)) ;
      Mem.set output idx 0.0
    done
  done ;

  ignore (Sarek.Kirc.gen conv2d_shared_kernel dev) ;
  let block_size = 16 in
  let blocks_x = width / block_size in
  let blocks_y = height / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    conv2d_shared_kernel
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
      for y = 1 to height - 2 do
        for x = 1 to width - 2 do
          let idx = (y * width) + x in
          let sum = ref 0.0 in
          for dy = -1 to 1 do
            for dx = -1 to 1 do
              let nidx = ((y + dy) * width) + x + dx in
              sum := !sum +. Mem.get input nidx
            done
          done ;
          let expected = !sum /. 9.0 in
          let got = Mem.get output idx in
          if abs_float (got -. expected) > 0.01 then begin
            if !errors < 10 then
              Printf.printf
                "  Mismatch at (%d,%d): expected %.2f, got %.2f\n"
                x
                y
                expected
                got ;
            incr errors
          end
        done
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_convolution" in
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
      run_conv1d_test
      "1D convolution (3-point)" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_conv2d_test
      "2D convolution (3x3)" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_conv2d_shared_test
      "2D convolution (shared)" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_sobel_test
      "Sobel edge detection"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    Printf.printf "Testing convolution operations with size=%d\n%!" cfg.size ;

    Printf.printf "\n1D convolution (3-point):\n%!" ;
    let time_ms, ok = run_conv1d_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\n2D convolution (3x3):\n%!" ;
    let time_ms, ok = run_conv2d_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\n2D convolution (shared memory):\n%!" ;
    let time_ms, ok = run_conv2d_shared_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nSobel edge detection:\n%!" ;
    let time_ms, ok = run_sobel_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nConvolution tests PASSED"
  end
