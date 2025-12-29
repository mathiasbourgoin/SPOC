(******************************************************************************
 * E2E test for Sarek PPX - Stencil operations
 *
 * Tests 1D and 2D stencil patterns with shared memory optimization.
 * Stencil operations are fundamental for image processing, PDE solvers, etc.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

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

(** Run 1D stencil test *)
let run_stencil_1d dev =
  let n = cfg.size in
  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  (* Initialize with sine wave *)
  for i = 0 to n - 1 do
    Mem.set input i (sin (float_of_int i *. 0.1)) ;
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

  (* Verify *)
  let ok =
    if cfg.verify then begin
      Mem.to_cpu input () ;
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      Mem.unsafe_rw true ;
      let errors = ref 0 in
      for i = 1 to n - 2 do
        let left = Mem.get input (i - 1) in
        let center = Mem.get input i in
        let right = Mem.get input (i + 1) in
        let expected = (left +. center +. right) /. 3.0 in
        let got = Mem.get output i in
        if abs_float (got -. expected) > 0.0001 then incr errors
      done ;
      Mem.unsafe_rw false ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run 1D stencil with shared memory test *)
let run_stencil_1d_shared dev =
  let n = cfg.size in
  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  (* Initialize with sine wave *)
  for i = 0 to n - 1 do
    Mem.set input i (sin (float_of_int i *. 0.1)) ;
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

  (* Verify *)
  let ok =
    if cfg.verify then begin
      Mem.to_cpu input () ;
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      Mem.unsafe_rw true ;
      let errors = ref 0 in
      for i = 1 to n - 2 do
        let left = Mem.get input (i - 1) in
        let center = Mem.get input i in
        let right = Mem.get input (i + 1) in
        let expected = (left +. center +. right) /. 3.0 in
        let got = Mem.get output i in
        if abs_float (got -. expected) > 0.0001 then incr errors
      done ;
      Mem.unsafe_rw false ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run 2D stencil test *)
let run_stencil_2d dev =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) in
  let width = dim in
  let height = dim in
  let n = width * height in
  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  (* Initialize with pattern *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let idx = (y * width) + x in
      Mem.set input idx (float_of_int ((x + y) mod 10)) ;
      Mem.set output idx 0.0
    done
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

  (* Verify *)
  let ok =
    if cfg.verify then begin
      Mem.to_cpu input () ;
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      Mem.unsafe_rw true ;
      let errors = ref 0 in
      for y = 1 to height - 2 do
        for x = 1 to width - 2 do
          let idx = (y * width) + x in
          let center = Mem.get input idx in
          let left = Mem.get input (idx - 1) in
          let right = Mem.get input (idx + 1) in
          let up = Mem.get input (idx - width) in
          let down = Mem.get input (idx + width) in
          let expected =
            (left +. right +. up +. down -. (4.0 *. center)) /. 4.0
          in
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
      run_stencil_1d
      "1D stencil (simple)" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_stencil_1d_shared
      "1D stencil (shared memory)" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_stencil_2d
      "2D stencil (Laplacian)"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    Printf.printf "Testing stencil operations with size=%d\n%!" cfg.size ;

    Printf.printf "\n1D Stencil (simple):\n%!" ;
    let time_ms, ok = run_stencil_1d dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\n1D Stencil (shared memory):\n%!" ;
    let time_ms, ok = run_stencil_1d_shared dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\n2D Stencil (Laplacian):\n%!" ;
    let time_ms, ok = run_stencil_2d dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nStencil tests PASSED"
  end
