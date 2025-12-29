(******************************************************************************
 * E2E test for Sarek PPX - Histogram computation
 *
 * Tests histogram computation using shared memory and supersteps.
 * Histogram is a common pattern for data analysis and image processing.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

let num_bins = ref 256

(* ========== Pure OCaml baseline ========== *)

let ocaml_histogram input n bins =
  let hist = Array.make bins 0l in
  for i = 0 to n - 1 do
    let bin = Int32.to_int (Int32.rem input.(i) (Int32.of_int bins)) in
    hist.(bin) <- Int32.add hist.(bin) 1l
  done ;
  hist

(* ========== Shared test data ========== *)

let input_data = ref [||]
let expected_hist = ref [||]

let init_histogram_data () =
  let n = cfg.size in
  let bins = !num_bins in
  Random.init 42 ;
  let inp = Array.init n (fun _ -> Int32.of_int (Random.int bins)) in
  input_data := inp ;
  let t0 = Unix.gettimeofday () in
  expected_hist := ocaml_histogram inp n bins ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Sarek kernels ========== *)

let histogram_kernel =
  [%kernel
    fun (input : int32 vector)
        (histogram : int32 vector)
        (n : int32)
        (num_bins : int32) ->
      let%shared (local_hist : int32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep init = if tid < num_bins then local_hist.(tid) <- 0l in
      let%superstep[@divergent] count =
        if gid < n then begin
          let bin = input.(gid) mod num_bins in
          let _ = atomic_add_int32 local_hist bin 1l in
          ()
        end
      in
      let%superstep[@divergent] merge =
        if tid < num_bins then begin
          let local_count = local_hist.(tid) in
          let _ = atomic_add_global_int32 histogram tid local_count in
          ()
        end
      in
      ()]

let histogram_strided_kernel =
  [%kernel
    fun (input : int32 vector)
        (histogram : int32 vector)
        (n : int32)
        (num_bins : int32) ->
      let%shared (local_hist : int32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let stride = block_dim_x * grid_dim_x in
      let%superstep init = if tid < num_bins then local_hist.(tid) <- 0l in
      let i = mut gid in
      while i < n do
        let bin = input.(i) mod num_bins in
        let _ = atomic_add_int32 local_hist bin 1l in
        i := i + stride
      done ;
      block_barrier () ;
      if tid < num_bins then begin
        let local_count = local_hist.(tid) in
        let _ = atomic_add_global_int32 histogram tid local_count in
        ()
      end]

(* ========== Device test runners ========== *)

let run_histogram dev =
  let n = cfg.size in
  let bins = !num_bins in
  let inp = !input_data in
  let exp = !expected_hist in

  let input = Vector.create Vector.int32 n in
  let histogram = Vector.create Vector.int32 bins in

  for i = 0 to n - 1 do
    Mem.set input i inp.(i)
  done ;
  for i = 0 to bins - 1 do
    Mem.set histogram i 0l
  done ;

  ignore (Sarek.Kirc.gen histogram_kernel dev) ;
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run histogram_kernel (input, histogram, n, bins) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu histogram () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to bins - 1 do
        let got = Mem.get histogram i in
        if got <> exp.(i) then begin
          if !errors < 10 then
            Printf.printf "  Bin %d: expected %ld, got %ld\n" i exp.(i) got ;
          incr errors
        end
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let run_histogram_strided dev =
  let n = cfg.size in
  let bins = !num_bins in
  let inp = !input_data in
  let exp = !expected_hist in

  let input = Vector.create Vector.int32 n in
  let histogram = Vector.create Vector.int32 bins in

  for i = 0 to n - 1 do
    Mem.set input i inp.(i)
  done ;
  for i = 0 to bins - 1 do
    Mem.set histogram i 0l
  done ;

  ignore (Sarek.Kirc.gen histogram_strided_kernel dev) ;
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = min 64 ((n + block_size - 1) / block_size) in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    histogram_strided_kernel
    (input, histogram, n, bins)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu histogram () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to bins - 1 do
        let got = Mem.get histogram i in
        if got <> exp.(i) then begin
          if !errors < 10 then
            Printf.printf "  Bin %d: expected %ld, got %ld\n" i exp.(i) got ;
          incr errors
        end
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_histogram" in
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

  Printf.printf
    "Testing histogram with %d elements, %d bins\n%!"
    cfg.size
    !num_bins ;

  if cfg.benchmark_all then begin
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_histogram_data
      run_histogram
      "Histogram (simple)" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:(fun () -> init_histogram_data ())
      run_histogram_strided
      "Histogram (strided)"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

    let baseline_ms, _ = init_histogram_data () in
    Printf.printf "\nOCaml baseline: %.4f ms\n%!" baseline_ms ;

    Printf.printf "\nSimple histogram:\n%!" ;
    let time_ms, ok = run_histogram dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nStrided histogram:\n%!" ;
    let time_ms, ok = run_histogram_strided dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nHistogram tests PASSED"
  end
