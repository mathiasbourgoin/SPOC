(******************************************************************************
 * E2E test for Sarek PPX - Histogram computation with V2 comparison
 *
 * Tests histogram computation using shared memory, supersteps, and atomics.
 * Histogram is a common pattern for data analysis and image processing.
 ******************************************************************************)

(* Module aliases *)
module Spoc_Vector = Spoc.Vector
module Spoc_Devices = Spoc.Devices
module Spoc_Mem = Spoc.Mem
module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

let cfg = Test_helpers.default_config ()

(** Get appropriate block size for V2 device *)
let get_block_size_v2 (dev : V2_Device.t) =
  if dev.capabilities.is_cpu then
    if cfg.block_size > 1 then cfg.block_size else 64
  else cfg.block_size

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

(* ========== SPOC test runners ========== *)

let run_histogram_spoc dev =
  let n = cfg.size in
  let bins = !num_bins in
  let inp = !input_data in
  let exp = !expected_hist in

  let input = Spoc_Vector.create Spoc_Vector.int32 n in
  let histogram = Spoc_Vector.create Spoc_Vector.int32 bins in

  for i = 0 to n - 1 do
    Spoc_Mem.set input i inp.(i)
  done ;
  for i = 0 to bins - 1 do
    Spoc_Mem.set histogram i 0l
  done ;

  ignore (Sarek.Kirc.gen histogram_kernel dev) ;
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = (n + block_size - 1) / block_size in
  let block = {Spoc.Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Spoc.Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    histogram_kernel
    (input, histogram, n, bins)
    (block, grid)
    0
    dev ;
  Spoc_Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Spoc_Mem.to_cpu histogram () ;
      Spoc_Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to bins - 1 do
        let got = Spoc_Mem.get histogram i in
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

let run_histogram_strided_spoc dev =
  let n = cfg.size in
  let bins = !num_bins in
  let inp = !input_data in
  let exp = !expected_hist in

  let input = Spoc_Vector.create Spoc_Vector.int32 n in
  let histogram = Spoc_Vector.create Spoc_Vector.int32 bins in

  for i = 0 to n - 1 do
    Spoc_Mem.set input i inp.(i)
  done ;
  for i = 0 to bins - 1 do
    Spoc_Mem.set histogram i 0l
  done ;

  ignore (Sarek.Kirc.gen histogram_strided_kernel dev) ;
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = min 64 ((n + block_size - 1) / block_size) in
  let block = {Spoc.Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Spoc.Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    histogram_strided_kernel
    (input, histogram, n, bins)
    (block, grid)
    0
    dev ;
  Spoc_Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Spoc_Mem.to_cpu histogram () ;
      Spoc_Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to bins - 1 do
        let got = Spoc_Mem.get histogram i in
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

(* ========== V2 test runners ========== *)

let run_histogram_v2 (dev : V2_Device.t) =
  let n = cfg.size in
  let bins = !num_bins in
  let inp = !input_data in
  let exp = !expected_hist in

  let _, kirc = histogram_kernel in
  let ir =
    match kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  let input = V2_Vector.create V2_Vector.int32 n in
  let histogram = V2_Vector.create V2_Vector.int32 bins in

  for i = 0 to n - 1 do
    V2_Vector.set input i inp.(i)
  done ;
  for i = 0 to bins - 1 do
    V2_Vector.set histogram i 0l
  done ;

  let block_size = min 256 (get_block_size_v2 dev) in
  let num_blocks = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d num_blocks in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec histogram;
        Sarek.Execute.Int32 (Int32.of_int n);
        Sarek.Execute.Int32 (Int32.of_int bins);
      ]
    ~block
    ~grid
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = V2_Vector.to_array histogram in
      let errors = ref 0 in
      for i = 0 to bins - 1 do
        if result.(i) <> exp.(i) then begin
          if !errors < 10 then
            Printf.printf
              "  Bin %d: expected %ld, got %ld\n"
              i
              exp.(i)
              result.(i) ;
          incr errors
        end
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let run_histogram_strided_v2 (dev : V2_Device.t) =
  let n = cfg.size in
  let bins = !num_bins in
  let inp = !input_data in
  let exp = !expected_hist in

  let _, kirc = histogram_strided_kernel in
  let ir =
    match kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  let input = V2_Vector.create V2_Vector.int32 n in
  let histogram = V2_Vector.create V2_Vector.int32 bins in

  for i = 0 to n - 1 do
    V2_Vector.set input i inp.(i)
  done ;
  for i = 0 to bins - 1 do
    V2_Vector.set histogram i 0l
  done ;

  let block_size = min 256 (get_block_size_v2 dev) in
  let num_blocks = min 64 ((n + block_size - 1) / block_size) in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d num_blocks in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec histogram;
        Sarek.Execute.Int32 (Int32.of_int n);
        Sarek.Execute.Int32 (Int32.of_int bins);
      ]
    ~block
    ~grid
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = V2_Vector.to_array histogram in
      let errors = ref 0 in
      for i = 0 to bins - 1 do
        if result.(i) <> exp.(i) then begin
          if !errors < 10 then
            Printf.printf
              "  Bin %d: expected %ld, got %ld\n"
              i
              exp.(i)
              result.(i) ;
          incr errors
        end
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(* ========== Main ========== *)

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

  print_endline "=== Histogram Test (SPOC + V2 Comparison) ===" ;
  Printf.printf "Size: %d elements, %d bins\n\n" cfg.size !num_bins ;

  let spoc_devs = Spoc_Devices.init () in
  if Array.length spoc_devs = 0 then begin
    print_endline "No GPU devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices spoc_devs ;

  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  Printf.printf "\nFound %d V2 device(s)\n\n" (Array.length v2_devs) ;

  ignore (init_histogram_data ()) ;

  if cfg.benchmark_all then begin
    print_endline "=== Simple Histogram ===" ;
    print_endline (String.make 80 '-') ;
    Printf.printf
      "%-35s %10s %10s %8s %8s\n"
      "Device"
      "SPOC(ms)"
      "V2(ms)"
      "SPOC"
      "V2" ;
    print_endline (String.make 80 '-') ;

    let all_ok = ref true in

    Array.iter
      (fun v2_dev ->
        let name = v2_dev.V2_Device.name in
        let framework = v2_dev.V2_Device.framework in

        let spoc_dev_opt =
          Array.find_opt
            (fun d -> d.Spoc_Devices.general_info.Spoc_Devices.name = name)
            spoc_devs
        in

        let spoc_time, spoc_ok =
          match spoc_dev_opt with
          | Some spoc_dev ->
              let time, ok = run_histogram_spoc spoc_dev in
              (Printf.sprintf "%.4f" time, if ok then "OK" else "FAIL")
          | None -> ("-", "SKIP")
        in

        let v2_time, v2_ok = run_histogram_v2 v2_dev in
        let v2_time_str, v2_status =
          if v2_time < 0.0 then ("-", "SKIP")
          else (Printf.sprintf "%.4f" v2_time, if v2_ok then "OK" else "FAIL")
        in

        if spoc_ok = "FAIL" || v2_status = "FAIL" then all_ok := false ;

        Printf.printf
          "%-35s %10s %10s %8s %8s\n"
          (Printf.sprintf "%s (%s)" name framework)
          spoc_time
          v2_time_str
          spoc_ok
          v2_status)
      v2_devs ;

    print_endline (String.make 80 '-') ;

    print_endline "\n=== Strided Histogram ===" ;
    print_endline (String.make 80 '-') ;
    Printf.printf
      "%-35s %10s %10s %8s %8s\n"
      "Device"
      "SPOC(ms)"
      "V2(ms)"
      "SPOC"
      "V2" ;
    print_endline (String.make 80 '-') ;

    (* Re-init data for strided test *)
    ignore (init_histogram_data ()) ;

    Array.iter
      (fun v2_dev ->
        let name = v2_dev.V2_Device.name in
        let framework = v2_dev.V2_Device.framework in

        let spoc_dev_opt =
          Array.find_opt
            (fun d -> d.Spoc_Devices.general_info.Spoc_Devices.name = name)
            spoc_devs
        in

        let spoc_time, spoc_ok =
          match spoc_dev_opt with
          | Some spoc_dev ->
              let time, ok = run_histogram_strided_spoc spoc_dev in
              (Printf.sprintf "%.4f" time, if ok then "OK" else "FAIL")
          | None -> ("-", "SKIP")
        in

        let v2_time, v2_ok = run_histogram_strided_v2 v2_dev in
        let v2_time_str, v2_status =
          if v2_time < 0.0 then ("-", "SKIP")
          else (Printf.sprintf "%.4f" v2_time, if v2_ok then "OK" else "FAIL")
        in

        if spoc_ok = "FAIL" || v2_status = "FAIL" then all_ok := false ;

        Printf.printf
          "%-35s %10s %10s %8s %8s\n"
          (Printf.sprintf "%s (%s)" name framework)
          spoc_time
          v2_time_str
          spoc_ok
          v2_status)
      v2_devs ;

    print_endline (String.make 80 '-') ;

    if !all_ok then print_endline "\n=== All histogram tests PASSED ==="
    else begin
      print_endline "\n=== Some histogram tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    let dev = Test_helpers.get_device cfg spoc_devs in
    let dev_name = dev.Spoc_Devices.general_info.Spoc_Devices.name in
    Printf.printf "Using device: %s\n%!" dev_name ;

    Printf.printf "\n--- Simple Histogram ---\n%!" ;
    Printf.printf "Running SPOC path...\n%!" ;
    let spoc_time, spoc_ok = run_histogram_spoc dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      spoc_time
      (if spoc_ok then "PASSED" else "FAILED") ;

    let v2_dev_opt =
      Array.find_opt (fun d -> d.V2_Device.name = dev_name) v2_devs
    in
    (match v2_dev_opt with
    | Some v2_dev ->
        let t, ok = run_histogram_v2 v2_dev in
        if t < 0.0 then Printf.printf "  V2: SKIP (no V2 IR)\n%!"
        else
          Printf.printf
            "  V2: %.4f ms, %s\n%!"
            t
            (if ok then "PASSED" else "FAILED")
    | None -> Printf.printf "No matching V2 device\n%!") ;

    (* Re-init for strided *)
    ignore (init_histogram_data ()) ;

    Printf.printf "\n--- Strided Histogram ---\n%!" ;
    Printf.printf "Running SPOC path...\n%!" ;
    let spoc_time, spoc_ok = run_histogram_strided_spoc dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      spoc_time
      (if spoc_ok then "PASSED" else "FAILED") ;

    (match v2_dev_opt with
    | Some v2_dev ->
        let t, ok = run_histogram_strided_v2 v2_dev in
        if t < 0.0 then Printf.printf "  V2: SKIP (no V2 IR)\n%!"
        else
          Printf.printf
            "  V2: %.4f ms, %s\n%!"
            t
            (if ok then "PASSED" else "FAILED")
    | None -> Printf.printf "No matching V2 device\n%!") ;

    print_endline "\nHistogram tests PASSED"
  end
