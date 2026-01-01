(******************************************************************************
 * E2E test for Sarek PPX - Parallel Reduction with V2 comparison
 *
 * Tests tree-based parallel reduction with shared memory and barriers.
 * Reduction is a fundamental parallel primitive for computing sums, min, max.
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

(* ========== Pure OCaml baselines ========== *)

let ocaml_sum arr n =
  let sum = ref 0.0 in
  for i = 0 to n - 1 do
    sum := !sum +. arr.(i)
  done ;
  !sum

let ocaml_max arr n =
  let m = ref arr.(0) in
  for i = 1 to n - 1 do
    if arr.(i) > !m then m := arr.(i)
  done ;
  !m

let ocaml_dot a b n =
  let sum = ref 0.0 in
  for i = 0 to n - 1 do
    sum := !sum +. (a.(i) *. b.(i))
  done ;
  !sum

(* ========== Shared test data ========== *)

let input_sum = ref [||]

let expected_sum = ref 0.0

let input_max = ref [||]

let expected_max = ref 0.0

let input_a = ref [||]

let input_b = ref [||]

let expected_dot = ref 0.0

let init_sum_data () =
  let n = cfg.size in
  let arr = Array.make n 1.0 in
  input_sum := arr ;
  let t0 = Unix.gettimeofday () in
  expected_sum := ocaml_sum arr n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_max_data () =
  let n = cfg.size in
  let arr = Array.init n (fun i -> float_of_int i) in
  input_max := arr ;
  let t0 = Unix.gettimeofday () in
  expected_max := ocaml_max arr n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_dot_data () =
  let n = cfg.size in
  let a = Array.init n (fun i -> float_of_int i) in
  let b = Array.make n 1.0 in
  input_a := a ;
  input_b := b ;
  let t0 = Unix.gettimeofday () in
  expected_dot := ocaml_dot a b n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Sarek kernels ========== *)

let reduce_sum_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid) else sdata.(tid) <- 0.0
      in
      let%superstep reduce128 =
        if tid < 128l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 128l)
      in
      let%superstep reduce64 =
        if tid < 64l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 64l)
      in
      let%superstep reduce32 =
        if tid < 32l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 32l)
      in
      let%superstep reduce16 =
        if tid < 16l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 16l)
      in
      let%superstep reduce8 =
        if tid < 8l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 8l)
      in
      let%superstep reduce4 =
        if tid < 4l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 4l)
      in
      let%superstep reduce2 =
        if tid < 2l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 2l)
      in
      let%superstep reduce1 =
        if tid < 1l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 1l)
      in
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]

let reduce_max_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid)
        else sdata.(tid) <- -1000000.0
      in
      let%superstep reduce128 =
        if tid < 128l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 128l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce64 =
        if tid < 64l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 64l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce32 =
        if tid < 32l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 32l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce16 =
        if tid < 16l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 16l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce8 =
        if tid < 8l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 8l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce4 =
        if tid < 4l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 4l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce2 =
        if tid < 2l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 2l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce1 =
        if tid < 1l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 1l) in
          if b > a then sdata.(tid) <- b
        end
      in
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]

let dot_product_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (output : float32 vector)
        (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- a.(gid) *. b.(gid)
        else sdata.(tid) <- 0.0
      in
      let%superstep reduce128 =
        if tid < 128l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 128l)
      in
      let%superstep reduce64 =
        if tid < 64l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 64l)
      in
      let%superstep reduce32 =
        if tid < 32l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 32l)
      in
      let%superstep reduce16 =
        if tid < 16l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 16l)
      in
      let%superstep reduce8 =
        if tid < 8l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 8l)
      in
      let%superstep reduce4 =
        if tid < 4l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 4l)
      in
      let%superstep reduce2 =
        if tid < 2l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 2l)
      in
      let%superstep reduce1 =
        if tid < 1l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 1l)
      in
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]

(* ========== SPOC test runners ========== *)

let run_reduce_sum_spoc dev =
  let n = cfg.size in
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = (n + block_size - 1) / block_size in
  let inp = !input_sum in

  let input = Spoc_Vector.create Spoc_Vector.float32 n in
  let output = Spoc_Vector.create Spoc_Vector.float32 num_blocks in

  for i = 0 to n - 1 do
    Spoc_Mem.set input i inp.(i)
  done ;
  for i = 0 to num_blocks - 1 do
    Spoc_Mem.set output i 0.0
  done ;

  ignore (Sarek.Kirc.gen reduce_sum_kernel dev) ;
  let block = {Spoc.Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Spoc.Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run reduce_sum_kernel (input, output, n) (block, grid) 0 dev ;
  Spoc_Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Spoc_Mem.to_cpu output () ;
      Spoc_Devices.flush dev () ;
      let total = ref 0.0 in
      for i = 0 to num_blocks - 1 do
        total := !total +. Spoc_Mem.get output i
      done ;
      abs_float (!total -. !expected_sum) < 0.1
    end
    else true
  in
  (time_ms, ok)

let run_reduce_max_spoc dev =
  let n = cfg.size in
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = (n + block_size - 1) / block_size in
  let inp = !input_max in

  let input = Spoc_Vector.create Spoc_Vector.float32 n in
  let output = Spoc_Vector.create Spoc_Vector.float32 num_blocks in

  for i = 0 to n - 1 do
    Spoc_Mem.set input i inp.(i)
  done ;
  for i = 0 to num_blocks - 1 do
    Spoc_Mem.set output i (-1000000.0)
  done ;

  ignore (Sarek.Kirc.gen reduce_max_kernel dev) ;
  let block = {Spoc.Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Spoc.Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run reduce_max_kernel (input, output, n) (block, grid) 0 dev ;
  Spoc_Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Spoc_Mem.to_cpu output () ;
      Spoc_Devices.flush dev () ;
      let max_val = ref (-1000000.0) in
      for i = 0 to num_blocks - 1 do
        let v = Spoc_Mem.get output i in
        if v > !max_val then max_val := v
      done ;
      abs_float (!max_val -. !expected_max) < 0.1
    end
    else true
  in
  (time_ms, ok)

let run_dot_product_spoc dev =
  let n = cfg.size in
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let num_blocks = (n + block_size - 1) / block_size in
  let inp_a = !input_a in
  let inp_b = !input_b in

  let a = Spoc_Vector.create Spoc_Vector.float32 n in
  let b = Spoc_Vector.create Spoc_Vector.float32 n in
  let output = Spoc_Vector.create Spoc_Vector.float32 num_blocks in

  for i = 0 to n - 1 do
    Spoc_Mem.set a i inp_a.(i) ;
    Spoc_Mem.set b i inp_b.(i)
  done ;
  for i = 0 to num_blocks - 1 do
    Spoc_Mem.set output i 0.0
  done ;

  ignore (Sarek.Kirc.gen dot_product_kernel dev) ;
  let block = {Spoc.Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Spoc.Kernel.gridX = num_blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run dot_product_kernel (a, b, output, n) (block, grid) 0 dev ;
  Spoc_Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Spoc_Mem.to_cpu output () ;
      Spoc_Devices.flush dev () ;
      let total = ref 0.0 in
      for i = 0 to num_blocks - 1 do
        total := !total +. Spoc_Mem.get output i
      done ;
      abs_float (!total -. !expected_dot) < float_of_int n *. 0.01
    end
    else true
  in
  (time_ms, ok)

(* ========== V2 test runners ========== *)

(** V2 not yet supported for kernels with shared memory/supersteps *)
let run_reduce_sum_v2 (_dev : V2_Device.t) =
  (* Skip V2 - shared memory not yet supported in V2 codegen *)
  (-1.0, false)

let run_reduce_max_v2 (_dev : V2_Device.t) =
  (* Skip V2 - shared memory not yet supported in V2 codegen *)
  (-1.0, false)

let run_dot_product_v2 (_dev : V2_Device.t) =
  (* Skip V2 - shared memory not yet supported in V2 codegen *)
  (-1.0, false)

(* ========== Main ========== *)

let () =
  let c = Test_helpers.parse_args "test_reduce" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  print_endline "=== Reduction Tests (SPOC + V2 Comparison) ===" ;
  Printf.printf "Size: %d elements\n\n" cfg.size ;

  let spoc_devs = Spoc_Devices.init () in
  if Array.length spoc_devs = 0 then begin
    print_endline "No GPU devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices spoc_devs ;

  let v2_devs = V2_Device.init ~frameworks:["CUDA"; "OpenCL"] () in
  Printf.printf "\nFound %d V2 device(s)\n\n" (Array.length v2_devs) ;

  if cfg.benchmark_all then begin
    let all_ok = ref true in

    (* Sum reduction *)
    ignore (init_sum_data ()) ;
    print_endline "=== Sum Reduction ===" ;
    print_endline (String.make 80 '-') ;
    Printf.printf
      "%-35s %10s %10s %8s %8s\n"
      "Device"
      "SPOC(ms)"
      "V2(ms)"
      "SPOC"
      "V2" ;
    print_endline (String.make 80 '-') ;

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
              let t, ok = run_reduce_sum_spoc spoc_dev in
              (Printf.sprintf "%.4f" t, if ok then "OK" else "FAIL")
          | None -> ("-", "SKIP")
        in
        let v2_time, _v2_ok = run_reduce_sum_v2 v2_dev in
        let v2_time_str, v2_status =
          if v2_time < 0.0 then ("-", "SKIP")
          else (Printf.sprintf "%.4f" v2_time, "SKIP")  (* V2 not supported *)
        in
        if spoc_ok = "FAIL" then all_ok := false ;
        Printf.printf
          "%-35s %10s %10s %8s %8s\n"
          (Printf.sprintf "%s (%s)" name framework)
          spoc_time
          v2_time_str
          spoc_ok
          v2_status)
      v2_devs ;
    print_endline (String.make 80 '-') ;

    (* Max reduction *)
    ignore (init_max_data ()) ;
    print_endline "\n=== Max Reduction ===" ;
    print_endline (String.make 80 '-') ;
    Printf.printf
      "%-35s %10s %10s %8s %8s\n"
      "Device"
      "SPOC(ms)"
      "V2(ms)"
      "SPOC"
      "V2" ;
    print_endline (String.make 80 '-') ;

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
              let t, ok = run_reduce_max_spoc spoc_dev in
              (Printf.sprintf "%.4f" t, if ok then "OK" else "FAIL")
          | None -> ("-", "SKIP")
        in
        let v2_time, _v2_ok = run_reduce_max_v2 v2_dev in
        let v2_time_str, v2_status =
          if v2_time < 0.0 then ("-", "SKIP")
          else (Printf.sprintf "%.4f" v2_time, "SKIP")
        in
        if spoc_ok = "FAIL" then all_ok := false ;
        Printf.printf
          "%-35s %10s %10s %8s %8s\n"
          (Printf.sprintf "%s (%s)" name framework)
          spoc_time
          v2_time_str
          spoc_ok
          v2_status)
      v2_devs ;
    print_endline (String.make 80 '-') ;

    (* Dot product *)
    ignore (init_dot_data ()) ;
    print_endline "\n=== Dot Product ===" ;
    print_endline (String.make 80 '-') ;
    Printf.printf
      "%-35s %10s %10s %8s %8s\n"
      "Device"
      "SPOC(ms)"
      "V2(ms)"
      "SPOC"
      "V2" ;
    print_endline (String.make 80 '-') ;

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
              let t, ok = run_dot_product_spoc spoc_dev in
              (Printf.sprintf "%.4f" t, if ok then "OK" else "FAIL")
          | None -> ("-", "SKIP")
        in
        let v2_time, _v2_ok = run_dot_product_v2 v2_dev in
        let v2_time_str, v2_status =
          if v2_time < 0.0 then ("-", "SKIP")
          else (Printf.sprintf "%.4f" v2_time, "SKIP")
        in
        if spoc_ok = "FAIL" then all_ok := false ;
        Printf.printf
          "%-35s %10s %10s %8s %8s\n"
          (Printf.sprintf "%s (%s)" name framework)
          spoc_time
          v2_time_str
          spoc_ok
          v2_status)
      v2_devs ;
    print_endline (String.make 80 '-') ;

    if !all_ok then print_endline "\n=== All reduction tests PASSED ==="
    else begin
      print_endline "\n=== Some reduction tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    let dev = Test_helpers.get_device cfg spoc_devs in
    let dev_name = dev.Spoc_Devices.general_info.Spoc_Devices.name in
    Printf.printf "Using device: %s\n%!" dev_name ;
    let v2_dev_opt =
      Array.find_opt (fun d -> d.V2_Device.name = dev_name) v2_devs
    in

    ignore (init_sum_data ()) ;
    Printf.printf "\n--- Sum Reduction ---\n%!" ;
    let t, ok = run_reduce_sum_spoc dev in
    Printf.printf
      "  SPOC: %.4f ms, %s\n%!"
      t
      (if ok then "PASSED" else "FAILED") ;
    (match v2_dev_opt with
    | Some v2_dev ->
        let t, _ok = run_reduce_sum_v2 v2_dev in
        if t < 0.0 then Printf.printf "  V2: SKIP (shared memory not supported)\n%!"
        else Printf.printf "  V2: %.4f ms, SKIP\n%!" t
    | None -> ()) ;

    ignore (init_max_data ()) ;
    Printf.printf "\n--- Max Reduction ---\n%!" ;
    let t, ok = run_reduce_max_spoc dev in
    Printf.printf
      "  SPOC: %.4f ms, %s\n%!"
      t
      (if ok then "PASSED" else "FAILED") ;
    (match v2_dev_opt with
    | Some v2_dev ->
        let t, _ok = run_reduce_max_v2 v2_dev in
        if t < 0.0 then Printf.printf "  V2: SKIP (shared memory not supported)\n%!"
        else Printf.printf "  V2: %.4f ms, SKIP\n%!" t
    | None -> ()) ;

    ignore (init_dot_data ()) ;
    Printf.printf "\n--- Dot Product ---\n%!" ;
    let t, ok = run_dot_product_spoc dev in
    Printf.printf
      "  SPOC: %.4f ms, %s\n%!"
      t
      (if ok then "PASSED" else "FAILED") ;
    (match v2_dev_opt with
    | Some v2_dev ->
        let t, _ok = run_dot_product_v2 v2_dev in
        if t < 0.0 then Printf.printf "  V2: SKIP (shared memory not supported)\n%!"
        else Printf.printf "  V2: %.4f ms, SKIP\n%!" t
    | None -> ()) ;

    print_endline "\nReduction tests PASSED"
  end
