(******************************************************************************
 * E2E test for Sarek PPX - Parallel Reduction
 *
 * Tests tree-based parallel reduction with shared memory and barriers.
 * Reduction is a fundamental parallel primitive for computing sums, min, max.
 * V2 runtime only.
 ******************************************************************************)

(* Module aliases *)
module V2_Device = Spoc_core.Device
module V2_Vector = Spoc_core.Vector
module V2_Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

let cfg = Test_helpers.default_config ()

(** Get appropriate block size for V2 device *)
let get_block_size_v2 (dev : V2_Device.t) =
  if dev.capabilities.is_cpu then
    if cfg.block_size > 1 then cfg.block_size else 64
  else cfg.block_size

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

(* ========== V2 test runners ========== *)

let run_reduce_sum_v2 (dev : V2_Device.t) =
  let n = cfg.size in
  let block_size = min 256 (get_block_size_v2 dev) in
  let num_blocks = (n + block_size - 1) / block_size in
  let inp = !input_sum in

  let _, kirc = reduce_sum_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  let input = V2_Vector.create V2_Vector.float32 n in
  let output = V2_Vector.create V2_Vector.float32 num_blocks in

  for i = 0 to n - 1 do
    V2_Vector.set input i inp.(i)
  done ;
  for i = 0 to num_blocks - 1 do
    V2_Vector.set output i 0.0
  done ;

  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d num_blocks in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    ~block
    ~grid
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = V2_Vector.to_array output in
      let total = Array.fold_left ( +. ) 0.0 result in
      abs_float (total -. !expected_sum) < 0.1
    end
    else true
  in
  (time_ms, ok)

let run_reduce_max_v2 (dev : V2_Device.t) =
  let n = cfg.size in
  let block_size = min 256 (get_block_size_v2 dev) in
  let num_blocks = (n + block_size - 1) / block_size in
  let inp = !input_max in

  let _, kirc = reduce_max_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  let input = V2_Vector.create V2_Vector.float32 n in
  let output = V2_Vector.create V2_Vector.float32 num_blocks in

  for i = 0 to n - 1 do
    V2_Vector.set input i inp.(i)
  done ;
  for i = 0 to num_blocks - 1 do
    V2_Vector.set output i (-1000000.0)
  done ;

  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d num_blocks in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    ~block
    ~grid
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = V2_Vector.to_array output in
      let max_val = Array.fold_left max (-1000000.0) result in
      abs_float (max_val -. !expected_max) < 0.1
    end
    else true
  in
  (time_ms, ok)

let run_dot_product_v2 (dev : V2_Device.t) =
  let n = cfg.size in
  let block_size = min 256 (get_block_size_v2 dev) in
  let num_blocks = (n + block_size - 1) / block_size in
  let inp_a = !input_a in
  let inp_b = !input_b in

  let _, kirc = dot_product_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  let a = V2_Vector.create V2_Vector.float32 n in
  let b = V2_Vector.create V2_Vector.float32 n in
  let output = V2_Vector.create V2_Vector.float32 num_blocks in

  for i = 0 to n - 1 do
    V2_Vector.set a i inp_a.(i) ;
    V2_Vector.set b i inp_b.(i)
  done ;
  for i = 0 to num_blocks - 1 do
    V2_Vector.set output i 0.0
  done ;

  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d num_blocks in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec a;
        Sarek.Execute.Vec b;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    ~block
    ~grid
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = V2_Vector.to_array output in
      let total = Array.fold_left ( +. ) 0.0 result in
      abs_float (total -. !expected_dot) < float_of_int n *. 0.01
    end
    else true
  in
  (time_ms, ok)

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

  print_endline "=== Reduction Tests (V2) ===" ;
  Printf.printf "Size: %d elements\n\n" cfg.size ;

  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length v2_devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices v2_devs ;
  Printf.printf "\nFound %d V2 device(s)\n\n" (Array.length v2_devs) ;

  let all_ok = ref true in

  (* Sum reduction *)
  ignore (init_sum_data ()) ;
  print_endline "=== Sum Reduction ===" ;
  Array.iter
    (fun v2_dev ->
      let name = v2_dev.V2_Device.name in
      let framework = v2_dev.V2_Device.framework in
      let v2_time, v2_ok = run_reduce_sum_v2 v2_dev in
      Printf.printf
        "  %s (%s): %.4f ms, %s\n%!"
        name
        framework
        v2_time
        (if v2_ok then "OK" else "FAIL") ;
      if not v2_ok then all_ok := false)
    v2_devs ;

  (* Max reduction *)
  ignore (init_max_data ()) ;
  print_endline "\n=== Max Reduction ===" ;
  Array.iter
    (fun v2_dev ->
      let name = v2_dev.V2_Device.name in
      let framework = v2_dev.V2_Device.framework in
      let v2_time, v2_ok = run_reduce_max_v2 v2_dev in
      Printf.printf
        "  %s (%s): %.4f ms, %s\n%!"
        name
        framework
        v2_time
        (if v2_ok then "OK" else "FAIL") ;
      if not v2_ok then all_ok := false)
    v2_devs ;

  (* Dot product *)
  ignore (init_dot_data ()) ;
  print_endline "\n=== Dot Product ===" ;
  Array.iter
    (fun v2_dev ->
      let name = v2_dev.V2_Device.name in
      let framework = v2_dev.V2_Device.framework in
      let v2_time, v2_ok = run_dot_product_v2 v2_dev in
      Printf.printf
        "  %s (%s): %.4f ms, %s\n%!"
        name
        framework
        v2_time
        (if v2_ok then "OK" else "FAIL") ;
      if not v2_ok then all_ok := false)
    v2_devs ;

  if !all_ok then print_endline "\n=== All reduction tests PASSED ==="
  else begin
    print_endline "\n=== Some reduction tests FAILED ===" ;
    exit 1
  end
