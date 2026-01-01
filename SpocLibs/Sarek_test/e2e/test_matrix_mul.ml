(******************************************************************************
 * E2E test for Sarek PPX - Matrix Multiplication with V2 comparison
 *
 * Tests naive and tiled matrix multiplication with shared memory.
 * Matrix multiplication is the canonical GPU compute benchmark.
 *
 * V2 comparison: naive kernel (1D indexing) only.
 * Tiled kernel (shared memory + supersteps) runs SPOC-only.
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

let tile_size = ref 16

(* ========== Pure OCaml baseline ========== *)

(** Pure OCaml matrix multiplication *)
let ocaml_matmul a b c m n k =
  for row = 0 to m - 1 do
    for col = 0 to n - 1 do
      let sum = ref 0.0 in
      for i = 0 to k - 1 do
        sum := !sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- !sum
    done
  done

(* ========== Shared test data ========== *)

let input_a = ref [||]

let input_b = ref [||]

let expected_c = ref [||]

let matrix_dim = ref 0

(** Initialize matrices and compute expected result *)
let init_matmul_data () =
  let dim = int_of_float (sqrt (float_of_int cfg.size)) in
  matrix_dim := dim ;
  let m, n, k = (dim, dim, dim) in
  let a = Array.init (m * k) (fun i -> float_of_int (i mod 10) /. 10.0) in
  let b = Array.init (k * n) (fun i -> float_of_int ((i + 1) mod 10) /. 10.0) in
  let c = Array.make (m * n) 0.0 in
  input_a := a ;
  input_b := b ;
  expected_c := c ;
  let t0 = Unix.gettimeofday () in
  ocaml_matmul a b c m n k ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Sarek kernels ========== *)

(** Naive matrix multiplication - V2 compatible (1D indexing). Each thread
    computes one output element. *)
let matmul_naive_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let open Std in
      let tid = global_thread_id in
      let row = tid / n in
      let col = tid mod n in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

(* NOTE: Not V2 compatible - uses shared memory and supersteps *)

(** Tiled matrix multiplication with shared memory and supersteps. *)
let matmul_tiled_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let%shared (tile_a : float32) = 256l in
      let%shared (tile_b : float32) = 256l in
      let tx = thread_idx_x in
      let ty = thread_idx_y in
      let row = ty + (block_dim_y * block_idx_y) in
      let col = tx + (block_dim_x * block_idx_x) in
      let tile_size = 16l in
      let num_tiles = (k + tile_size - 1l) / tile_size in
      let sum = mut 0.0 in
      for t = 0 to num_tiles - 1l do
        let%superstep load_a =
          let a_col = (t * tile_size) + tx in
          if row < m && a_col < k then
            tile_a.((ty * tile_size) + tx) <- a.((row * k) + a_col)
          else tile_a.((ty * tile_size) + tx) <- 0.0
        in
        let%superstep load_b =
          let b_row = (t * tile_size) + ty in
          if b_row < k && col < n then
            tile_b.((ty * tile_size) + tx) <- b.((b_row * n) + col)
          else tile_b.((ty * tile_size) + tx) <- 0.0
        in
        let%superstep _compute =
          for i = 0 to tile_size - 1l do
            sum :=
              sum
              +. (tile_a.((ty * tile_size) + i) *. tile_b.((i * tile_size) + tx))
          done
        in
        ()
      done ;
      if row < m && col < n then c.((row * n) + col) <- sum]

(* ========== SPOC test runners ========== *)

(** Run naive matrix multiplication test - returns (compile_ms, exec_ms, ok) *)
let run_matmul_naive_spoc dev =
  let dim = !matrix_dim in
  let m, n, k = (dim, dim, dim) in
  let inp_a = !input_a in
  let inp_b = !input_b in
  let exp_c = !expected_c in

  let a = Spoc_Vector.create Spoc_Vector.float32 (m * k) in
  let b = Spoc_Vector.create Spoc_Vector.float32 (k * n) in
  let c = Spoc_Vector.create Spoc_Vector.float32 (m * n) in

  for i = 0 to (m * k) - 1 do
    Spoc_Mem.set a i inp_a.(i)
  done ;
  for i = 0 to (k * n) - 1 do
    Spoc_Mem.set b i inp_b.(i)
  done ;
  for i = 0 to (m * n) - 1 do
    Spoc_Mem.set c i 0.0
  done ;

  (* Use 1D block/grid for V2 compatibility *)
  let block_size = 256 in
  let total_elements = m * n in
  let blocks = (total_elements + block_size - 1) / block_size in
  let block = {Spoc.Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Spoc.Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  (* Measure compile time separately *)
  let tc0 = Unix.gettimeofday () in
  ignore (Sarek.Kirc.gen matmul_naive_kernel dev) ;
  let tc1 = Unix.gettimeofday () in
  let compile_ms = (tc1 -. tc0) *. 1000.0 in

  (* Measure execution time *)
  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run matmul_naive_kernel (a, b, c, m, n, k) (block, grid) 0 dev ;
  Spoc_Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let exec_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Spoc_Mem.to_cpu c () ;
      Spoc_Devices.flush dev () ;
      Spoc_Mem.unsafe_rw true ;
      let errors = ref 0 in
      let check_count = min 100 (m * n) in
      for idx = 0 to check_count - 1 do
        let expected = exp_c.(idx) in
        let got = Spoc_Mem.get c idx in
        (* Use relative tolerance for float32 accumulation *)
        let rel_tol = 0.001 in
        let abs_tol = 0.1 in
        let diff = abs_float (got -. expected) in
        let rel_err =
          if abs_float expected > 1e-6 then diff /. abs_float expected else diff
        in
        if diff > abs_tol && rel_err > rel_tol then incr errors
      done ;
      Spoc_Mem.unsafe_rw false ;
      !errors = 0
    end
    else true
  in
  (compile_ms, exec_ms, ok)

(** Run tiled matrix multiplication test *)
let run_matmul_tiled_spoc dev =
  let dim = !matrix_dim in
  let dim = (dim + !tile_size - 1) / !tile_size * !tile_size in
  let m, n, k = (dim, dim, dim) in

  (* Need to recompute expected for padded dimension *)
  let inp_a = !input_a in
  let inp_b = !input_b in
  let orig_dim = !matrix_dim in

  let a = Spoc_Vector.create Spoc_Vector.float32 (m * k) in
  let b = Spoc_Vector.create Spoc_Vector.float32 (k * n) in
  let c = Spoc_Vector.create Spoc_Vector.float32 (m * n) in

  (* Initialize with padding *)
  for row = 0 to m - 1 do
    for col = 0 to k - 1 do
      let idx = (row * k) + col in
      if row < orig_dim && col < orig_dim then
        Spoc_Mem.set a idx inp_a.((row * orig_dim) + col)
      else Spoc_Mem.set a idx 0.0
    done
  done ;
  for row = 0 to k - 1 do
    for col = 0 to n - 1 do
      let idx = (row * n) + col in
      if row < orig_dim && col < orig_dim then
        Spoc_Mem.set b idx inp_b.((row * orig_dim) + col)
      else Spoc_Mem.set b idx 0.0
    done
  done ;
  for i = 0 to (m * n) - 1 do
    Spoc_Mem.set c i 0.0
  done ;

  let block_size = !tile_size in
  let blocks_x = n / block_size in
  let blocks_y = m / block_size in
  let block =
    {Spoc.Kernel.blockX = block_size; blockY = block_size; blockZ = 1}
  in
  let grid = {Spoc.Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  (* Pre-compile (excluded from timing) *)
  ignore (Sarek.Kirc.gen matmul_tiled_kernel dev) ;
  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run matmul_tiled_kernel (a, b, c, m, n, k) (block, grid) 0 dev ;
  Spoc_Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Spoc_Mem.to_cpu c () ;
      Spoc_Devices.flush dev () ;
      Spoc_Mem.unsafe_rw true ;
      let errors = ref 0 in
      let exp_c = !expected_c in
      let check_count = min 100 (orig_dim * orig_dim) in
      for idx = 0 to check_count - 1 do
        let row = idx / orig_dim in
        let col = idx mod orig_dim in
        let expected = exp_c.(idx) in
        let got = Spoc_Mem.get c ((row * n) + col) in
        (* Use relative tolerance for float32 accumulation *)
        let rel_tol = 0.001 in
        let abs_tol = 0.1 in
        let diff = abs_float (got -. expected) in
        let rel_err =
          if abs_float expected > 1e-6 then diff /. abs_float expected else diff
        in
        if diff > abs_tol && rel_err > rel_tol then incr errors
      done ;
      Spoc_Mem.unsafe_rw false ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(* ========== V2 test runner ========== *)

(** Run naive matrix multiplication on V2 - returns (compile_ms, exec_ms, ok) *)
let run_matmul_naive_v2 (dev : V2_Device.t) =
  let dim = !matrix_dim in
  let m, n, k = (dim, dim, dim) in
  let inp_a = !input_a in
  let inp_b = !input_b in
  let exp_c = !expected_c in
  let _, kirc = matmul_naive_kernel in
  let ir =
    match kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "No V2 IR"
  in

  let a = V2_Vector.create V2_Vector.float32 (m * k) in
  let b = V2_Vector.create V2_Vector.float32 (k * n) in
  let c = V2_Vector.create V2_Vector.float32 (m * n) in

  for i = 0 to (m * k) - 1 do
    V2_Vector.set a i inp_a.(i)
  done ;
  for i = 0 to (k * n) - 1 do
    V2_Vector.set b i inp_b.(i)
  done ;
  for i = 0 to (m * n) - 1 do
    V2_Vector.set c i 0.0
  done ;

  let block_size = 256 in
  let total_elements = m * n in
  let grid_size = (total_elements + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

  (* Measure compile time (first run triggers JIT) *)
  let tc0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec a;
        Sarek.Execute.Vec b;
        Sarek.Execute.Vec c;
        Sarek.Execute.Int32 (Int32.of_int m);
        Sarek.Execute.Int32 (Int32.of_int n);
        Sarek.Execute.Int32 (Int32.of_int k);
      ]
    ~block
    ~grid
    () ;
  V2_Transfer.flush dev ;
  let tc1 = Unix.gettimeofday () in
  let compile_ms = (tc1 -. tc0) *. 1000.0 in

  (* Reset output for execution timing *)
  for i = 0 to (m * n) - 1 do
    V2_Vector.set c i 0.0
  done ;

  (* Measure execution time (cached kernel) *)
  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec a;
        Sarek.Execute.Vec b;
        Sarek.Execute.Vec c;
        Sarek.Execute.Int32 (Int32.of_int m);
        Sarek.Execute.Int32 (Int32.of_int n);
        Sarek.Execute.Int32 (Int32.of_int k);
      ]
    ~block
    ~grid
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let exec_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = V2_Vector.to_array c in
      let errors = ref 0 in
      let check_count = min 100 (m * n) in
      for idx = 0 to check_count - 1 do
        let expected = exp_c.(idx) in
        let got = result.(idx) in
        (* Use relative tolerance for float32 accumulation across k iterations *)
        let rel_tol = 0.001 in
        (* 0.1% relative error *)
        let abs_tol = 0.1 in
        (* Also allow small absolute error *)
        let diff = abs_float (got -. expected) in
        let rel_err =
          if abs_float expected > 1e-6 then diff /. abs_float expected else diff
        in
        if diff > abs_tol && rel_err > rel_tol then begin
          if !errors < 5 then
            Printf.printf
              "  V2 mismatch at %d: expected %.6f, got %.6f (rel_err=%.4f%%)\n"
              idx
              expected
              got
              (rel_err *. 100.0) ;
          incr errors
        end
      done ;
      !errors = 0
    end
    else true
  in
  (compile_ms, exec_ms, ok)

(* ========== Main ========== *)

let () =
  let c = Test_helpers.parse_args "test_matrix_mul" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  print_endline "=== Matrix Multiplication Test (SPOC + V2 Comparison) ===" ;

  let dim = int_of_float (sqrt (float_of_int cfg.size)) in
  Printf.printf
    "Matrix dimensions: %dx%d (total elements: %d)\n\n"
    dim
    dim
    (dim * dim) ;

  let spoc_devs = Spoc_Devices.init () in
  if Array.length spoc_devs = 0 then begin
    print_endline "No GPU devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices spoc_devs ;

  let v2_devs = V2_Device.init ~frameworks:["CUDA"; "OpenCL"] () in
  Printf.printf "\nFound %d V2 device(s)\n\n" (Array.length v2_devs) ;

  ignore (init_matmul_data ()) ;

  if cfg.benchmark_all then begin
    (* Benchmark naive *)
    print_endline "=== Naive Matrix Multiplication ===" ;
    print_endline (String.make 120 '-') ;
    Printf.printf
      "%-35s %12s %12s %12s %12s %6s %6s\n"
      "Device"
      "SPOC Comp"
      "SPOC Exec"
      "V2 Comp"
      "V2 Exec"
      "SPOC"
      "V2" ;
    print_endline (String.make 120 '-') ;

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

        let spoc_comp, spoc_exec, spoc_ok =
          match spoc_dev_opt with
          | Some spoc_dev ->
              let comp, exec, ok = run_matmul_naive_spoc spoc_dev in
              ( Printf.sprintf "%.2f" comp,
                Printf.sprintf "%.2f" exec,
                if ok then "OK" else "FAIL" )
          | None -> ("-", "-", "SKIP")
        in

        let v2_comp, v2_exec, v2_ok = run_matmul_naive_v2 v2_dev in
        let v2_status = if v2_ok then "OK" else "FAIL" in

        if not v2_ok then all_ok := false ;
        if spoc_ok = "FAIL" then all_ok := false ;

        Printf.printf
          "%-35s %12s %12s %12.2f %12.2f %6s %6s\n"
          (Printf.sprintf "%s (%s)" name framework)
          spoc_comp
          spoc_exec
          v2_comp
          v2_exec
          spoc_ok
          v2_status)
      v2_devs ;

    print_endline (String.make 120 '-') ;

    (* Benchmark tiled (SPOC only) *)
    print_endline "\n=== Tiled Matrix Multiplication (SPOC only) ===" ;
    print_endline (String.make 60 '-') ;
    Printf.printf "%-35s %10s %8s\n" "Device" "SPOC(ms)" "Status" ;
    print_endline (String.make 60 '-') ;

    Array.iter
      (fun spoc_dev ->
        let name = spoc_dev.Spoc_Devices.general_info.Spoc_Devices.name in
        let time, ok = run_matmul_tiled_spoc spoc_dev in
        Printf.printf
          "%-35s %10.4f %8s\n"
          name
          time
          (if ok then "OK" else "FAIL") ;
        if not ok then all_ok := false)
      spoc_devs ;

    print_endline (String.make 60 '-') ;

    if !all_ok then
      print_endline "\n=== All matrix multiplication tests PASSED ==="
    else begin
      print_endline "\n=== Some matrix multiplication tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    let dev = Test_helpers.get_device cfg spoc_devs in
    let dev_name = dev.Spoc_Devices.general_info.Spoc_Devices.name in
    Printf.printf "Using device: %s\n%!" dev_name ;

    Printf.printf "\n--- Naive Matrix Multiplication ---\n%!" ;
    Printf.printf "Running SPOC path...\n%!" ;
    let spoc_comp, spoc_exec, spoc_ok = run_matmul_naive_spoc dev in
    Printf.printf
      "  Compile: %.2f ms, Exec: %.2f ms, %s\n%!"
      spoc_comp
      spoc_exec
      (if spoc_ok then "PASSED" else "FAILED") ;

    let v2_dev_opt =
      Array.find_opt (fun d -> d.V2_Device.name = dev_name) v2_devs
    in
    (match v2_dev_opt with
    | Some v2_dev ->
        Printf.printf "Running V2 path...\n%!" ;
        let v2_comp, v2_exec, v2_ok = run_matmul_naive_v2 v2_dev in
        Printf.printf
          "  Compile: %.2f ms, Exec: %.2f ms, %s\n%!"
          v2_comp
          v2_exec
          (if v2_ok then "PASSED" else "FAILED")
    | None -> Printf.printf "No matching V2 device\n%!") ;

    Printf.printf "\n--- Tiled Matrix Multiplication (SPOC only) ---\n%!" ;
    let time, ok = run_matmul_tiled_spoc dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nMatrix multiplication tests PASSED"
  end
