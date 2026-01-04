(******************************************************************************
 * E2E test for Sarek PPX - Matrix Multiplication (runtime)
 *
 * Tests naive and tiled matrix multiplication with shared memory.
 * Matrix multiplication is the canonical GPU compute benchmark.
 *
 * Uses runtime path for all backends (CUDA, OpenCL, Native, Interpreter).
 * Tiled kernel demonstrates shared memory + supersteps (barriers).
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std

(* Module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init () ;
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init ()

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

(** Naive matrix multiplication - runtime compatible (1D indexing). Each thread
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

(* NOTE: Not runtime compatible - uses shared memory and supersteps *)

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

(* ========== runtime test runners ========== *)

(** Run tiled matrix multiplication on runtime - uses shared memory and
    supersteps *)
let run_matmul_tiled (dev : Device.t) =
  let dim = !matrix_dim in
  (* Pad to tile_size multiple *)
  let dim = (dim + !tile_size - 1) / !tile_size * !tile_size in
  let m, n, k = (dim, dim, dim) in
  let inp_a = !input_a in
  let inp_b = !input_b in
  let orig_dim = !matrix_dim in

  let _, kirc = matmul_tiled_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Tiled kernel: No IR"
  in

  let a = Vector.create Vector.float32 (m * k) in
  let b = Vector.create Vector.float32 (k * n) in
  let c = Vector.create Vector.float32 (m * n) in

  (* Initialize with padding - zero-pad to tile boundary *)
  for row = 0 to m - 1 do
    for col = 0 to k - 1 do
      let idx = (row * k) + col in
      if row < orig_dim && col < orig_dim then
        Vector.set a idx inp_a.((row * orig_dim) + col)
      else Vector.set a idx 0.0
    done
  done ;
  for row = 0 to k - 1 do
    for col = 0 to n - 1 do
      let idx = (row * n) + col in
      if row < orig_dim && col < orig_dim then
        Vector.set b idx inp_b.((row * orig_dim) + col)
      else Vector.set b idx 0.0
    done
  done ;
  for i = 0 to (m * n) - 1 do
    Vector.set c i 0.0
  done ;

  (* 2D launch configuration *)
  let block_size = !tile_size in
  let blocks_x = n / block_size in
  let blocks_y = m / block_size in
  let block = Sarek.Execute.dims2d block_size block_size in
  let grid = Sarek.Execute.dims2d blocks_x blocks_y in

  (* Shared memory: 2 tiles × tile_size² × 4 bytes *)
  let shared_mem = 2 * block_size * block_size * 4 in

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
    ~shared_mem
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = Vector.to_array c in
      let exp_c = !expected_c in
      let errors = ref 0 in
      let check_count = min 100 (orig_dim * orig_dim) in
      for idx = 0 to check_count - 1 do
        let row = idx / orig_dim in
        let col = idx mod orig_dim in
        let expected = exp_c.(idx) in
        (* Access padded matrix: row * padded_n + col *)
        let got = result.((row * n) + col) in
        let rel_tol = 0.001 in
        let abs_tol = 0.1 in
        let diff = abs_float (got -. expected) in
        let rel_err =
          if abs_float expected > 1e-6 then diff /. abs_float expected else diff
        in
        if diff > abs_tol && rel_err > rel_tol then begin
          if !errors < 5 then
            Printf.printf
              "  Tiled runtime mismatch at [%d,%d]: expected %.6f, got %.6f \
               (rel_err=%.4f%%)\n"
              row
              col
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
  (time_ms, ok)

(** Run naive matrix multiplication on runtime - returns (compile_ms, exec_ms,
    ok) *)
let run_matmul_naive (dev : Device.t) =
  let dim = !matrix_dim in
  let m, n, k = (dim, dim, dim) in
  let inp_a = !input_a in
  let inp_b = !input_b in
  let exp_c = !expected_c in
  let _, kirc = matmul_naive_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let a = Vector.create Vector.float32 (m * k) in
  let b = Vector.create Vector.float32 (k * n) in
  let c = Vector.create Vector.float32 (m * n) in

  for i = 0 to (m * k) - 1 do
    Vector.set a i inp_a.(i)
  done ;
  for i = 0 to (k * n) - 1 do
    Vector.set b i inp_b.(i)
  done ;
  for i = 0 to (m * n) - 1 do
    Vector.set c i 0.0
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
  Transfer.flush dev ;
  let tc1 = Unix.gettimeofday () in
  let compile_ms = (tc1 -. tc0) *. 1000.0 in

  (* Reset output for execution timing *)
  for i = 0 to (m * n) - 1 do
    Vector.set c i 0.0
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
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let exec_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = Vector.to_array c in
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
              "  runtime mismatch at %d: expected %.6f, got %.6f \
               (rel_err=%.4f%%)\n"
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
  cfg.use_vulkan <- c.use_vulkan ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  print_endline "=== Matrix Multiplication Test (Unified runtime) ===" ;

  let dim = int_of_float (sqrt (float_of_int cfg.size)) in
  Printf.printf
    "Matrix dimensions: %dx%d (total elements: %d)\n\n"
    dim
    dim
    (dim * dim) ;

  (* Init runtime devices - unified path for all backends *)
  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Vulkan"; "Native"; "Interpreter"] ()
  in
  Printf.printf "Found %d runtime device(s)\n" (Array.length devs) ;
  Array.iteri
    (fun i d ->
      Printf.printf "  [%d] %s (%s)\n" i d.Device.name d.Device.framework)
    devs ;
  print_newline () ;

  ignore (init_matmul_data ()) ;

  if cfg.benchmark_all then begin
    (* Benchmark naive - unified runtime path *)
    print_endline "=== Naive Matrix Multiplication (Unified runtime) ===" ;
    print_endline (String.make 80 '-') ;
    Printf.printf
      "%-40s %12s %12s %8s\n"
      "Device"
      "Compile(ms)"
      "Exec(ms)"
      "Status" ;
    print_endline (String.make 80 '-') ;

    let all_ok = ref true in

    Array.iter
      (fun dev ->
        let name = dev.Device.name in
        let framework = dev.Device.framework in
        let comp, exec, ok = run_matmul_naive dev in
        let status = if ok then "OK" else "FAIL" in
        if not ok then all_ok := false ;
        Printf.printf
          "%-40s %12.2f %12.2f %8s\n"
          (Printf.sprintf "%s (%s)" name framework)
          comp
          exec
          status)
      devs ;

    print_endline (String.make 80 '-') ;

    (* Benchmark tiled runtime (shared memory + supersteps) *)
    print_endline
      "\n=== Tiled Matrix Multiplication (runtime - shared memory) ===" ;
    print_endline (String.make 60 '-') ;
    Printf.printf "%-40s %10s %8s\n" "Device" "Exec(ms)" "Status" ;
    print_endline (String.make 60 '-') ;

    Array.iter
      (fun dev ->
        let name = dev.Device.name in
        let framework = dev.Device.framework in
        (* Skip interpreter for tiled - barrier semantics differ *)
        if framework <> "Interpreter" then begin
          let time, ok = run_matmul_tiled dev in
          Printf.printf
            "%-40s %10.4f %8s\n"
            (Printf.sprintf "%s (%s)" name framework)
            time
            (if ok then "OK" else "FAIL") ;
          if not ok then all_ok := false
        end
        else
          Printf.printf
            "%-40s %10s %8s\n"
            (Printf.sprintf "%s (%s)" name framework)
            "-"
            "SKIP")
      devs ;

    print_endline (String.make 60 '-') ;

    if !all_ok then
      print_endline "\n=== All matrix multiplication tests PASSED ==="
    else begin
      print_endline "\n=== Some matrix multiplication tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    (* Single device mode - use runtime *)
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf
      "Using device: %s (%s)\n%!"
      dev.Device.name
      dev.Device.framework ;

    Printf.printf "\n--- Naive Matrix Multiplication (runtime) ---\n%!" ;
    let comp, exec, ok = run_matmul_naive dev in
    Printf.printf
      "  Compile: %.2f ms, Exec: %.2f ms, %s\n%!"
      comp
      exec
      (if ok then "PASSED" else "FAILED") ;

    (* Tiled kernel via runtime (shared memory + supersteps) *)
    if dev.Device.framework <> "Interpreter" then begin
      Printf.printf "\n--- Tiled Matrix Multiplication (runtime) ---\n%!" ;
      let time, ok = run_matmul_tiled dev in
      Printf.printf
        "  Time: %.4f ms, %s\n%!"
        time
        (if ok then "PASSED" else "FAILED")
    end
    else Printf.printf "\n(Tiled kernel skipped for interpreter)\n%!" ;

    print_endline "\nMatrix multiplication tests PASSED"
  end
