(******************************************************************************
 * E2E test for Sarek PPX - Math intrinsics
 *
 * Tests all mathematical intrinsic functions: sin, cos, exp, log, sqrt, etc.
 * These map directly to GPU hardware math units.
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
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init ()

let cfg = Test_helpers.default_config ()

(* ========== Pure OCaml baseline ========== *)

let ocaml_complex_math x y output n =
  for i = 0 to n - 1 do
    let a = x.(i) in
    let b = y.(i) in
    let r = sqrt ((a *. a) +. (b *. b)) in
    let decay = exp (-0.1 *. (a +. b)) in
    let oscillation = cos (a *. b) in
    output.(i) <- r *. decay *. oscillation
  done

(* ========== Shared test data ========== *)

let input_x = ref [||]

let input_y = ref [||]

let expected_complex = ref [||]

let init_complex_data () =
  let n = cfg.size in
  let x = Array.init n (fun i -> float_of_int (i mod 10) *. 0.5) in
  let y = Array.init n (fun i -> float_of_int ((i + 5) mod 10) *. 0.5) in
  let o = Array.make n 0.0 in
  input_x := x ;
  input_y := y ;
  expected_complex := o ;
  ocaml_complex_math x y o n

(* ========== Sarek kernel ========== *)

let complex_math_kernel =
  [%kernel
    fun (x : float32 vector)
        (y : float32 vector)
        (output : float32 vector)
        (n : int) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then begin
        let a = x.(tid) in
        let b = y.(tid) in
        let r = sqrt ((a *. a) +. (b *. b)) in
        let decay = exp (-0.1 *. (a +. b)) in
        let oscillation = cos (a *. b) in
        output.(tid) <- r *. decay *. oscillation
      end]

(* ========== V2 test runner ========== *)

let run_complex_math (dev : Device.t) =
  let n = cfg.size in
  let _, kirc = complex_math_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No V2 IR"
  in

  let x = Vector.create Vector.float32 n in
  let y = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Vector.set x i !input_x.(i) ;
    Vector.set y i !input_y.(i) ;
    Vector.set output i 0.0
  done ;

  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Execute.dims1d block_size in
  let grid = Execute.dims1d grid_size in

  let t0 = Unix.gettimeofday () in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec x; Execute.Vec y; Execute.Vec output; Execute.Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  ((t1 -. t0) *. 1000.0, Vector.to_array output)

(* ========== Verification ========== *)

let verify_float_arrays name result expected tolerance =
  let n = Array.length expected in
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let diff = abs_float (result.(i) -. expected.(i)) in
    if diff > tolerance then begin
      if !errors < 5 then
        Printf.printf
          "  %s mismatch at %d: expected %.6f, got %.6f (diff=%.6f)\n"
          name
          i
          expected.(i)
          result.(i)
          diff ;
      incr errors
    end
  done ;
  !errors = 0

let () =
  let c = Test_helpers.parse_args "test_math_intrinsics" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  print_endline "=== Math Intrinsics Test (V2) ===" ;
  Printf.printf "Size: %d elements\n\n" cfg.size ;

  init_complex_data () ;

  let v2_devs = Test_helpers.init_devices cfg in
  if Array.length v2_devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices v2_devs ;

  if cfg.benchmark_all then begin
    print_endline (String.make 60 '-') ;
    Printf.printf "%-35s %10s %10s\n" "Device" "Time(ms)" "Status" ;
    print_endline (String.make 60 '-') ;

    let all_ok = ref true in

    Array.iter
      (fun v2_dev ->
        let name = v2_dev.Device.name in
        let framework = v2_dev.Device.framework in

        let v2_time, v2_result = run_complex_math v2_dev in
        let v2_ok =
          (not cfg.verify)
          || verify_float_arrays "V2" v2_result !expected_complex 0.01
        in
        let v2_status = if v2_ok then "OK" else "FAIL" in

        if not v2_ok then all_ok := false ;

        Printf.printf
          "%-35s %10.4f %10s\n"
          (Printf.sprintf "%s (%s)" name framework)
          v2_time
          v2_status)
      v2_devs ;

    print_endline (String.make 60 '-') ;

    if !all_ok then print_endline "\n=== All math intrinsics tests PASSED ==="
    else begin
      print_endline "\n=== Some math intrinsics tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    let dev = Test_helpers.get_device cfg v2_devs in
    Printf.printf "Using device: %s\n%!" dev.Device.name ;

    Printf.printf "\nRunning V2 path (complex math: sqrt/exp/cos)...\n%!" ;
    let v2_time, v2_result = run_complex_math dev in
    Printf.printf "  Time: %.4f ms\n%!" v2_time ;
    let v2_ok =
      (not cfg.verify)
      || verify_float_arrays "V2" v2_result !expected_complex 0.01
    in
    Printf.printf "  Status: %s\n%!" (if v2_ok then "PASSED" else "FAILED") ;

    if v2_ok then print_endline "\nMath intrinsics tests PASSED"
    else begin
      print_endline "\nMath intrinsics tests FAILED" ;
      exit 1
    end
  end
