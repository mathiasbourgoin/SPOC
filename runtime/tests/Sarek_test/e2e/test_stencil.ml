(******************************************************************************
 * E2E test for Sarek PPX - Stencil operations
 *
 * Tests 1D stencil pattern.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std

(* Module aliases *)
module V2_Device = Spoc_core.Device
module V2_Vector = Spoc_core.Vector
module V2_Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init () ;
  Sarek_native.Native_plugin_v2.init () ;
  Sarek_interpreter.Interpreter_plugin_v2.init ()

let cfg = Test_helpers.default_config ()

(* ========== Pure OCaml baseline ========== *)

let ocaml_stencil_1d input output n =
  for i = 1 to n - 2 do
    let left = input.(i - 1) in
    let center = input.(i) in
    let right = input.(i + 1) in
    output.(i) <- (left +. center +. right) /. 3.0
  done

(* ========== Shared test data ========== *)

let input_1d = ref [||]

let expected_1d = ref [||]

let init_1d_data () =
  let n = cfg.size in
  let inp = Array.init n (fun i -> sin (float_of_int i *. 0.1)) in
  let exp = Array.make n 0.0 in
  input_1d := inp ;
  expected_1d := exp ;
  ocaml_stencil_1d inp exp n

(* ========== Sarek kernel ========== *)

let stencil_1d_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int) ->
      let open Std in
      let tid = global_thread_id in
      if tid > 0 && tid < n - 1 then
        let left = input.(tid - 1) in
        let center = input.(tid) in
        let right = input.(tid + 1) in
        output.(tid) <- (left +. center +. right) /. 3.0]

(* ========== V2 test runner ========== *)

let run_stencil_1d_v2 (dev : V2_Device.t) =
  let n = cfg.size in
  let _, kirc = stencil_1d_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "No V2 IR"
  in

  let input = V2_Vector.create V2_Vector.float32 n in
  let output = V2_Vector.create V2_Vector.float32 n in

  for i = 0 to n - 1 do
    V2_Vector.set input i !input_1d.(i) ;
    V2_Vector.set output i 0.0
  done ;

  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Execute.dims1d block_size in
  let grid = Execute.dims1d grid_size in

  let t0 = Unix.gettimeofday () in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec input; Execute.Vec output; Execute.Int n]
    ~block
    ~grid
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  ((t1 -. t0) *. 1000.0, V2_Vector.to_array output)

(* ========== Verification ========== *)

let verify_float_arrays name result expected tolerance =
  let n = Array.length expected in
  let errors = ref 0 in
  for i = 1 to n - 2 do
    let diff = abs_float (result.(i) -. expected.(i)) in
    if diff > tolerance then begin
      if !errors < 5 then
        Printf.printf
          "  %s mismatch at %d: expected %.6f, got %.6f\n"
          name
          i
          expected.(i)
          result.(i) ;
      incr errors
    end
  done ;
  !errors = 0

let () =
  let c = Test_helpers.parse_args "test_stencil" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  print_endline "=== 1D Stencil Test (V2) ===" ;
  Printf.printf "Size: %d elements\n\n" cfg.size ;

  init_1d_data () ;

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
        let name = v2_dev.V2_Device.name in
        let framework = v2_dev.V2_Device.framework in

        let v2_time, v2_result = run_stencil_1d_v2 v2_dev in
        let v2_ok =
          (not cfg.verify)
          || verify_float_arrays "V2" v2_result !expected_1d 0.0001
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

    if !all_ok then print_endline "\n=== All stencil tests PASSED ==="
    else begin
      print_endline "\n=== Some stencil tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    let dev = Test_helpers.get_device cfg v2_devs in
    Printf.printf "Using device: %s\n%!" dev.V2_Device.name ;

    Printf.printf "\nRunning V2 path (1D stencil)...\n%!" ;
    let v2_time, v2_result = run_stencil_1d_v2 dev in
    Printf.printf "  Time: %.4f ms\n%!" v2_time ;
    let v2_ok =
      (not cfg.verify) || verify_float_arrays "V2" v2_result !expected_1d 0.0001
    in
    Printf.printf "  Status: %s\n%!" (if v2_ok then "PASSED" else "FAILED") ;

    if v2_ok then print_endline "\nStencil tests PASSED"
    else begin
      print_endline "\nStencil tests FAILED" ;
      exit 1
    end
  end
