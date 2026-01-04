(******************************************************************************
 * E2E test for Sarek PPX - Matrix Transpose
 *
 * Tests naive matrix transpose with 1D kernel.
 * Transpose is a memory-bound operation that benefits from coalescing.
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

let ocaml_transpose input output width height =
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let in_idx = (y * width) + x in
      let out_idx = (x * height) + y in
      output.(out_idx) <- input.(in_idx)
    done
  done

(* ========== Shared test data ========== *)

let input_data = ref [||]

let expected_data = ref [||]

let matrix_dim = ref 0

let init_transpose_data () =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int cfg.size))) in
  matrix_dim := dim ;
  let n = dim * dim in
  let inp = Array.init n (fun i -> float_of_int i) in
  let out = Array.make n 0.0 in
  input_data := inp ;
  expected_data := out ;
  ocaml_transpose inp out dim dim

(* ========== Sarek kernel ========== *)

let transpose_naive_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int)
        (height : int) ->
      let open Std in
      let tid = global_thread_id in
      let n = width * height in
      if tid < n then begin
        let x = tid mod width in
        let y = tid / width in
        let in_idx = (y * width) + x in
        let out_idx = (x * height) + y in
        output.(out_idx) <- input.(in_idx)
      end]

(* ========== V2 test runner ========== *)

let run_transpose (dev : Device.t) =
  let dim = !matrix_dim in
  let n = dim * dim in
  let _, kirc = transpose_naive_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No V2 IR"
  in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Vector.set input i !input_data.(i) ;
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
    ~args:
      [Execute.Vec input; Execute.Vec output; Execute.Int dim; Execute.Int dim]
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
          "  %s mismatch at %d: expected %.2f, got %.2f\n"
          name
          i
          expected.(i)
          result.(i) ;
      incr errors
    end
  done ;
  !errors = 0

let () =
  let c = Test_helpers.parse_args "test_transpose" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  print_endline "=== Matrix Transpose Test (V2) ===" ;
  Printf.printf "Size: %d elements\n" cfg.size ;

  init_transpose_data () ;
  Printf.printf "Matrix dimensions: %dx%d\n\n" !matrix_dim !matrix_dim ;

  let devs = Test_helpers.init_devices cfg in
  if Array.length devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices devs ;

  if cfg.benchmark_all then begin
    print_endline (String.make 60 '-') ;
    Printf.printf "%-35s %10s %10s\n" "Device" "Time(ms)" "Status" ;
    print_endline (String.make 60 '-') ;

    let all_ok = ref true in

    Array.iter
      (fun dev ->
        let name = dev.Device.name in
        let framework = dev.Device.framework in

        let time, result = run_transpose dev in
        let ok =
          (not cfg.verify)
          || verify_float_arrays "V2" result !expected_data 0.001
        in
        let status = if ok then "OK" else "FAIL" in

        if not ok then all_ok := false ;

        Printf.printf
          "%-35s %10.4f %10s\n"
          (Printf.sprintf "%s (%s)" name framework)
          time
          status)
      devs ;

    print_endline (String.make 60 '-') ;

    if !all_ok then print_endline "\n=== All transpose tests PASSED ==="
    else begin
      print_endline "\n=== Some transpose tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Device.name ;

    Printf.printf "\nRunning V2 path (naive transpose)...\n%!" ;
    let time, result = run_transpose dev in
    Printf.printf "  Time: %.4f ms\n%!" time ;
    let ok =
      (not cfg.verify) || verify_float_arrays "V2" result !expected_data 0.001
    in
    Printf.printf "  Status: %s\n%!" (if ok then "PASSED" else "FAILED") ;

    if ok then print_endline "\nTranspose tests PASSED"
    else begin
      print_endline "\nTranspose tests FAILED" ;
      exit 1
    end
  end
