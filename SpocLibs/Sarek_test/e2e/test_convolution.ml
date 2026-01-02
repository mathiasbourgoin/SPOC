(******************************************************************************
 * E2E test for Sarek PPX - 1D Convolution with V2 comparison
 *
 * Tests 1D convolution (3-point filter). V2 comparison for 1D only.
 * 2D convolution, Sobel, and shared memory variants run SPOC-only.
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

(* ========== Pure OCaml baseline ========== *)

let ocaml_conv1d input output n =
  output.(0) <- input.(0) ;
  for i = 1 to n - 2 do
    output.(i) <-
      (0.25 *. input.(i - 1)) +. (0.5 *. input.(i)) +. (0.25 *. input.(i + 1))
  done ;
  output.(n - 1) <- input.(n - 1)

(* ========== Shared test data ========== *)

let input_1d = ref [||]

let expected_1d = ref [||]

let init_conv1d_data () =
  let n = cfg.size in
  let inp = Array.init n (fun i -> sin (float_of_int i *. 0.1)) in
  let out = Array.make n 0.0 in
  input_1d := inp ;
  expected_1d := out ;
  ocaml_conv1d inp out n

(* ========== Sarek kernel (V2 compatible) ========== *)

let conv1d_3point_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int) ->
      let open Std in
      let tid = global_thread_id in
      if tid > 0 && tid < n - 1 then begin
        let left = input.(tid - 1) in
        let center = input.(tid) in
        let right = input.(tid + 1) in
        output.(tid) <- (0.25 *. left) +. (0.5 *. center) +. (0.25 *. right)
      end
      else if tid = 0 || tid = n - 1 then output.(tid) <- input.(tid)]

(* ========== SPOC test runner ========== *)

let run_conv1d_spoc dev =
  let n = cfg.size in
  let inp = !input_1d in

  let input = Spoc_Vector.create Spoc_Vector.float32 n in
  let output = Spoc_Vector.create Spoc_Vector.float32 n in

  for i = 0 to n - 1 do
    Spoc_Mem.set input i inp.(i) ;
    Spoc_Mem.set output i 0.0
  done ;

  ignore (Sarek.Kirc.gen conv1d_3point_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Spoc.Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Spoc.Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run conv1d_3point_kernel (input, output, n) (block, grid) 0 dev ;
  Spoc_Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in

  Spoc_Mem.to_cpu output () ;
  Spoc_Devices.flush dev () ;

  let result = Array.init n (fun i -> Spoc_Mem.get output i) in
  ((t1 -. t0) *. 1000.0, result)

(* ========== V2 test runner ========== *)

let run_conv1d_v2 (dev : V2_Device.t) =
  let n = cfg.size in
  let _, kirc = conv1d_3point_kernel in
  let ir =
    match kirc.Sarek.Kirc.body_v2 with
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
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [Sarek.Execute.Vec input; Sarek.Execute.Vec output; Sarek.Execute.Int n]
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
  let c = Test_helpers.parse_args "test_convolution" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  print_endline "=== 1D Convolution Test (SPOC + V2 Comparison) ===" ;
  Printf.printf "Size: %d elements\n\n" cfg.size ;

  init_conv1d_data () ;

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

  if cfg.benchmark_all then begin
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
              let time, result = run_conv1d_spoc spoc_dev in
              let ok =
                (not cfg.verify)
                || verify_float_arrays "SPOC" result !expected_1d 0.0001
              in
              (Printf.sprintf "%.4f" time, if ok then "OK" else "FAIL")
          | None -> ("-", "SKIP")
        in

        let v2_time, v2_result = run_conv1d_v2 v2_dev in
        let v2_ok =
          (not cfg.verify)
          || verify_float_arrays "V2" v2_result !expected_1d 0.0001
        in
        let v2_status = if v2_ok then "OK" else "FAIL" in

        if not v2_ok then all_ok := false ;
        if spoc_ok = "FAIL" then all_ok := false ;

        Printf.printf
          "%-35s %10s %10.4f %8s %8s\n"
          (Printf.sprintf "%s (%s)" name framework)
          spoc_time
          v2_time
          spoc_ok
          v2_status)
      v2_devs ;

    print_endline (String.make 80 '-') ;

    if !all_ok then print_endline "\n=== All convolution tests PASSED ==="
    else begin
      print_endline "\n=== Some convolution tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    let dev = Test_helpers.get_device cfg spoc_devs in
    let dev_name = dev.Spoc_Devices.general_info.Spoc_Devices.name in
    Printf.printf "Using device: %s\n%!" dev_name ;

    Printf.printf "\nRunning SPOC path (1D conv)...\n%!" ;
    let spoc_time, spoc_result = run_conv1d_spoc dev in
    Printf.printf "  Time: %.4f ms\n%!" spoc_time ;
    let spoc_ok =
      (not cfg.verify)
      || verify_float_arrays "SPOC" spoc_result !expected_1d 0.0001
    in
    Printf.printf "  Status: %s\n%!" (if spoc_ok then "PASSED" else "FAILED") ;

    let v2_dev_opt =
      Array.find_opt (fun d -> d.V2_Device.name = dev_name) v2_devs
    in
    match v2_dev_opt with
    | Some v2_dev ->
        Printf.printf "\nRunning V2 path (1D conv)...\n%!" ;
        let v2_time, v2_result = run_conv1d_v2 v2_dev in
        Printf.printf "  Time: %.4f ms\n%!" v2_time ;
        let v2_ok =
          (not cfg.verify)
          || verify_float_arrays "V2" v2_result !expected_1d 0.0001
        in
        Printf.printf "  Status: %s\n%!" (if v2_ok then "PASSED" else "FAILED") ;

        if spoc_ok && v2_ok then
          print_endline "\nConvolution tests PASSED (both paths)"
        else begin
          print_endline "\nConvolution tests FAILED" ;
          exit 1
        end
    | None ->
        Printf.printf "\nNo matching V2 device found\n%!" ;
        if spoc_ok then print_endline "\nConvolution tests PASSED (SPOC only)"
        else begin
          print_endline "\nConvolution tests FAILED" ;
          exit 1
        end
  end
