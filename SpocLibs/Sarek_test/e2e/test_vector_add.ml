(******************************************************************************
 * E2E test for Sarek PPX - Vector Add with V2 comparison
 *
 * This test verifies that kernels compiled with the PPX can generate valid
 * GPU code and execute correctly via both SPOC and V2 runtime paths.
 ******************************************************************************)

(* Module aliases to avoid conflicts *)
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

let size = ref 1024

let dev_id = ref 0

let block_size = ref 256

let verify = ref true

let use_interpreter = ref false

let use_native = ref false

let benchmark_all = ref false

let usage () =
  Printf.printf "Usage: %s [options]\n" Sys.argv.(0) ;
  Printf.printf "Options:\n" ;
  Printf.printf "  -d <id>       Device ID (default: 0)\n" ;
  Printf.printf "  --interpreter Use CPU interpreter device\n" ;
  Printf.printf "  --native      Use native CPU runtime device\n" ;
  Printf.printf "  --benchmark   Run on all devices and compare times\n" ;
  Printf.printf "  -s <size>     Vector size (default: 1024)\n" ;
  Printf.printf "  -b <size>     Block/work-group size (default: 256)\n" ;
  Printf.printf "  -no-verify    Skip result verification\n" ;
  Printf.printf "  -h            Show this help\n" ;
  exit 0

let parse_args () =
  let i = ref 1 in
  while !i < Array.length Sys.argv do
    (match Sys.argv.(!i) with
    | "-d" ->
        incr i ;
        dev_id := int_of_string Sys.argv.(!i)
    | "--interpreter" -> use_interpreter := true
    | "--native" -> use_native := true
    | "--benchmark" -> benchmark_all := true
    | "-s" ->
        incr i ;
        size := int_of_string Sys.argv.(!i)
    | "-b" ->
        incr i ;
        block_size := int_of_string Sys.argv.(!i)
    | "-no-verify" -> verify := false
    | "-h" | "--help" -> usage ()
    | _ -> ()) ;
    incr i
  done

(* Define kernel - works with both SPOC and V2 *)
let vector_add =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) +. b.(tid)]

(* Run kernel via SPOC path *)
let run_spoc_on_device dev =
  let a = Spoc_Vector.create Spoc_Vector.float32 !size in
  let b = Spoc_Vector.create Spoc_Vector.float32 !size in
  let c = Spoc_Vector.create Spoc_Vector.float32 !size in

  for i = 0 to !size - 1 do
    Spoc_Mem.set a i (float_of_int i) ;
    Spoc_Mem.set b i (float_of_int (i * 2)) ;
    Spoc_Mem.set c i 0.0
  done ;

  ignore (Sarek.Kirc.gen vector_add dev) ;

  let threadsPerBlock =
    match dev.Spoc_Devices.specific_info with
    | Spoc_Devices.OpenCLInfo clI -> (
        match clI.Spoc_Devices.device_type with
        | Spoc_Devices.CL_DEVICE_TYPE_CPU -> 1
        | _ -> !block_size)
    | _ -> !block_size
  in
  let blocksPerGrid = (!size + threadsPerBlock - 1) / threadsPerBlock in
  let block =
    {
      Spoc.Kernel.blockX = threadsPerBlock;
      Spoc.Kernel.blockY = 1;
      Spoc.Kernel.blockZ = 1;
    }
  in
  let grid =
    {
      Spoc.Kernel.gridX = blocksPerGrid;
      Spoc.Kernel.gridY = 1;
      Spoc.Kernel.gridZ = 1;
    }
  in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run vector_add (a, b, c, !size) (block, grid) 0 dev ;
  Spoc_Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Read back results *)
  Spoc_Mem.to_cpu c () ;
  Spoc_Devices.flush dev () ;

  let result = Array.make !size 0.0 in
  for i = 0 to !size - 1 do
    result.(i) <- Spoc_Mem.get c i
  done ;
  (time_ms, result)

(* Run kernel via V2 path *)
let run_v2_on_device (dev : V2_Device.t) =
  let _, kirc = vector_add in
  let ir =
    match kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  let a = V2_Vector.create V2_Vector.float32 !size in
  let b = V2_Vector.create V2_Vector.float32 !size in
  let c = V2_Vector.create V2_Vector.float32 !size in

  for i = 0 to !size - 1 do
    V2_Vector.set a i (float_of_int i) ;
    V2_Vector.set b i (float_of_int (i * 2)) ;
    V2_Vector.set c i 0.0
  done ;

  let block_sz = 256 in
  let grid_sz = (!size + block_sz - 1) / block_sz in
  let block = Sarek.Execute.dims1d block_sz in
  let grid = Sarek.Execute.dims1d grid_sz in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec a;
        Sarek.Execute.Vec b;
        Sarek.Execute.Vec c;
        Sarek.Execute.Int !size;
      ]
    ~block
    ~grid
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (time_ms, V2_Vector.to_array c)

(* Compute expected results on CPU *)
let compute_expected () =
  Array.init !size (fun i -> float_of_int i +. float_of_int (i * 2))

(* Verify results against expected *)
let verify_results name result expected =
  let errors = ref 0 in
  for i = 0 to !size - 1 do
    let diff = abs_float (result.(i) -. expected.(i)) in
    if diff > 0.001 then begin
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
  parse_args () ;

  print_endline "=== Vector Add Test (SPOC + V2 Comparison) ===" ;
  Printf.printf "Size: %d elements\n\n" !size ;

  (* Initialize SPOC devices *)
  let spoc_devs =
    if !use_interpreter then
      Spoc_Devices.init
        ~interpreter:(Some Spoc_Devices.Sequential)
        ~native:!use_native
        ()
    else Spoc_Devices.init ~native:!use_native ()
  in
  if Array.length spoc_devs = 0 then begin
    print_endline "No GPU devices found" ;
    exit 1
  end ;

  (* Initialize V2 devices *)
  let v2_devs = V2_Device.init ~frameworks:["CUDA"; "OpenCL"] () in

  Printf.printf
    "Found %d SPOC device(s), %d V2 device(s)\n\n"
    (Array.length spoc_devs)
    (Array.length v2_devs) ;

  let expected = compute_expected () in

  if !benchmark_all then begin
    print_endline (String.make 90 '-') ;
    Printf.printf
      "%-35s %10s %10s %10s %10s\n"
      "Device"
      "SPOC(ms)"
      "V2(ms)"
      "SPOC"
      "V2" ;
    print_endline (String.make 90 '-') ;

    let all_ok = ref true in

    (* Test each V2 device *)
    Array.iter
      (fun v2_dev ->
        let name = v2_dev.V2_Device.name in
        let framework = v2_dev.V2_Device.framework in

        (* Find matching SPOC device *)
        let spoc_dev_opt =
          Array.find_opt
            (fun d -> d.Spoc_Devices.general_info.Spoc_Devices.name = name)
            spoc_devs
        in

        let spoc_time, spoc_ok =
          match spoc_dev_opt with
          | Some spoc_dev ->
              let time, result = run_spoc_on_device spoc_dev in
              let ok =
                if !verify then verify_results "SPOC" result expected else true
              in
              (Printf.sprintf "%.4f" time, if ok then "OK" else "FAIL")
          | None -> ("-", "SKIP")
        in

        let v2_time, v2_result = run_v2_on_device v2_dev in
        let v2_ok =
          if !verify then verify_results "V2" v2_result expected else true
        in
        let v2_status = if v2_ok then "OK" else "FAIL" in

        if not v2_ok then all_ok := false ;
        if spoc_ok = "FAIL" then all_ok := false ;

        Printf.printf
          "%-35s %10s %10.4f %10s %10s\n"
          (Printf.sprintf "%s (%s)" name framework)
          spoc_time
          v2_time
          spoc_ok
          v2_status)
      v2_devs ;

    print_endline (String.make 90 '-') ;

    if !all_ok then print_endline "\n=== All tests PASSED ==="
    else begin
      print_endline "\n=== Some tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    (* Single device mode *)
    let dev =
      if !use_native then (
        match Spoc_Devices.find_native_id spoc_devs with
        | Some id -> spoc_devs.(id)
        | None ->
            print_endline "No native CPU device found" ;
            exit 1)
      else if !use_interpreter then (
        match Spoc_Devices.find_interpreter_id spoc_devs with
        | Some id -> spoc_devs.(id)
        | None ->
            print_endline "No interpreter device found" ;
            exit 1)
      else spoc_devs.(!dev_id)
    in
    let dev_name = dev.Spoc_Devices.general_info.Spoc_Devices.name in
    Printf.printf "Using device: %s\n%!" dev_name ;

    (* Run SPOC *)
    Printf.printf "\nRunning SPOC path...\n%!" ;
    let spoc_time, spoc_result = run_spoc_on_device dev in
    Printf.printf "  Time: %.4f ms\n%!" spoc_time ;
    let spoc_ok =
      if !verify then verify_results "SPOC" spoc_result expected else true
    in
    Printf.printf "  Status: %s\n%!" (if spoc_ok then "PASSED" else "FAILED") ;

    (* Find matching V2 device *)
    let v2_dev_opt =
      Array.find_opt (fun d -> d.V2_Device.name = dev_name) v2_devs
    in

    match v2_dev_opt with
    | Some v2_dev ->
        Printf.printf "\nRunning V2 path...\n%!" ;
        let v2_time, v2_result = run_v2_on_device v2_dev in
        Printf.printf "  Time: %.4f ms\n%!" v2_time ;
        let v2_ok =
          if !verify then verify_results "V2" v2_result expected else true
        in
        Printf.printf "  Status: %s\n%!" (if v2_ok then "PASSED" else "FAILED") ;

        if spoc_ok && v2_ok then print_endline "\nE2E Test PASSED (both paths)"
        else begin
          print_endline "\nE2E Test FAILED" ;
          exit 1
        end
    | None ->
        Printf.printf "\nNo matching V2 device found for comparison\n%!" ;
        if spoc_ok then print_endline "\nE2E Test PASSED (SPOC only)"
        else begin
          print_endline "\nE2E Test FAILED" ;
          exit 1
        end
  end
