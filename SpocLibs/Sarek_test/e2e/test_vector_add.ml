(******************************************************************************
 * E2E test for Sarek PPX - Vector Add
 *
 * This test verifies that kernels compiled with the PPX can generate valid
 * GPU code and execute correctly via the V2 runtime.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std

(* Module aliases *)
module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init () ;
  Sarek_native.Native_plugin_v2.init () ;
  Sarek_interpreter.Interpreter_plugin_v2.init ()

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

(* Define kernel *)
let vector_add =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) +. b.(tid)]

(* Run kernel via V2 path *)
let run_v2_on_device (dev : V2_Device.t) =
  let _, kirc = vector_add in
  let ir =
    match kirc.Sarek.Kirc_types.body_v2 with
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

  let block_sz = !block_size in
  let grid_sz = (!size + block_sz - 1) / block_sz in
  let block = Execute.dims1d block_sz in
  let grid = Execute.dims1d grid_sz in

  let t0 = Unix.gettimeofday () in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec a; Execute.Vec b; Execute.Vec c; Execute.Int !size]
    ~block
    ~grid
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (time_ms, V2_Vector.to_array c)

(* Run kernel via interpreter directly *)
let run_interpreter () =
  let _, kirc = vector_add in
  let ir =
    match kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  (* Create interpreter arrays *)
  let a =
    Array.init !size (fun i -> Sarek_ir_interp.VFloat32 (float_of_int i))
  in
  let b =
    Array.init !size (fun i -> Sarek_ir_interp.VFloat32 (float_of_int (i * 2)))
  in
  let c = Array.make !size (Sarek_ir_interp.VFloat32 0.0) in

  let block_sz = min 256 !size in
  let grid_sz = (!size + block_sz - 1) / block_sz in

  let t0 = Unix.gettimeofday () in
  Sarek_ir_interp.run_kernel
    ir
    ~block:(block_sz, 1, 1)
    ~grid:(grid_sz, 1, 1)
    [
      ("a", Sarek_ir_interp.ArgArray a);
      ("b", Sarek_ir_interp.ArgArray b);
      ("c", Sarek_ir_interp.ArgArray c);
      ( "n",
        Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VInt32 (Int32.of_int !size))
      );
    ] ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Extract results *)
  let result =
    Array.map (function Sarek_ir_interp.VFloat32 f -> f | _ -> 0.0) c
  in

  (time_ms, result)

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

  print_endline "=== Vector Add Test (V2) ===" ;
  Printf.printf "Size: %d elements\n\n" !size ;

  (* Initialize V2 devices - include all frameworks *)
  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in

  if Array.length v2_devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;

  Printf.printf "Found %d device(s)\n\n" (Array.length v2_devs) ;

  let expected = compute_expected () in

  if !benchmark_all then begin
    print_endline (String.make 60 '-') ;
    Printf.printf "%-35s %10s %10s\n" "Device" "Time(ms)" "Status" ;
    print_endline (String.make 60 '-') ;

    let all_ok = ref true in

    (* Test each V2 device *)
    Array.iter
      (fun v2_dev ->
        let name = v2_dev.V2_Device.name in
        let framework = v2_dev.V2_Device.framework in

        let v2_time, v2_result = run_v2_on_device v2_dev in
        let v2_ok =
          if !verify then verify_results "V2" v2_result expected else true
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

    (* Run interpreter test *)
    print_endline "\n=== Interpreter Test ===" ;
    let interp_time, interp_result = run_interpreter () in
    let interp_ok =
      if !verify then verify_results "Interpreter" interp_result expected
      else true
    in
    Printf.printf
      "Interpreter: %.4f ms - %s\n"
      interp_time
      (if interp_ok then "OK" else "FAIL") ;
    if not interp_ok then all_ok := false ;

    if !all_ok then print_endline "\n=== All tests PASSED ==="
    else begin
      print_endline "\n=== Some tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    (* Single device mode *)
    let v2_dev =
      if !use_native then (
        match
          Array.find_opt (fun d -> d.V2_Device.framework = "Native") v2_devs
        with
        | Some d -> d
        | None ->
            print_endline "No native CPU device found" ;
            exit 1)
      else if !use_interpreter then (
        match
          Array.find_opt
            (fun d -> d.V2_Device.framework = "Interpreter")
            v2_devs
        with
        | Some d -> d
        | None ->
            print_endline "No interpreter device found" ;
            exit 1)
      else v2_devs.(!dev_id)
    in
    let dev_name = v2_dev.V2_Device.name in
    Printf.printf "Using device: %s\n%!" dev_name ;

    (* Run V2 *)
    Printf.printf "\nRunning V2 path...\n%!" ;
    let v2_time, v2_result = run_v2_on_device v2_dev in
    Printf.printf "  Time: %.4f ms\n%!" v2_time ;
    let v2_ok =
      if !verify then verify_results "V2" v2_result expected else true
    in
    Printf.printf "  Status: %s\n%!" (if v2_ok then "PASSED" else "FAILED") ;

    if v2_ok then print_endline "\nE2E Test PASSED"
    else begin
      print_endline "\nE2E Test FAILED" ;
      exit 1
    end
  end
