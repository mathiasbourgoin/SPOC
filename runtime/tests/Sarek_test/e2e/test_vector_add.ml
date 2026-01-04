(******************************************************************************
 * E2E test for Sarek PPX - Vector Add
 *
 * This test verifies that kernels compiled with the PPX can generate valid
 * GPU code and execute correctly via the GPU runtime.
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

(* Define kernel - vector addition c = a + b *)
let vector_add =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) + b.(tid)]

(* Run kernel via runtime path *)
let run_v2_on_device (dev : Device.t) =
  let _, kirc = vector_add in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  let a = Vector.create Vector.float32 !size in
  let b = Vector.create Vector.float32 !size in
  let c = Vector.create Vector.float32 !size in

  for i = 0 to !size - 1 do
    Vector.set a i (float_of_int i) ;
    Vector.set b i (float_of_int (i * 2)) ;
    (* Initialize output with a sentinel to detect missing writes *)
    Vector.set c i (-999.0)
    (* Changed to -999 to distinguish from inputs *)
  done ;

  (* DEBUG: Also initialize a and b with distinct sentinels first *)
  (* Printf.eprintf
    "[TEST] Before compute: a[0]=%.1f b[0]=%.1f c[0]=%.1f\n%!"
    (Vector.get a 0)
    (Vector.get b 0)
    (Vector.get c 0) ; *)
  let block_sz = !block_size in
  let grid_sz = (!size + block_sz - 1) / block_sz in
  let block = Execute.dims1d block_sz in
  let grid = Execute.dims1d grid_sz in

  (* Warmup run to exclude compilation time *)
  if !benchmark_all then (
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:[Execute.Vec a; Execute.Vec b; Execute.Vec c; Execute.Int !size]
      ~block
      ~grid
      () ;
    Transfer.flush dev) ;

  let t0 = Unix.gettimeofday () in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec a; Execute.Vec b; Execute.Vec c; Execute.Int !size]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Read back all three vectors to see what's in them *)
  (* let result_a = Vector.to_array a in
  let result_b = Vector.to_array b in *)
  let result_c = Vector.to_array c in
  (* Printf.eprintf "[TEST] After compute:\n" ;
  Printf.eprintf
    "  a[0]=%.1f a[1]=%.1f a[2]=%.1f\n%!"
    result_a.(0)
    result_a.(1)
    result_a.(2) ;
  Printf.eprintf
    "  b[0]=%.1f b[1]=%.1f b[2]=%.1f\n%!"
    result_b.(0)
    result_b.(1)
    result_b.(2) ;
  Printf.eprintf
    "  c[0]=%.1f c[1]=%.1f c[2]=%.1f\n%!"
    result_c.(0)
    result_c.(1)
    result_c.(2) ; *)
  (time_ms, result_c)

(* Run kernel via interpreter directly *)
let run_interpreter () =
  let _, kirc = vector_add in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  (* Create interpreter arrays *)
  let a =
    Array.init !size (fun i -> Sarek_ir_interp.VFloat32 (float_of_int i))
  in
  let b =
    Array.init !size (fun i -> Sarek_ir_interp.VFloat32 (float_of_int (i * 2)))
  in
  (* Initialize output with a sentinel to detect missing writes *)
  let c = Array.make !size (Sarek_ir_interp.VFloat32 (-1.0)) in

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

  print_endline "=== Vector Add Test (runtime) ===" ;
  Printf.printf "Size: %d elements\n\n" !size ;

  (* Initialize runtime devices - include all frameworks *)
  let devs =
    Device.init
      ~frameworks:["CUDA"; "OpenCL"; "Vulkan"; "Native"; "Interpreter"]
      ()
  in

  if Array.length devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;

  Printf.printf "Found %d device(s)\n\n" (Array.length devs) ;

  let expected = compute_expected () in

  if !benchmark_all then begin
    print_endline (String.make 60 '-') ;
    Printf.printf "%-35s %10s %10s\n" "Device" "Time(ms)" "Status" ;
    print_endline (String.make 60 '-') ;

    let all_ok = ref true in

    (* Test each runtime device *)
    Array.iter
      (fun dev ->
        let name = dev.Device.name in
        let framework = dev.Device.framework in

        let time, result = run_v2_on_device dev in
        let ok =
          if !verify then verify_results "runtime" result expected else true
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
    let dev =
      if !use_native then (
        match Array.find_opt (fun d -> d.Device.framework = "Native") devs with
        | Some d -> d
        | None ->
            print_endline "No native CPU device found" ;
            exit 1)
      else if !use_interpreter then (
        match
          Array.find_opt (fun d -> d.Device.framework = "Interpreter") devs
        with
        | Some d -> d
        | None ->
            print_endline "No interpreter device found" ;
            exit 1)
      else devs.(!dev_id)
    in
    let dev_name = dev.Device.name in
    Printf.printf "Using device: %s\n%!" dev_name ;

    (* Run runtime *)
    Printf.printf "\nRunning runtime path...\n%!" ;
    let time, result = run_v2_on_device dev in
    Printf.printf "  Time: %.4f ms\n%!" time ;
    let ok =
      if !verify then verify_results "runtime" result expected else true
    in
    Printf.printf "  Status: %s\n%!" (if ok then "PASSED" else "FAILED") ;

    if ok then print_endline "\nE2E Test PASSED"
    else begin
      print_endline "\nE2E Test FAILED" ;
      exit 1
    end
  end
