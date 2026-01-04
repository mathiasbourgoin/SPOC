(******************************************************************************
 * V2 Runtime Test
 *
 * Tests V2 runtime produces correct output by comparing against CPU reference.
 * V2 runtime only.
 ******************************************************************************)

(* Module aliases *)
module V2_Device = Spoc_core.Device
module V2_Vector = Spoc_core.Vector
module V2_Transfer = Spoc_core.Transfer
module Std = Sarek_stdlib.Std

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

let size = 1024

(* Define kernel using Sarek PPX *)
let vector_add =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) +. b.(tid)]

(* Run via V2 runtime path and return results *)
let run_v2 (dev : V2_Device.t) : float array =
  (* Get the IR from the kernel *)
  let _, kirc = vector_add in
  let ir =
    match kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  (* Create V2 vectors *)
  let a = V2_Vector.create V2_Vector.float32 size in
  let b = V2_Vector.create V2_Vector.float32 size in
  let c = V2_Vector.create V2_Vector.float32 size in

  (* Initialize *)
  for i = 0 to size - 1 do
    V2_Vector.set a i (float_of_int i) ;
    V2_Vector.set b i (float_of_int (i * 2)) ;
    V2_Vector.set c i 0.0
  done ;

  (* Configure grid/block *)
  let block_size = 256 in
  let grid_size = (size + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

  (* Run via V2 *)
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec a;
        Sarek.Execute.Vec b;
        Sarek.Execute.Vec c;
        Sarek.Execute.Int size;
      ]
    ~block
    ~grid
    () ;

  V2_Transfer.flush dev ;
  V2_Vector.to_array c

(* Compare result array against expected *)
let verify_results (result : float array) (expected : float array) : int =
  let errors = ref 0 in
  for i = 0 to Array.length result - 1 do
    let diff = abs_float (result.(i) -. expected.(i)) in
    if diff > 0.001 then begin
      if !errors < 5 then
        Printf.printf
          "  Mismatch at %d: expected=%.2f, got=%.2f\n"
          i
          expected.(i)
          result.(i) ;
      incr errors
    end
  done ;
  !errors

let () =
  print_endline "=== V2 Runtime Test ===" ;
  print_endline "Testing V2 execution against CPU reference\n" ;

  (* Initialize V2 devices *)
  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length v2_devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;

  Printf.printf "Found %d V2 device(s)\n\n" (Array.length v2_devs) ;

  print_endline (String.make 60 '-') ;
  Printf.printf "%-40s %10s %10s\n" "Device" "V2" "Status" ;
  print_endline (String.make 60 '-') ;

  let all_ok = ref true in

  (* Compute expected results (CPU reference) *)
  let expected =
    Array.init size (fun i -> float_of_int i +. float_of_int (i * 2))
  in

  (* Test each V2 device *)
  Array.iter
    (fun v2_dev ->
      let name = v2_dev.V2_Device.name in
      let framework = v2_dev.V2_Device.framework in
      let v2_result = run_v2 v2_dev in
      let errors = verify_results v2_result expected in
      let status = if errors = 0 then "PASS" else "FAIL" in
      if errors > 0 then all_ok := false ;
      Printf.printf
        "%-40s %10s %10s\n"
        (name ^ " (" ^ framework ^ ")")
        "OK"
        status)
    v2_devs ;

  print_endline (String.make 60 '-') ;

  if !all_ok then begin
    print_endline "\n=== All tests PASSED ===" ;
    print_endline "V2 runtime produces correct results"
  end
  else begin
    print_endline "\n=== Some tests FAILED ===" ;
    exit 1
  end
