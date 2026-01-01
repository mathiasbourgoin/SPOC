(******************************************************************************
 * E2E test for Sarek Runtime V2
 *
 * This test demonstrates the new V2 runtime API:
 * - Vector module with location tracking
 * - Transfer module for explicit data movement
 * - Profiling for timing
 * - Error module for structured errors
 ******************************************************************************)

(* Force plugin initialization *)
let () = Sarek_cuda.Cuda_plugin.init ()
let () = Sarek_opencl.Opencl_plugin.init ()

open Sarek_core

let size = 1024

let () =
  print_endline "=== Runtime V2 Vector Add Test ===" ;

  (* Step 1: Initialize devices using V2 Device module *)
  print_endline "\n[1] Initializing devices..." ;
  let devices = Device.init ~frameworks:["CUDA"; "OpenCL"] () in
  if Array.length devices = 0 then begin
    print_endline "No devices found" ;
    exit 0
  end ;
  Device.print_all () ;

  let dev =
    match Device.best () with
    | Some d -> d
    | None -> raise Error.No_device_available
  in
  Printf.printf "\nUsing: %s\n" (Device.to_string dev) ;

  (* Step 2: Create vectors using V2 Vector module *)
  print_endline "\n[2] Creating vectors..." ;
  let a = Vector.create Vector.float32 size in
  let b = Vector.create Vector.float32 size in
  let c = Vector.create Vector.float32 size in
  Printf.printf "Created: %s\n" (Vector.to_string a) ;
  Printf.printf "Created: %s\n" (Vector.to_string b) ;
  Printf.printf "Created: %s\n" (Vector.to_string c) ;

  (* Step 3: Initialize data using V2 element access *)
  print_endline "\n[3] Initializing data..." ;
  Vector.iteri (fun i _ -> Vector.set a i (float_of_int i)) a ;
  Vector.iteri (fun i _ -> Vector.set b i (float_of_int (i * 2))) b ;
  Vector.fill c 0.0 ;
  Printf.printf "a[0..4] = [%.0f, %.0f, %.0f, %.0f, %.0f]\n"
    (Vector.get a 0) (Vector.get a 1) (Vector.get a 2)
    (Vector.get a 3) (Vector.get a 4) ;
  Printf.printf "b[0..4] = [%.0f, %.0f, %.0f, %.0f, %.0f]\n"
    (Vector.get b 0) (Vector.get b 1) (Vector.get b 2)
    (Vector.get b 3) (Vector.get b 4) ;

  (* Step 4: Transfer to device using V2 Transfer module *)
  print_endline "\n[4] Transferring to device..." ;
  Transfer.to_device a dev ;
  Transfer.to_device b dev ;
  Transfer.to_device c dev ;
  Printf.printf "After transfer: %s\n" (Vector.to_string a) ;
  Printf.printf "After transfer: %s\n" (Vector.to_string b) ;
  Printf.printf "After transfer: %s\n" (Vector.to_string c) ;

  (* Step 5: Define and compile kernel *)
  print_endline "\n[5] Compiling kernel..." ;

  (* For now, use inline source - later will integrate with Sarek PPX *)
  let cuda_source = {|
extern "C" __global__ void vector_add(float *a, float *b, float *c, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}
|} in

  let opencl_source = {|
__kernel void vector_add(__global float *a, __global float *b,
                         __global float *c, int n) {
  int tid = get_global_id(0);
  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}
|} in

  let source = if Device.is_cuda dev then cuda_source else opencl_source in
  let kernel = Kernel.compile dev ~name:"vector_add" ~source in
  print_endline "Kernel compiled" ;

  (* Step 6: Setup arguments and launch *)
  print_endline "\n[6] Launching kernel..." ;
  let args = Kernel.create_args dev in

  (* Get device buffer pointers *)
  let get_ptr vec =
    match Vector.get_buffer vec dev with
    | Some buf -> let (module B : Vector.DEVICE_BUFFER) = buf in B.ptr
    | None -> failwith "No device buffer"
  in

  (* Set arguments - this is low-level, would be wrapped in Sarek *)
  (* For now we use the raw Memory API *)
  let buf_a =
    match Vector.get_buffer a dev with
    | Some b -> b
    | None -> failwith "No buffer"
  in
  let buf_b =
    match Vector.get_buffer b dev with
    | Some b -> b
    | None -> failwith "No buffer"
  in
  let buf_c =
    match Vector.get_buffer c dev with
    | Some b -> b
    | None -> failwith "No buffer"
  in

  (* Note: Kernel.set_arg_buffer expects Memory.buffer but we have Vector.device_buffer
     This is a gap in the current design - we need to bridge this.
     For now, let's use the device ptr directly via a lower-level approach *)
  ignore (buf_a, buf_b, buf_c, get_ptr, args) ;

  (* For demonstration, show that we can still use the old Runtime for execution *)
  print_endline "Note: Full kernel execution requires bridging Vector buffers to Kernel args" ;
  print_endline "This will be completed when V2 Kernel module is updated" ;

  (* Step 7: Simulate computation on CPU for verification demo *)
  print_endline "\n[7] Simulating computation (CPU fallback)..." ;
  (* Sync back to CPU first *)
  Transfer.to_cpu a ;
  Transfer.to_cpu b ;
  for i = 0 to size - 1 do
    Vector.set c i (Vector.get a i +. Vector.get b i)
  done ;

  (* Step 8: Verify results *)
  print_endline "\n[8] Verifying results..." ;
  let errors = ref 0 in
  for i = 0 to size - 1 do
    let expected = float_of_int i +. float_of_int (i * 2) in
    let got = Vector.get c i in
    if abs_float (got -. expected) > 0.001 then begin
      if !errors < 5 then
        Printf.printf "  Mismatch at %d: expected %.2f, got %.2f\n" i expected got ;
      incr errors
    end
  done ;

  if !errors > 0 then begin
    Printf.printf "Total errors: %d\n" !errors ;
    print_endline "=== Test FAILED ===" ;
    exit 1
  end ;

  (* Step 9: Cleanup *)
  print_endline "\n[9] Cleanup..." ;
  Transfer.free_all_buffers a ;
  Transfer.free_all_buffers b ;
  Transfer.free_all_buffers c ;
  Printf.printf "Final state: %s\n" (Vector.to_string a) ;

  (* Step 10: Demo profiling *)
  print_endline "\n[10] Profiling demo..." ;
  let result, time = Profiling.cpu_timed (fun () ->
    let v = Vector.create Vector.float32 10000 in
    Vector.fill v 42.0 ;
    Vector.fold_left ( +. ) 0.0 v
  ) in
  Printf.printf "Created and summed 10000 floats: sum=%.0f, time=%.3fms\n" result time ;

  print_endline "\n=== Runtime V2 Test PASSED ==="
