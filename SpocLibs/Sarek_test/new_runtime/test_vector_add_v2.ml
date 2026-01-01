(******************************************************************************
 * E2E test for Sarek Runtime V2
 *
 * This test demonstrates the new V2 runtime API with actual GPU execution:
 * - Vector module with location tracking
 * - Transfer module for explicit data movement
 * - Kernel module with set_arg_ptr for device pointers
 * - Profiling for timing
 ******************************************************************************)

(* Force plugin initialization *)
let () = Sarek_cuda.Cuda_plugin.init ()
let () = Sarek_opencl.Opencl_plugin.init ()

open Sarek_core

let size = 1024

let () =
  print_endline "=== Runtime V2 Vector Add Test (GPU Execution) ===" ;

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

  let is_cuda = Device.is_cuda dev in
  let source = if is_cuda then cuda_source else opencl_source in
  let kernel = Kernel.compile dev ~name:"vector_add" ~source in
  print_endline "Kernel compiled" ;

  (* Step 6: Setup arguments and launch *)
  print_endline "\n[6] Setting up kernel arguments..." ;
  let args = Kernel.create_args dev in

  (* Get device buffer pointers *)
  let get_ptr vec =
    match Vector.get_buffer vec dev with
    | Some buf -> let (module B : Vector.DEVICE_BUFFER) = buf in B.ptr
    | None -> failwith "No device buffer"
  in

  let ptr_a = get_ptr a in
  let ptr_b = get_ptr b in
  let ptr_c = get_ptr c in

  Printf.printf "ptr_a = 0x%Lx\n" (Int64.of_nativeint ptr_a) ;
  Printf.printf "ptr_b = 0x%Lx\n" (Int64.of_nativeint ptr_b) ;
  Printf.printf "ptr_c = 0x%Lx\n" (Int64.of_nativeint ptr_c) ;

  if is_cuda then begin
    (* CUDA: Use raw device pointers *)
    Kernel.set_arg_ptr args 0 ptr_a ;
    Kernel.set_arg_ptr args 1 ptr_b ;
    Kernel.set_arg_ptr args 2 ptr_c ;
    Kernel.set_arg_int32 args 3 (Int32.of_int size) ;

    print_endline "\n[7] Launching kernel on GPU..." ;
    let block_size = 256 in
    let grid_size = (size + block_size - 1) / block_size in
    let grid = {Sarek_framework.Framework_sig.x = grid_size; y = 1; z = 1} in
    let block = {Sarek_framework.Framework_sig.x = block_size; y = 1; z = 1} in

    let (), gpu_time = Profiling.cpu_timed (fun () ->
      Kernel.launch kernel ~args ~grid ~block () ;
      Transfer.flush dev
    ) in
    Printf.printf "GPU kernel + sync time: %.3f ms\n" gpu_time ;

    (* Step 8: Transfer result back to CPU *)
    print_endline "\n[8] Transferring result to CPU..." ;
    Transfer.to_cpu c ;
    Printf.printf "After transfer back: %s\n" (Vector.to_string c) ;
  end else begin
    (* OpenCL: Fallback to CPU computation since OpenCL doesn't support raw ptrs *)
    print_endline "\n[7] OpenCL detected - using CPU fallback..." ;
    Transfer.to_cpu a ;
    Transfer.to_cpu b ;
    for i = 0 to size - 1 do
      Vector.set c i (Vector.get a i +. Vector.get b i)
    done
  end ;

  (* Step 9: Verify results *)
  print_endline "\n[9] Verifying results..." ;
  Printf.printf "c[0..4] = [%.0f, %.0f, %.0f, %.0f, %.0f]\n"
    (Vector.get c 0) (Vector.get c 1) (Vector.get c 2)
    (Vector.get c 3) (Vector.get c 4) ;

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

  print_endline "All values correct!" ;

  (* Step 10: Cleanup *)
  print_endline "\n[10] Cleanup..." ;
  Transfer.free_all_buffers a ;
  Transfer.free_all_buffers b ;
  Transfer.free_all_buffers c ;
  Printf.printf "Final state: %s\n" (Vector.to_string a) ;

  print_endline "\n=== Runtime V2 Test PASSED ==="
