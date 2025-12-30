(******************************************************************************
 * Test for the new Sarek Runtime with ctypes plugins
 *
 * This test verifies that the new plugin-based runtime can:
 * 1. Discover available devices via the plugin architecture
 * 2. Compile GPU kernels using NVRTC/OpenCL
 * 3. Execute kernels and transfer data
 ******************************************************************************)

(* Force plugin initialization *)
let () = Sarek_cuda.Cuda_plugin.init ()

let () = Sarek_opencl.Opencl_plugin.init ()

let () =
  print_endline "=== Sarek Runtime Test ===" ;

  (* Test 1: Device discovery *)
  print_endline "\n[1] Testing device discovery..." ;
  let devices = Sarek_core.Device.init () in
  Printf.printf "Found %d device(s)\n" (Array.length devices) ;
  Array.iter
    (fun d -> Printf.printf "  - %s\n" (Sarek_core.Device.to_string d))
    devices ;

  if Array.length devices = 0 then begin
    print_endline "No GPU devices found - skipping GPU tests" ;
    print_endline "=== Test PASSED (no GPU) ===" ;
    exit 0
  end ;

  (* Test 2: Get best device *)
  print_endline "\n[2] Testing device selection..." ;
  let dev =
    match Sarek_core.Device.best () with
    | Some d -> d
    | None -> failwith "No device available"
  in
  Printf.printf "Selected device: %s (%s)\n" dev.name dev.framework ;

  (* Test 3: Memory allocation *)
  print_endline "\n[3] Testing memory allocation..." ;
  let size = 1024 in
  let buf = Sarek_core.Runtime.alloc_float32 dev size in
  Printf.printf "Allocated buffer of %d float32 elements\n" size ;

  (* Test 4: Data transfer *)
  print_endline "\n[4] Testing host-to-device transfer..." ;
  let host_data =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout size
  in
  for i = 0 to size - 1 do
    Bigarray.Array1.set host_data i (float_of_int i)
  done ;
  Sarek_core.Runtime.to_device ~src:host_data ~dst:buf ;
  print_endline "Host-to-device transfer complete" ;

  (* Test 5: Kernel compilation *)
  print_endline "\n[5] Testing kernel compilation..." ;
  let kernel_source =
    match dev.framework with
    | "CUDA" ->
        {|
extern "C" __global__ void add_one(float* arr, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    arr[idx] = arr[idx] + 1.0f;
  }
}
|}
    | "OpenCL" ->
        {|
__kernel void add_one(__global float* arr, int n) {
  int idx = get_global_id(0);
  if (idx < n) {
    arr[idx] = arr[idx] + 1.0f;
  }
}
|}
    | _ -> failwith ("Unknown framework: " ^ dev.framework)
  in
  let _kernel =
    Sarek_core.Kernel.compile dev ~name:"add_one" ~source:kernel_source
  in
  print_endline "Kernel compiled successfully" ;

  (* Test 6: Kernel execution *)
  print_endline "\n[6] Testing kernel execution..." ;
  let block = Sarek_core.Runtime.dims1d 256 in
  let grid = Sarek_core.Runtime.dims1d ((size + 255) / 256) in
  Sarek_core.Runtime.run
    dev
    ~name:"add_one"
    ~source:kernel_source
    ~args:
      [
        Sarek_core.Runtime.ArgBuffer buf;
        Sarek_core.Runtime.ArgInt32 (Int32.of_int size);
      ]
    ~grid
    ~block
    () ;
  print_endline "Kernel executed" ;

  (* Test 7: Device-to-host transfer and verification *)
  print_endline "\n[7] Testing device-to-host transfer and verification..." ;
  let result = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout size in
  Sarek_core.Runtime.from_device ~src:buf ~dst:result ;

  let errors = ref 0 in
  for i = 0 to size - 1 do
    let expected = float_of_int i +. 1.0 in
    let got = Bigarray.Array1.get result i in
    if abs_float (got -. expected) > 0.001 then begin
      if !errors < 5 then
        Printf.printf
          "  Mismatch at %d: expected %.2f, got %.2f\n"
          i
          expected
          got ;
      incr errors
    end
  done ;

  if !errors > 0 then begin
    Printf.printf "Total errors: %d\n" !errors ;
    print_endline "=== Test FAILED ===" ;
    exit 1
  end ;

  (* Cleanup *)
  Sarek_core.Runtime.free buf ;

  print_endline "\n=== Test PASSED ==="
