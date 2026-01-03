(******************************************************************************
 * E2E test: External kernel execution via run_source
 *
 * Tests running pre-written CUDA/OpenCL kernels through the V2 runtime
 * without using the Sarek DSL.
 ******************************************************************************)

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

(** OpenCL vector add kernel source *)
let opencl_vector_add =
  {|
__kernel void vector_add(
    __global const float* a,
    int a_len,
    __global const float* b,
    int b_len,
    __global float* c,
    int c_len,
    int n)
{
    int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
|}

(** CUDA vector add kernel source *)
let cuda_vector_add =
  {|
extern "C" __global__ void vector_add(
    const float* a,
    int a_len,
    const float* b,
    int b_len,
    float* c,
    int c_len,
    int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
|}

(** Run external kernel test on a device *)
let run_test dev =
  let n = cfg.size in
  let a = V2_Vector.create V2_Vector.float32 n in
  let b = V2_Vector.create V2_Vector.float32 n in
  let c = V2_Vector.create V2_Vector.float32 n in

  (* Initialize vectors *)
  for i = 0 to n - 1 do
    V2_Vector.set a i (float_of_int i) ;
    V2_Vector.set b i (float_of_int (i * 2)) ;
    V2_Vector.set c i 0.0
  done ;

  let threads = cfg.block_size in
  let grid_x = (n + threads - 1) / threads in

  (* Select source based on device framework *)
  let source, lang =
    match dev.V2_Device.framework with
    | "CUDA" -> (cuda_vector_add, Sarek.Execute.CUDA_Source)
    | "OpenCL" -> (opencl_vector_add, Sarek.Execute.OpenCL_Source)
    | fw -> failwith ("External kernels not supported on " ^ fw)
  in

  let t0 = Unix.gettimeofday () in

  (* Run external kernel *)
  Sarek.Execute.run_source
    ~device:dev
    ~source
    ~lang
    ~kernel_name:"vector_add"
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    [
      Sarek.Execute.Vec a;
      Sarek.Execute.Vec b;
      Sarek.Execute.Vec c;
      Sarek.Execute.Int32 (Int32.of_int n);
    ] ;

  V2_Transfer.flush dev ;
  V2_Transfer.to_cpu c ;

  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Verify results *)
  let ok = ref true in
  for i = 0 to n - 1 do
    let expected = float_of_int i +. float_of_int (i * 2) in
    let got = V2_Vector.get c i in
    if abs_float (got -. expected) > 1e-3 then begin
      ok := false ;
      if i < 5 then
        Printf.printf "  Mismatch at %d: got %f expected %f\n%!" i got expected
    end
  done ;

  (!ok, time_ms)

(** Test that Native/Interpreter correctly reject external kernels *)
let test_rejection dev =
  let n = 10 in
  let a = V2_Vector.create V2_Vector.float32 n in
  try
    Sarek.Execute.run_source
      ~device:dev
      ~source:opencl_vector_add
      ~lang:Sarek.Execute.OpenCL_Source
      ~kernel_name:"vector_add"
      ~block:(Sarek.Execute.dims1d 1)
      ~grid:(Sarek.Execute.dims1d 1)
      [Sarek.Execute.Vec a] ;
    (* Should have raised an error *)
    false
  with
  | Sarek.Execute.Execution_error _ -> true
  | Failure _ -> true

let () =
  let c = Test_helpers.parse_args "test_external_kernel" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  print_endline "=== External Kernel Test ===" ;
  let devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices devs ;

  (* Test external kernel execution on GPU devices *)
  print_endline "\nTesting external kernel execution:" ;
  let gpu_devs =
    Array.to_list devs
    |> List.filter (fun d ->
        d.V2_Device.framework = "CUDA" || d.V2_Device.framework = "OpenCL")
    |> fun devs ->
    (* If -d flag specified, filter to just that device *)
    if cfg.dev_id >= 0 && cfg.dev_id < List.length devs then
      [List.nth devs cfg.dev_id]
    else devs
  in

  if List.length gpu_devs = 0 then
    print_endline "  No GPU devices available for external kernel test"
  else
    List.iter
      (fun dev ->
        Printf.printf "  [%s] %s: %!" dev.V2_Device.framework dev.V2_Device.name ;
        try
          let ok, time = run_test dev in
          Printf.printf
            "%.2f ms, %s\n%!"
            time
            (if ok then "PASSED" else "FAILED") ;
          if not ok then exit 1
        with e ->
          Printf.printf "FAIL (%s)\n%!" (Printexc.to_string e) ;
          exit 1)
      gpu_devs ;

  (* Test that Native/Interpreter reject external kernels *)
  print_endline "\nTesting rejection on non-GPU backends:" ;
  let non_gpu_devs =
    Array.to_list devs
    |> List.filter (fun d ->
        d.V2_Device.framework = "Native"
        || d.V2_Device.framework = "Interpreter")
  in

  List.iter
    (fun dev ->
      Printf.printf "  [%s] %s: %!" dev.V2_Device.framework dev.V2_Device.name ;
      if test_rejection dev then print_endline "correctly rejected"
      else begin
        print_endline "FAIL (should have been rejected)" ;
        exit 1
      end)
    non_gpu_devs ;

  print_endline "\n=== All tests PASSED ==="
