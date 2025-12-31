(******************************************************************************
 * E2E test for Native plugin integration with new runtime
 *
 * This test verifies that:
 * 1. Native plugin can be used through the new runtime
 * 2. Kernels registered with Native_plugin.register_kernel are executed
 * 3. SPOC custom type vectors work with the native plugin
 * 4. Results are correct
 *
 * NOTE ON BUFFER ARCHITECTURE (Phase 4 TODO):
 *
 * Currently this test uses two separate buffer systems:
 * - sarek_core.Memory.buffer for simple numeric arrays (new runtime)
 * - Spoc.Vector for custom types (old SPOC infrastructure)
 *
 * This is a temporary situation. In Phase 4, we should unify these into a
 * single buffer type that:
 *
 * 1. Uses ctypes raw memory (works on all backends: CUDA, OpenCL, Native)
 * 2. Has an element descriptor:
 *    - Numeric: size + Bigarray kind (for float32, int32, etc.)
 *    - Custom: size + get/set functions (for records like point, float4, etc.)
 * 3. Is completely backend-agnostic - the buffer doesn't know about frameworks
 * 4. Supports multi-device: same buffer can have copies on multiple devices
 * 5. Tracks dirty state to know which copies need synchronization
 *
 * The key insight is that custom types work IDENTICALLY on all backends:
 * - Same memory layout (ctypes-allocated raw memory)
 * - Same get/set functions
 * - Only difference is where memory lives (GPU vs host) and transfer ops
 *
 * For now, we use SPOC Vector for custom types since it already works.
 ******************************************************************************)

open Spoc

(* Force plugin initialization *)
let () = Sarek_native.Native_plugin.init ()

let size = 1024

type float32 = float

(* Custom record type for 2D points - PPX generates point_custom for Vector.Custom *)
type point = {x : float32; y : float32} [@@sarek.type]

(* A kernel that transforms points: dst[i] = {x = src[i].x + 1; y = src[i].y * 2} *)
let transform_points_kernel args (gx, _gy, _gz) (bx, _by, _bz) =
  (* Extract arguments from Obj.t array - these are SPOC customarrays *)
  let src : Vector.customarray = Obj.obj args.(0) in
  let dst : Vector.customarray = Obj.obj args.(1) in
  let n : int32 = Obj.obj args.(2) in
  let n = Int32.to_int n in

  (* Compute total threads and iterate *)
  let total_threads = gx * bx in
  for tid = 0 to total_threads - 1 do
    if tid < n then begin
      let p : point = point_custom.get src tid in
      point_custom.set dst tid {x = p.x +. 1.0; y = p.y *. 2.0}
    end
  done

(* Register the kernel *)
let () =
  Sarek_native.Native_plugin.register_kernel
    "transform_points"
    transform_points_kernel

(* A simple kernel function that adds vectors: c[i] = a[i] + b[i] *)
let vector_add_kernel args (gx, _gy, _gz) (bx, _by, _bz) =
  (* Extract arguments from Obj.t array *)
  let a : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t =
    Obj.obj args.(0)
  in
  let b : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t =
    Obj.obj args.(1)
  in
  let c : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t =
    Obj.obj args.(2)
  in
  let n : int32 = Obj.obj args.(3) in
  let n = Int32.to_int n in

  (* Compute total threads and iterate *)
  let total_threads = gx * bx in
  for tid = 0 to total_threads - 1 do
    if tid < n then
      Bigarray.Array1.set
        c
        tid
        (Bigarray.Array1.get a tid +. Bigarray.Array1.get b tid)
  done

(* Register the kernel *)
let () =
  Sarek_native.Native_plugin.register_kernel "vector_add" vector_add_kernel

let () =
  print_endline "=== Native Runtime Integration Test ===" ;

  (* Step 1: Initialize native devices *)
  print_endline "\n[1] Initializing Native devices..." ;
  let devices = Sarek_core.Device.init ~frameworks:["Native"] () in
  Printf.printf "Found %d Native device(s)\n" (Array.length devices) ;
  if Array.length devices = 0 then begin
    print_endline "No Native devices found - test failed" ;
    exit 1
  end ;
  let dev = devices.(0) in
  Printf.printf "Using device: %s (%s)\n" dev.name dev.framework ;

  (* Step 2: Check kernel is registered *)
  print_endline "\n[2] Checking kernel registration..." ;
  let registered = Sarek_native.Native_plugin.kernel_registered "vector_add" in
  Printf.printf "Kernel 'vector_add' registered: %b\n" registered ;
  if not registered then begin
    print_endline "Kernel not registered - test failed" ;
    exit 1
  end ;

  (* Step 3: Allocate buffers *)
  print_endline "\n[3] Allocating buffers..." ;
  let buf_a = Sarek_core.Runtime.alloc_float32 dev size in
  let buf_b = Sarek_core.Runtime.alloc_float32 dev size in
  let buf_c = Sarek_core.Runtime.alloc_float32 dev size in
  print_endline "Buffers allocated" ;

  (* Step 4: Initialize data *)
  print_endline "\n[4] Initializing data..." ;
  let host_a = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout size in
  let host_b = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout size in
  let host_c = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout size in
  for i = 0 to size - 1 do
    Bigarray.Array1.set host_a i (float_of_int i) ;
    Bigarray.Array1.set host_b i (float_of_int (i * 2)) ;
    Bigarray.Array1.set host_c i 0.0
  done ;
  Sarek_core.Runtime.to_device ~src:host_a ~dst:buf_a ;
  Sarek_core.Runtime.to_device ~src:host_b ~dst:buf_b ;
  Sarek_core.Runtime.to_device ~src:host_c ~dst:buf_c ;
  print_endline "Data initialized and transferred" ;

  (* Step 5: Execute kernel *)
  print_endline "\n[5] Executing kernel via new runtime..." ;
  let block = Sarek_core.Runtime.dims1d 256 in
  let grid = Sarek_core.Runtime.dims1d ((size + 255) / 256) in

  Sarek_core.Runtime.run
    dev
    ~name:"vector_add"
    ~source:"" (* Native doesn't need source *)
    ~args:
      [
        Sarek_core.Runtime.ArgBuffer buf_a;
        Sarek_core.Runtime.ArgBuffer buf_b;
        Sarek_core.Runtime.ArgBuffer buf_c;
        Sarek_core.Runtime.ArgInt32 (Int32.of_int size);
      ]
    ~grid
    ~block
    () ;
  print_endline "Kernel executed" ;

  (* Step 6: Verify results *)
  print_endline "\n[6] Verifying results..." ;
  Sarek_core.Runtime.from_device ~src:buf_c ~dst:host_c ;

  let errors = ref 0 in
  for i = 0 to size - 1 do
    let expected = float_of_int i +. float_of_int (i * 2) in
    let got = Bigarray.Array1.get host_c i in
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
  Sarek_core.Runtime.free buf_a ;
  Sarek_core.Runtime.free buf_b ;
  Sarek_core.Runtime.free buf_c ;

  print_endline "Vector add test passed!" ;

  (* ============================================================ *)
  (* Part 2: Custom type test using SPOC Vector                   *)
  (* ============================================================ *)
  print_endline "\n[7] Testing custom types with SPOC Vector..." ;

  let n_points = 100 in

  (* Create custom type vectors using SPOC infrastructure *)
  let src = Vector.create (Vector.Custom point_custom) n_points in
  let dst = Vector.create (Vector.Custom point_custom) n_points in

  (* Initialize source points *)
  for i = 0 to n_points - 1 do
    Mem.set src i {x = float_of_int i; y = float_of_int (i * 2)}
  done ;
  print_endline "Custom type vectors created and initialized" ;

  (* Check kernel is registered *)
  let registered =
    Sarek_native.Native_plugin.kernel_registered "transform_points"
  in
  Printf.printf "Kernel 'transform_points' registered: %b\n" registered ;

  (* Extract the underlying customarrays from SPOC Vectors *)
  let src_data =
    match Vector.vector src with
    | Vector.CustomArray (arr, _) -> arr
    | _ -> failwith "Expected CustomArray"
  in
  let dst_data =
    match Vector.vector dst with
    | Vector.CustomArray (arr, _) -> arr
    | _ -> failwith "Expected CustomArray"
  in

  (* Run kernel using run_kernel_raw *)
  print_endline "\n[8] Executing custom type kernel..." ;
  let grid = ((n_points + 255) / 256, 1, 1) in
  let block = (256, 1, 1) in
  Sarek_native.Native_plugin.run_kernel_raw
    ~name:"transform_points"
    ~args:
      [|Obj.repr src_data; Obj.repr dst_data; Obj.repr (Int32.of_int n_points)|]
    ~grid
    ~block ;
  print_endline "Custom type kernel executed" ;

  (* Verify results *)
  print_endline "\n[9] Verifying custom type results..." ;
  let errors = ref 0 in
  for i = 0 to n_points - 1 do
    let expected_x = float_of_int i +. 1.0 in
    let expected_y = float_of_int (i * 2) *. 2.0 in
    let result = Mem.get dst i in
    if
      abs_float (result.x -. expected_x) > 1e-3
      || abs_float (result.y -. expected_y) > 1e-3
    then begin
      if !errors < 5 then
        Printf.printf
          "  Error at %d: expected {x=%.1f; y=%.1f}, got {x=%.1f; y=%.1f}\n"
          i
          expected_x
          expected_y
          result.x
          result.y ;
      incr errors
    end
  done ;

  if !errors > 0 then begin
    Printf.printf "Custom type test: %d errors\n" !errors ;
    print_endline "=== Test FAILED ===" ;
    exit 1
  end ;

  print_endline "Custom type test passed!" ;
  print_endline "\n=== All Tests PASSED ==="
