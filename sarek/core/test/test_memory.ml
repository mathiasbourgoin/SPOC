(******************************************************************************
 * Unit tests for Memory module
 *
 * Tests buffer types, allocation API, and transfer operations.
 * Note: Most operations require a backend, so we test structure and
 * error handling paths.
 ******************************************************************************)

(** {1 Buffer Type Tests} *)

let test_buffer_type_structure () =
  (* Verify buffer record fields exist and have correct types
     by checking we can construct a dummy value *)
  let fake_caps : Spoc_framework.Framework_sig.capabilities = {
    max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    supports_fp64 = true;
    supports_atomics = true;
    warp_size = 32;
    max_registers_per_block = 16384;
    clock_rate_khz = 1000000;
    multiprocessor_count = 4;
    is_cpu = true;
  } in
  let fake_device : Spoc_core.Device.t = {
    id = 0;
    backend_id = 0;
    name = "Test";
    framework = "Test";
    capabilities = fake_caps;
  } in
  (* This compiles only if buffer has expected fields *)
  let buf : float Spoc_core.Memory.buffer = {
    device = fake_device;
    size = 100;
    elem_size = 4;
    handle = Obj.repr ();
  } in
  assert (buf.size = 100);
  assert (buf.elem_size = 4);
  print_endline "  buffer type structure: OK"

(** {1 Size/Device Accessor Tests} *)

let test_size_accessor () =
  let fake_caps : Spoc_framework.Framework_sig.capabilities = {
    max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    supports_fp64 = true;
    supports_atomics = true;
    warp_size = 32;
    max_registers_per_block = 16384;
    clock_rate_khz = 1000000;
    multiprocessor_count = 4;
    is_cpu = true;
  } in
  let fake_device : Spoc_core.Device.t = {
    id = 0;
    backend_id = 0;
    name = "Test";
    framework = "Test";
    capabilities = fake_caps;
  } in
  let buf : float Spoc_core.Memory.buffer = {
    device = fake_device;
    size = 200;
    elem_size = 8;
    handle = Obj.repr ();
  } in
  assert (Spoc_core.Memory.size buf = 200);
  assert ((Spoc_core.Memory.device buf).name = "Test");
  print_endline "  size/device accessors: OK"

(** {1 Allocation API Signature Tests} *)

let test_alloc_signature () =
  (* Verify alloc has correct signature by referencing it *)
  let _f : Spoc_core.Device.t -> int -> ('a, 'b) Bigarray.kind -> 'a Spoc_core.Memory.buffer =
    Spoc_core.Memory.alloc
  in
  print_endline "  alloc signature: OK"

let test_alloc_custom_signature () =
  let _f : Spoc_core.Device.t -> size:int -> elem_size:int -> 'a Spoc_core.Memory.buffer =
    Spoc_core.Memory.alloc_custom
  in
  print_endline "  alloc_custom signature: OK"

let test_free_signature () =
  let _f : 'a Spoc_core.Memory.buffer -> unit = Spoc_core.Memory.free in
  print_endline "  free signature: OK"

(** {1 Transfer API Signature Tests} *)

let test_host_to_device_signature () =
  let _f : src:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
           dst:'a Spoc_core.Memory.buffer -> unit =
    Spoc_core.Memory.host_to_device
  in
  print_endline "  host_to_device signature: OK"

let test_device_to_host_signature () =
  let _f : src:'a Spoc_core.Memory.buffer ->
           dst:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> unit =
    Spoc_core.Memory.device_to_host
  in
  print_endline "  device_to_host signature: OK"

let test_host_ptr_to_device_signature () =
  let _f : src_ptr:unit Ctypes.ptr -> dst:'a Spoc_core.Memory.buffer -> unit =
    Spoc_core.Memory.host_ptr_to_device
  in
  print_endline "  host_ptr_to_device signature: OK"

let test_device_to_host_ptr_signature () =
  let _f : src:'a Spoc_core.Memory.buffer -> dst_ptr:unit Ctypes.ptr -> unit =
    Spoc_core.Memory.device_to_host_ptr
  in
  print_endline "  device_to_host_ptr signature: OK"

let test_device_to_device_signature () =
  let _f : src:'a Spoc_core.Memory.buffer -> dst:'a Spoc_core.Memory.buffer -> unit =
    Spoc_core.Memory.device_to_device
  in
  print_endline "  device_to_device signature: OK"

(** {1 Error Handling Tests} *)

let test_alloc_unknown_framework () =
  (* Create a device with unknown framework *)
  let fake_caps : Spoc_framework.Framework_sig.capabilities = {
    max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    supports_fp64 = true;
    supports_atomics = true;
    warp_size = 32;
    max_registers_per_block = 16384;
    clock_rate_khz = 1000000;
    multiprocessor_count = 4;
    is_cpu = true;
  } in
  let fake_device : Spoc_core.Device.t = {
    id = 999;
    backend_id = 0;
    name = "Fake Device";
    framework = "NonExistentFramework";
    capabilities = fake_caps;
  } in
  let raised = ref false in
  (try
    let _ = Spoc_core.Memory.alloc fake_device 100 Bigarray.float32 in
    ()
  with Failure msg ->
    if String.sub msg 0 18 = "Unknown framework:" then
      raised := true);
  assert !raised;
  print_endline "  alloc unknown framework error: OK"

let test_alloc_custom_unknown_framework () =
  let fake_caps : Spoc_framework.Framework_sig.capabilities = {
    max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    supports_fp64 = true;
    supports_atomics = true;
    warp_size = 32;
    max_registers_per_block = 16384;
    clock_rate_khz = 1000000;
    multiprocessor_count = 4;
    is_cpu = true;
  } in
  let fake_device : Spoc_core.Device.t = {
    id = 999;
    backend_id = 0;
    name = "Fake Device";
    framework = "NonExistentFramework";
    capabilities = fake_caps;
  } in
  let raised = ref false in
  (try
    let _ = Spoc_core.Memory.alloc_custom fake_device ~size:100 ~elem_size:4 in
    ()
  with Failure msg ->
    if String.sub msg 0 18 = "Unknown framework:" then
      raised := true);
  assert !raised;
  print_endline "  alloc_custom unknown framework error: OK"

(** {1 Main} *)

let () =
  print_endline "Memory module tests:";
  test_buffer_type_structure ();
  test_size_accessor ();
  test_alloc_signature ();
  test_alloc_custom_signature ();
  test_free_signature ();
  test_host_to_device_signature ();
  test_device_to_host_signature ();
  test_host_ptr_to_device_signature ();
  test_device_to_host_ptr_signature ();
  test_device_to_device_signature ();
  test_alloc_unknown_framework ();
  test_alloc_custom_unknown_framework ();
  print_endline "All Memory module tests passed!"
