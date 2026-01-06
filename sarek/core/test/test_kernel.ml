(******************************************************************************
 * Unit tests for Kernel module
 *
 * Tests kernel compilation, argument handling, and launch API.
 * Note: Most operations require a backend, so we test structure and
 * error handling paths.
 ******************************************************************************)

let make_fake_caps () : Spoc_framework.Framework_sig.capabilities = {
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
}

(** {1 Kernel Type Tests} *)

let test_kernel_type_structure () =
  (* Verify kernel record fields exist and have correct types *)
  let fake_device : Spoc_core.Device.t = {
    id = 0;
    backend_id = 0;
    name = "Test";
    framework = "Test";
    capabilities = make_fake_caps ();
  } in
  let kernel : Spoc_core.Kernel.t = {
    device = fake_device;
    name = "test_kernel";
    handle = Obj.repr ();
  } in
  assert (kernel.name = "test_kernel");
  assert (kernel.device.name = "Test");
  print_endline "  kernel type structure: OK"

let test_args_type_structure () =
  (* Verify args record fields exist and have correct types *)
  let fake_device : Spoc_core.Device.t = {
    id = 0;
    backend_id = 0;
    name = "Test";
    framework = "Test";
    capabilities = make_fake_caps ();
  } in
  let args : Spoc_core.Kernel.args = {
    device = fake_device;
    handle = Obj.repr ();
  } in
  assert (args.device.name = "Test");
  print_endline "  args type structure: OK"

(** {1 Compilation API Signature Tests} *)

let test_compile_signature () =
  let _f : Spoc_core.Device.t -> name:string -> source:string -> Spoc_core.Kernel.t =
    Spoc_core.Kernel.compile
  in
  print_endline "  compile signature: OK"

let test_compile_cached_signature () =
  let _f : Spoc_core.Device.t -> name:string -> source:string -> Spoc_core.Kernel.t =
    Spoc_core.Kernel.compile_cached
  in
  print_endline "  compile_cached signature: OK"

(** {1 Args API Signature Tests} *)

let test_create_args_signature () =
  let _f : Spoc_core.Device.t -> Spoc_core.Kernel.args =
    Spoc_core.Kernel.create_args
  in
  print_endline "  create_args signature: OK"

let test_set_arg_buffer_signature () =
  let _f : Spoc_core.Kernel.args -> int -> 'a Spoc_core.Memory.buffer -> unit =
    Spoc_core.Kernel.set_arg_buffer
  in
  print_endline "  set_arg_buffer signature: OK"

let test_set_arg_int32_signature () =
  let _f : Spoc_core.Kernel.args -> int -> int32 -> unit =
    Spoc_core.Kernel.set_arg_int32
  in
  print_endline "  set_arg_int32 signature: OK"

let test_set_arg_int64_signature () =
  let _f : Spoc_core.Kernel.args -> int -> int64 -> unit =
    Spoc_core.Kernel.set_arg_int64
  in
  print_endline "  set_arg_int64 signature: OK"

let test_set_arg_float32_signature () =
  let _f : Spoc_core.Kernel.args -> int -> float -> unit =
    Spoc_core.Kernel.set_arg_float32
  in
  print_endline "  set_arg_float32 signature: OK"

let test_set_arg_float64_signature () =
  let _f : Spoc_core.Kernel.args -> int -> float -> unit =
    Spoc_core.Kernel.set_arg_float64
  in
  print_endline "  set_arg_float64 signature: OK"

let test_set_arg_ptr_signature () =
  let _f : Spoc_core.Kernel.args -> int -> nativeint -> unit =
    Spoc_core.Kernel.set_arg_ptr
  in
  print_endline "  set_arg_ptr signature: OK"

(** {1 Launch API Signature Tests} *)

let test_launch_signature () =
  let _f : Spoc_core.Kernel.t ->
           args:Spoc_core.Kernel.args ->
           grid:Spoc_framework.Framework_sig.dims ->
           block:Spoc_framework.Framework_sig.dims ->
           ?shared_mem:int ->
           unit -> unit =
    Spoc_core.Kernel.launch
  in
  print_endline "  launch signature: OK"

let test_clear_cache_signature () =
  let _f : Spoc_core.Device.t -> unit = Spoc_core.Kernel.clear_cache in
  print_endline "  clear_cache signature: OK"

(** {1 Error Handling Tests} *)

let make_fake_device () : Spoc_core.Device.t =
  {
    id = 999;
    backend_id = 0;
    name = "Fake Device";
    framework = "NonExistentFramework";
    capabilities = make_fake_caps ();
  }

let test_compile_unknown_framework () =
  let fake_device = make_fake_device () in
  let raised = ref false in
  (try
    let _ = Spoc_core.Kernel.compile fake_device ~name:"test" ~source:"" in
    ()
  with Failure msg ->
    if String.sub msg 0 18 = "Unknown framework:" then
      raised := true);
  assert !raised;
  print_endline "  compile unknown framework error: OK"

let test_compile_cached_unknown_framework () =
  let fake_device = make_fake_device () in
  let raised = ref false in
  (try
    let _ = Spoc_core.Kernel.compile_cached fake_device ~name:"test" ~source:"" in
    ()
  with Failure msg ->
    if String.sub msg 0 18 = "Unknown framework:" then
      raised := true);
  assert !raised;
  print_endline "  compile_cached unknown framework error: OK"

let test_create_args_unknown_framework () =
  let fake_device = make_fake_device () in
  let raised = ref false in
  (try
    let _ = Spoc_core.Kernel.create_args fake_device in
    ()
  with Failure msg ->
    if String.sub msg 0 18 = "Unknown framework:" then
      raised := true);
  assert !raised;
  print_endline "  create_args unknown framework error: OK"

let test_clear_cache_unknown_framework () =
  (* clear_cache should not raise for unknown framework - it's a no-op *)
  let fake_device = make_fake_device () in
  Spoc_core.Kernel.clear_cache fake_device;
  print_endline "  clear_cache unknown framework (no-op): OK"

(** {1 Framework_sig.dims Tests} *)

let test_dims_usage () =
  let grid : Spoc_framework.Framework_sig.dims = { x = 128; y = 1; z = 1 } in
  let block : Spoc_framework.Framework_sig.dims = { x = 256; y = 1; z = 1 } in
  assert (grid.x = 128);
  assert (block.x = 256);
  print_endline "  dims usage: OK"

(** {1 Main} *)

let () =
  print_endline "Kernel module tests:";
  test_kernel_type_structure ();
  test_args_type_structure ();
  test_compile_signature ();
  test_compile_cached_signature ();
  test_create_args_signature ();
  test_set_arg_buffer_signature ();
  test_set_arg_int32_signature ();
  test_set_arg_int64_signature ();
  test_set_arg_float32_signature ();
  test_set_arg_float64_signature ();
  test_set_arg_ptr_signature ();
  test_launch_signature ();
  test_clear_cache_signature ();
  test_compile_unknown_framework ();
  test_compile_cached_unknown_framework ();
  test_create_args_unknown_framework ();
  test_clear_cache_unknown_framework ();
  test_dims_usage ();
  print_endline "All Kernel module tests passed!"
