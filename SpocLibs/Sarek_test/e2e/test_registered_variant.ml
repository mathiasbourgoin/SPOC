(******************************************************************************
 * E2E test: register a Sarek variant type outside kernels via [@@sarek.type].
 * Uses V2 runtime only.
 ******************************************************************************)

module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init () ;
  Sarek_native.Native_plugin_v2.init () ;
  Sarek_interpreter.Interpreter_plugin_v2.init ()

let cfg = Test_helpers.default_config ()

type float32 = float

type color = Red | Value of float32 [@@sarek.type]

let variant_kernel =
  snd
    [%kernel
      fun (xs : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let c = if xs.(tid) > 0.0 then Value xs.(tid) else Red in
          match c with
          | Red -> dst.(tid) <- 0.0
          | Value v -> dst.(tid) <- v +. 1.0]

let run_test dev =
  let ir =
    match variant_kernel.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in
  let n = cfg.size in
  let xs = V2_Vector.create V2_Vector.float32 n in
  let dst = V2_Vector.create V2_Vector.float32 n in
  for i = 0 to n - 1 do
    V2_Vector.set xs i (if i mod 2 = 0 then -1.0 else float_of_int i) ;
    V2_Vector.set dst i 0.0
  done ;

  let threads = cfg.block_size in
  let grid_x = (n + threads - 1) / threads in
  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Sarek.Execute.Vec xs; Sarek.Execute.Vec dst; Sarek.Execute.Int32 (Int32.of_int n)]
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok = ref true in
  for i = 0 to n - 1 do
    let expected = if i mod 2 = 0 then 0.0 else float_of_int i +. 1.0 in
    let got = V2_Vector.get dst i in
    if abs_float (got -. expected) > 1e-3 then (
      ok := false ;
      if i < 5 then
        Printf.printf "  Mismatch at %d: got %f expected %f\n%!" i got expected)
  done ;
  (!ok, time_ms)

let () =
  let c = Test_helpers.parse_args "test_registered_variant" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  (* Print IR for verification *)
  if cfg.dev_id < 0 then begin
    print_endline "=== Registered Variant IR ===" ;
    Sarek.Kirc_Ast.print_ast variant_kernel.Sarek.Kirc_types.body ;
    print_endline "=============================="
  end ;

  print_endline "=== Registered Variant Test ===" ;
  let devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices devs ;

  if cfg.benchmark_all || Option.is_some cfg.benchmark_devices then begin
    (* Run on all devices *)
    print_endline "\nBenchmarking on all devices:" ;
    Array.iteri (fun i dev ->
      Printf.printf "\n[%d] %s:\n%!" i dev.V2_Device.name ;
      try
        let ok, time = run_test dev in
        Printf.printf "  %.2f ms, %s\n%!" time (if ok then "PASSED" else "FAILED") ;
        if not ok then exit 1
      with e ->
        Printf.printf "  FAIL (%s)\n%!" (Printexc.to_string e) ;
        exit 1)
      devs
  end else begin
    (* Run on selected device *)
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "\nUsing device: %s\n%!" dev.V2_Device.name ;
    let ok, time = run_test dev in
    Printf.printf "  %.2f ms, %s\n%!" time (if ok then "PASSED" else "FAILED") ;
    if not ok then exit 1
  end ;

  print_endline "\n=== All tests PASSED ==="
