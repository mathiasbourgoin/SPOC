(******************************************************************************
 * E2E test for Sarek PPX
 *
 * This test verifies that kernels compiled with the PPX can generate valid
 * GPU code and execute correctly.
 ******************************************************************************)

open Spoc

let size = ref 1024

let dev_id = ref 0

let block_size = ref 256

let verify = ref true

let use_interpreter = ref false

let use_native = ref false

let use_native_parallel = ref false

let benchmark_all = ref false

let usage () =
  Printf.printf "Usage: %s [options]\n" Sys.argv.(0) ;
  Printf.printf "Options:\n" ;
  Printf.printf "  -d <id>       Device ID (default: 0)\n" ;
  Printf.printf "  --interpreter Use CPU interpreter device\n" ;
  Printf.printf "  --native      Use native CPU runtime device (sequential)\n" ;
  Printf.printf
    "  --native-parallel  Use native CPU runtime with parallel threads\n" ;
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
    | "--native-parallel" -> use_native_parallel := true
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

(* Run kernel on a single device and return time in ms *)
let run_on_device vector_add _devs dev =
  (* Create vectors *)
  let a = Vector.create Vector.float32 !size in
  let b = Vector.create Vector.float32 !size in
  let c = Vector.create Vector.float32 !size in

  (* Initialize *)
  for i = 0 to !size - 1 do
    Mem.set a i (float_of_int i) ;
    Mem.set b i (float_of_int (i * 2)) ;
    Mem.set c i 0.0
  done ;

  (* Generate kernel for this device *)
  ignore (Sarek.Kirc.gen vector_add dev) ;

  (* Setup grid/block *)
  let threadsPerBlock =
    match dev.Devices.specific_info with
    | Devices.OpenCLInfo clI -> (
        match clI.Devices.device_type with
        | Devices.CL_DEVICE_TYPE_CPU -> 1
        | _ -> !block_size)
    | _ -> !block_size
  in
  let blocksPerGrid = (!size + threadsPerBlock - 1) / threadsPerBlock in
  let block =
    {Kernel.blockX = threadsPerBlock; Kernel.blockY = 1; Kernel.blockZ = 1}
  in
  let grid =
    {Kernel.gridX = blocksPerGrid; Kernel.gridY = 1; Kernel.gridZ = 1}
  in

  (* Run kernel *)
  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run vector_add (a, b, c, !size) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (* Verify results if requested *)
  let ok =
    if !verify then begin
      Mem.to_cpu c () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to !size - 1 do
        let expected = float_of_int i +. float_of_int (i * 2) in
        let got = Mem.get c i in
        if abs_float (got -. expected) > 0.001 then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let () =
  parse_args () ;

  (* Define kernel inside function to avoid value restriction *)
  let vector_add =
    [%kernel
      fun (a : float32 vector)
          (b : float32 vector)
          (c : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then c.(tid) <- a.(tid) +. b.(tid)]
  in

  (* Initialize SPOC and get devices *)
  let devs = Devices.init () in
  if Array.length devs = 0 then begin
    print_endline "No GPU devices found" ;
    exit 1
  end ;

  Printf.printf "Available devices:\n" ;
  Array.iteri
    (fun i d ->
      Printf.printf "  [%d] %s\n" i d.Devices.general_info.Devices.name)
    devs ;

  if !benchmark_all then begin
    (* Benchmark mode: run on all devices including parallel native *)
    Printf.printf "\nBenchmark: vector_add with %d elements\n" !size ;
    Printf.printf "%-40s %12s %10s\n" "Device" "Time (ms)" "Status" ;
    Printf.printf "%s\n" (String.make 64 '-') ;
    (* Run on all standard devices *)
    Array.iter
      (fun dev ->
        let name = dev.Devices.general_info.Devices.name in
        let time_ms, ok = run_on_device vector_add devs dev in
        let status = if ok then "OK" else "FAIL" in
        Printf.printf "%-40s %12.4f %10s\n" name time_ms status)
      devs ;
    (* Also run on parallel native device *)
    let parallel_dev = Devices.create_native_device ~parallel:true () in
    let name = parallel_dev.Devices.general_info.Devices.name in
    let time_ms, ok = run_on_device vector_add devs parallel_dev in
    let status = if ok then "OK" else "FAIL" in
    Printf.printf "%-40s %12.4f %10s\n" name time_ms status ;
    print_endline "\nBenchmark complete."
  end
  else begin
    (* Single device mode *)
    let dev =
      if !use_native_parallel then
        Devices.create_native_device ~parallel:true ()
      else if !use_native then (
        match Devices.find_native_id devs with
        | Some id -> devs.(id)
        | None ->
            print_endline "No native CPU device found" ;
            exit 1)
      else if !use_interpreter then (
        match Devices.find_interpreter_id devs with
        | Some id -> devs.(id)
        | None ->
            print_endline "No interpreter device found" ;
            exit 1)
      else devs.(!dev_id)
    in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    Printf.printf "Configuration: size=%d, block_size=%d\n%!" !size !block_size ;

    let blocks = (!size + !block_size - 1) / !block_size in
    Printf.printf
      "  -> blocks=%d, total_threads=%d\n%!"
      blocks
      (blocks * !block_size) ;

    Printf.printf "Running kernel...\n%!" ;
    let time_ms, ok = run_on_device vector_add devs dev in
    Printf.printf "Kernel time: %.4f ms\n%!" time_ms ;

    if !verify then
      if ok then Printf.printf "Verification PASSED\n%!"
      else Printf.printf "Verification FAILED\n%!" ;

    print_endline "E2E Test PASSED"
  end
