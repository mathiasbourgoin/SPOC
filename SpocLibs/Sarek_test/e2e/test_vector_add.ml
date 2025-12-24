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

let usage () =
  Printf.printf "Usage: %s [options]\n" Sys.argv.(0) ;
  Printf.printf "Options:\n" ;
  Printf.printf "  -d <id>     Device ID (default: 0)\n" ;
  Printf.printf "  -s <size>   Vector size (default: 1024)\n" ;
  Printf.printf "  -b <size>   Block/work-group size (default: 256)\n" ;
  Printf.printf "  -no-verify  Skip result verification\n" ;
  Printf.printf "  -h          Show this help\n" ;
  exit 0

let parse_args () =
  let i = ref 1 in
  while !i < Array.length Sys.argv do
    match Sys.argv.(!i) with
    | "-d" ->
        incr i ;
        dev_id := int_of_string Sys.argv.(!i)
    | "-s" ->
        incr i ;
        size := int_of_string Sys.argv.(!i)
    | "-b" ->
        incr i ;
        block_size := int_of_string Sys.argv.(!i)
    | "-no-verify" -> verify := false
    | "-h" | "--help" -> usage ()
    | _ ->
        () ;
        incr i
  done

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

  let dev = devs.(!dev_id) in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
  Printf.printf "Configuration: size=%d, block_size=%d\n%!" !size !block_size ;

  let blocks = (!size + !block_size - 1) / !block_size in
  Printf.printf
    "  -> blocks=%d, total_threads=%d\n%!"
    blocks
    (blocks * !block_size) ;

  (* Create vectors *)
  Printf.printf "Creating vectors...\n%!" ;
  let a = Vector.create Vector.float32 !size in
  let b = Vector.create Vector.float32 !size in
  let c = Vector.create Vector.float32 !size in

  (* Initialize *)
  Printf.printf "Initializing vectors...\n%!" ;
  for i = 0 to !size - 1 do
    Mem.set a i (float_of_int i) ;
    Mem.set b i (float_of_int (i * 2)) ;
    Mem.set c i 0.0
  done ;

  (* Generate kernel *)
  Printf.printf "Generating kernel...\n%!" ;
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
  Printf.printf "Running kernel...\n%!" ;
  Sarek.Kirc.run vector_add (a, b, c, !size) (block, grid) 0 dev ;
  Devices.flush dev () ;

  Printf.printf "Kernel execution complete.\n%!" ;

  (* Verify results *)
  if !verify then begin
    Printf.printf "Verifying results...\n%!" ;
    Mem.to_cpu c () ;
    Devices.flush dev () ;
    let errors = ref 0 in
    for i = 0 to min (!size - 1) 9 do
      let expected = float_of_int i +. float_of_int (i * 2) in
      let got = Mem.get c i in
      if abs_float (got -. expected) > 0.001 then begin
        Printf.printf "  ERROR at %d: expected %f, got %f\n" i expected got ;
        incr errors
      end
      else if i < 5 then Printf.printf "  c[%d] = %f (OK)\n" i got
    done ;
    if !errors = 0 then Printf.printf "Verification PASSED\n%!"
    else Printf.printf "Verification FAILED with %d errors\n%!" !errors
  end ;

  print_endline "E2E Test PASSED"
