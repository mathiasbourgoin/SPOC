(******************************************************************************
 * V2 Runtime Comparison Test
 *
 * Runs the same kernel via both SPOC and V2 runtime paths, comparing results
 * to validate that V2 produces correct output.
 ******************************************************************************)

(* Alias modules to avoid conflicts *)
module Spoc_Vector = Spoc.Vector
module Spoc_Devices = Spoc.Devices
module Spoc_Mem = Spoc.Mem
module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer

(* Force backend registration - OCaml won't init modules unless referenced *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

let size = 1024

(* Define kernel using Sarek PPX - works with both SPOC and V2 *)
let vector_add =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) +. b.(tid)]

(* Run via SPOC path and return results *)
let run_spoc (dev : Spoc_Devices.device) : float array =
  (* Create SPOC vectors *)
  let a = Spoc_Vector.create Spoc_Vector.float32 size in
  let b = Spoc_Vector.create Spoc_Vector.float32 size in
  let c = Spoc_Vector.create Spoc_Vector.float32 size in

  (* Initialize *)
  for i = 0 to size - 1 do
    Spoc_Mem.set a i (float_of_int i) ;
    Spoc_Mem.set b i (float_of_int (i * 2)) ;
    Spoc_Mem.set c i 0.0
  done ;

  (* Generate and run kernel *)
  let block_size =
    match dev.Spoc_Devices.specific_info with
    | Spoc_Devices.OpenCLInfo clI -> (
        match clI.Spoc_Devices.device_type with
        | Spoc_Devices.CL_DEVICE_TYPE_CPU -> 1
        | _ -> 256)
    | _ -> 256
  in
  let grid_size = (size + block_size - 1) / block_size in
  let block =
    {Spoc.Kernel.blockX = block_size; Spoc.Kernel.blockY = 1; Spoc.Kernel.blockZ = 1}
  in
  let grid = {Spoc.Kernel.gridX = grid_size; Spoc.Kernel.gridY = 1; Spoc.Kernel.gridZ = 1} in

  Sarek.Kirc.run vector_add (a, b, c, size) (block, grid) 0 dev ;
  Spoc_Devices.flush dev () ;

  (* Read back results *)
  Spoc_Mem.to_cpu c () ;
  Spoc_Devices.flush dev () ;

  let result = Array.make size 0.0 in
  for i = 0 to size - 1 do
    result.(i) <- Spoc_Mem.get c i
  done ;
  result

(* Run via V2 runtime path and return results *)
let run_v2 (dev : V2_Device.t) : float array =
  (* Get the IR from the kernel *)
  let _, kirc = vector_add in
  let ir =
    match kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  (* Create V2 vectors *)
  let a = V2_Vector.create V2_Vector.float32 size in
  let b = V2_Vector.create V2_Vector.float32 size in
  let c = V2_Vector.create V2_Vector.float32 size in

  (* Initialize *)
  for i = 0 to size - 1 do
    V2_Vector.set a i (float_of_int i) ;
    V2_Vector.set b i (float_of_int (i * 2)) ;
    V2_Vector.set c i 0.0
  done ;

  (* Configure grid/block *)
  let block_size = 256 in
  let grid_size = (size + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

  (* Run via V2 *)
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Sarek.Execute.Vec a; Sarek.Execute.Vec b; Sarek.Execute.Vec c; Sarek.Execute.Int size]
    ~block ~grid
    () ;

  (* Flush to ensure kernel is complete, then auto-sync handles the rest *)
  V2_Transfer.flush dev ;

  (* to_array triggers auto-sync from GPU since vector is Stale_CPU *)
  V2_Vector.to_array c

(* Compare two float arrays *)
let compare_results (spoc_result : float array) (v2_result : float array) : int =
  let errors = ref 0 in
  for i = 0 to Array.length spoc_result - 1 do
    let diff = abs_float (spoc_result.(i) -. v2_result.(i)) in
    if diff > 0.001 then begin
      if !errors < 5 then
        Printf.printf "  Mismatch at %d: SPOC=%.2f, V2=%.2f\n" i spoc_result.(i) v2_result.(i) ;
      incr errors
    end
  done ;
  !errors

let () =
  print_endline "=== V2 Runtime Comparison Test ===" ;
  print_endline "Comparing SPOC and V2 execution paths\n" ;

  (* Initialize SPOC devices *)
  let spoc_devs = Spoc_Devices.init ~native:false () in
  if Array.length spoc_devs = 0 then begin
    print_endline "No GPU devices found" ;
    exit 1
  end ;

  (* Initialize V2 devices *)
  let v2_devs = V2_Device.init ~frameworks:["CUDA"; "OpenCL"] () in

  Printf.printf "Found %d SPOC device(s), %d V2 device(s)\n\n"
    (Array.length spoc_devs) (Array.length v2_devs) ;

  print_endline (String.make 80 '-') ;
  Printf.printf "%-40s %15s %15s %10s\n" "Device" "SPOC" "V2" "Status" ;
  print_endline (String.make 80 '-') ;

  let all_ok = ref true in

  (* Test each V2 device *)
  Array.iter (fun v2_dev ->
    let name = v2_dev.V2_Device.name in

    (* Find matching SPOC device *)
    let spoc_dev_opt =
      Array.find_opt (fun d ->
        d.Spoc_Devices.general_info.Spoc_Devices.name = name
      ) spoc_devs
    in

    match spoc_dev_opt with
    | None ->
        Printf.printf "%-40s %15s %15s %10s\n" name "-" "found" "SKIP" ;
    | Some spoc_dev ->
        (* Run both paths *)
        let spoc_result = run_spoc spoc_dev in
        let v2_result = run_v2 v2_dev in

        (* Compare *)
        let errors = compare_results spoc_result v2_result in
        let status = if errors = 0 then "MATCH" else "DIFFER" in
        if errors > 0 then all_ok := false ;

        Printf.printf "%-40s %15s %15s %10s\n"
          name "OK" "OK" status
  ) v2_devs ;

  print_endline (String.make 80 '-') ;

  if !all_ok then begin
    print_endline "\n=== All results MATCH ===" ;
    print_endline "V2 runtime produces identical results to SPOC"
  end else begin
    print_endline "\n=== Some results DIFFER ===" ;
    exit 1
  end
