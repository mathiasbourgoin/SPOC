(******************************************************************************
 * E2E test for Sarek PPX with record type declarations at top level.
 *
 * The [@@sarek.type] attribute generates:
 *   - point_custom (for SPOC Vector)
 *   - point_custom_v2 (for V2 Sarek_core.Vector)
 ******************************************************************************)

open Spoc
module V2_Vector = Sarek_core.Vector
module V2_Device = Sarek_core.Device
module V2_Transfer = Sarek_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

type float32 = float

(* Define point type with sarek.type attribute - generates point_custom (SPOC)
   and point_custom_v2 (V2) *)
type point = {x : float32; y : float32} [@@sarek.type]

(* Top-level kernel - no type annotation needed if we just extract the kirc part *)
let point_copy_kirc =
  snd
    [%kernel
      fun (src : point vector) (dst : point vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then
          let p = src.(tid) in
          let next : point = {x = p.x +. 1.0; y = p.y} in
          dst.(tid) <- next]

let () =
  print_endline "=== Generated Kernel IR (ktype record) ===" ;
  Sarek.Kirc.print_ast point_copy_kirc.Sarek.Kirc.body ;
  print_endline "==========================================" ;

  let devices = Devices.init () in
  if Array.length devices = 0 then begin
    print_endline "No device found - IR generation test passed" ;
    exit 0
  end ;
  let dev = devices.(0) in
  Printf.printf "SPOC device: %s\n%!" dev.Devices.general_info.Devices.name ;
  print_string "SPOC: " ;
  (try
     (* For SPOC, need to use Kirc.gen + run, not just #run *)
     let kern =
       [%kernel
         fun (src : point vector) (dst : point vector) (n : int32) ->
           let tid = thread_idx_x in
           if tid < n then
             let p = src.(tid) in
             let next : point = {x = p.x +. 1.0; y = p.y} in
             dst.(tid) <- next]
     in
     let _ = Sarek.Kirc.gen kern dev in
     print_endline "Code generation succeeded"
   with e -> Printf.printf "Failed: %s\n%!" (Printexc.to_string e)) ;

  (* V2 execution *)
  print_endline "\n=== V2 Path ===" ;
  let v2_devs = V2_Device.init ~frameworks:["CUDA"; "OpenCL"] () in
  if Array.length v2_devs = 0 then
    print_endline "No V2 devices found - skipping"
  else begin
    let v2_dev = v2_devs.(0) in
    Printf.printf "V2 device: %s\n%!" v2_dev.V2_Device.name ;
    print_string "V2: " ;
    try
      let n = 64 in
      let src = V2_Vector.create_custom point_custom_v2 n in
      let dst = V2_Vector.create_custom point_custom_v2 n in
      for i = 0 to n - 1 do
        V2_Vector.set src i {x = float_of_int i; y = float_of_int (n - i)} ;
        V2_Vector.set dst i {x = 0.0; y = 0.0}
      done ;
      let threads = min 64 n in
      let grid_x = (n + threads - 1) / threads in
      let ir =
        match point_copy_kirc.Sarek.Kirc.body_v2 with
        | Some ir -> ir
        | None -> failwith "Kernel has no V2 IR"
      in
      Sarek.Execute.run_vectors
        ~device:v2_dev
        ~block:(Sarek.Execute.dims1d threads)
        ~grid:(Sarek.Execute.dims1d grid_x)
        ~ir
        ~args:
          [
            Sarek.Execute.Vec src;
            Sarek.Execute.Vec dst;
            Sarek.Execute.Int32 (Int32.of_int n);
          ]
        () ;
      V2_Transfer.flush v2_dev ;
      (* Verify: dst[i].x should be src[i].x + 1.0, dst[i].y should be src[i].y *)
      let ok = ref true in
      for i = 0 to n - 1 do
        let s = V2_Vector.get src i in
        let d = V2_Vector.get dst i in
        let expected_x = s.x +. 1.0 in
        let expected_y = s.y in
        if
          abs_float (d.x -. expected_x) > 1e-3
          || abs_float (d.y -. expected_y) > 1e-3
        then (
          ok := false ;
          if i < 5 then
            Printf.printf
              "  Mismatch at %d: got {%.1f, %.1f} expected {%.1f, %.1f}\n%!"
              i
              d.x
              d.y
              expected_x
              expected_y)
      done ;
      if !ok then print_endline "PASSED" else print_endline "FAILED"
    with e -> Printf.printf "FAIL (%s)\n%!" (Printexc.to_string e)
  end
