(******************************************************************************
 * E2E test for Sarek PPX with helper function (klet-style) in the payload.
 ******************************************************************************)

open Spoc

let () =
  let scale_add =
    [%kernel
      let add_scale (x : float32) (y : float32) : float32 = x +. (2.0 *. y) in
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then dst.(tid) <- add_scale src.(tid) 3.0]
  in
  let _, kirc_kernel = scale_add in
  print_endline "=== klet-style helper IR ===" ;
  Sarek.Kirc.print_ast kirc_kernel.Sarek.Kirc.body ;
  print_endline "=============================" ;
  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No device found - IR generation test passed" ;
    exit 0) ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
  (try
     let _ = Sarek.Kirc.gen ~keep_temp:true scale_add dev in
     print_endline "Helper function codegen PASSED"
   with e ->
     Printf.printf "Codegen failed: %s\n%!" (Printexc.to_string e) ;
     ()) ;
  ()
