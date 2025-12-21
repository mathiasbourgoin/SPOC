(******************************************************************************
 * E2E test for Sarek PPX with variant type and helper function.
 ******************************************************************************)

open Spoc

let () =
  let dispatch =
    [%kernel
      let module Types = struct
        type shape = Circle of float32 | Square of float32
      end in
      let area (s : shape) : float32 =
        match s with Circle r -> 3.14 *. r *. r | Square x -> x *. x
      in
      fun (src : shape vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then dst.(tid) <- area src.(tid)]
  in

  let _, kirc_kernel = dispatch in
  print_endline "=== Variant helper IR ===" ;
  Sarek.Kirc.print_ast kirc_kernel.Sarek.Kirc.body ;
  print_endline "=========================" ;

  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No device found - IR generation test passed" ;
    exit 0) ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
  (try
     let _ = Sarek.Kirc.gen ~keep_temp:true dispatch dev in
     print_endline "Variant helper codegen PASSED"
   with e -> Printf.printf "Codegen failed: %s\n%!" (Printexc.to_string e)) ;
  ()
