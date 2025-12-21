(******************************************************************************
 * E2E test for Sarek PPX with record type declarations inside the kernel.
 ******************************************************************************)

open Spoc

let () =
  let point_copy =
    [%kernel
      let module Types = struct
        type point = {x : float32; y : float32}
      end in
      fun (src : point vector) (dst : point vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then
          let p = src.(tid) in
          let next : point = {x = p.x +. 1.0; y = p.y} in
          dst.(tid) <- next]
  in

  let _, kirc_kernel = point_copy in
  print_endline "=== Generated Kernel IR (ktype record) ===" ;
  Sarek.Kirc.print_ast kirc_kernel.Sarek.Kirc.body ;
  print_endline "==========================================" ;

  let devices = Devices.init () in
  if Array.length devices = 0 then begin
    print_endline "No device found - IR generation test passed" ;
    exit 0
  end ;
  let dev = devices.(0) in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
  (try
     let _ = Sarek.Kirc.gen point_copy dev in
     print_endline "Code generation succeeded"
   with e ->
     Printf.printf "Code generation failed: %s\n%!" (Printexc.to_string e)) ;
  ()
