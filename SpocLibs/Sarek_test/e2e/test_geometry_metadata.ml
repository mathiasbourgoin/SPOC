open Spoc

let () =
  (* Use only externally registered Sarek metadata (point type + distance). *)
  let geom_kernel =
    [%kernel
      fun (a : Geometry_lib.point vector)
          (b : Geometry_lib.point vector)
          (out : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then out.(tid) <- Geometry_lib.distance a.(tid) b.(tid)]
  in

  let _, kirc_kernel = geom_kernel in
  print_endline "=== Geometry kernel IR ===" ;
  Sarek.Kirc.print_ast kirc_kernel.Sarek.Kirc.body ;
  print_endline "==========================" ;

  (* Optional code generation to ensure backend sees the external metadata. *)
  let devs = Devices.init () in
  if Array.length devs > 0 then begin
    let dev = devs.(0) in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    (try
       let _ = Sarek.Kirc.gen ~keep_temp:true geom_kernel dev in
       print_endline "Geometry kernel codegen: OK"
     with e ->
       Printf.printf
         "Geometry kernel codegen failed: %s\n%!"
         (Printexc.to_string e)) ;
    ()
  end
  else print_endline "No device found, IR generation only"
