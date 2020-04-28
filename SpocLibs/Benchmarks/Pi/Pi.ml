open Spoc
open Kirc

       
let piKern = kern rX rY (inside : int32 vector)->
  let open Std in
  let open Math in
  let open Float32 in
  let i = thread_idx_x + block_idx_x * block_dim_x in
  let r = sqrt ( add (mul rX.[< i >] rX.[< i>])
                     (mul rY.[< i >] rY.[< i >])) in
  if (r <=. 1.) then
    ($$ fun dev ->
       match dev.Devices.specific_info with
       | Devices.OpenCLInfo clI ->
          "atomic_inc (inside)"
       | _ -> "atomicAdd (inside,1)";; $$ : unit )
    


let cpt = ref 0

let tot_time = ref 0.

let measure_time f s iter =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "%s : %Fs  average : %Fs \n%!" s
                (t1 -. t0) ((t1 -. t0)/. (float_of_int iter));
  tot_time := !tot_time +.  (t1 -. t0);
  incr cpt;
  a;;

let devices = Devices.init ()

let size = ref 20_000_000

let _ =

  let vX = Vector.create Vector.float32 !size
  and vY =  Vector.create Vector.float32 !size
  and inside = Vector.create Vector.int32 1 in

  Mem.set inside 0 0l;

  Array.iteri (fun i dev -> measure_time (fun () ->
      ignore(Kirc.gen piKern dev);
    ) (Printf.sprintf "Time to generate kernel for \"%s\""
         dev.Devices.general_info.Devices.name) 1) devices ;

  for i = 0 to !size - 1 do
    Mem.set vX i (Random.float 1.);
    Mem.set vY i (Random.float 1.);
  done;

  let make_bg = fun dev size ->
    let threadsPerBlock = match dev.Devices.specific_info with
      | Devices.OpenCLInfo clI ->
         (match clI.Devices.device_type with
          | Devices.CL_DEVICE_TYPE_CPU -> 1
          | _  ->   256)
      | _  -> 256 in
    let blocksPerGrid =
      (size + threadsPerBlock -1) / threadsPerBlock
    in

    let block0 = {Spoc.Kernel.blockX = threadsPerBlock;
		  Spoc.Kernel.blockY = 1; Spoc.Kernel.blockZ = 1}
    and grid0= {Spoc.Kernel.gridX = blocksPerGrid;
	        Spoc.Kernel.gridY = 1; Spoc.Kernel.gridZ = 1} in
    block0,grid0
  in


  measure_time (fun () ->
      Kirc.run piKern (vX, vY, inside) (make_bg devices.(1) !size) 0 devices.(1);
      Devices.flush devices.(1) ();
    ) (Printf.sprintf "piCL on %s" devices.(1).Devices.general_info.Devices.name) 1;

  let pi = (float (Int32.to_int (Mem.get inside 0) * 4)) /. (float !size) in
  Printf.printf "PI = %.10g\n" pi;

  Mem.set inside 0 0l;
  measure_time (fun () ->
      Kirc.run piKern (vX, vY, inside) (make_bg devices.(0) !size) 0 devices.(0);
      Devices.flush devices.(0) ();
    ) (Printf.sprintf "piCU on %s" devices.(0).Devices.general_info.Devices.name) 1;

  let pi = (float (Int32.to_int (Mem.get inside 0) * 4)) /. (float !size) in
  Printf.printf "PI = %.10g\n" pi;

  let vX1 = Mem.sub_vector vX 0 (97* !size/100) in
  let vY1 = Mem.sub_vector vY 0 (97* !size/100) in

  let vX2 = Mem.sub_vector vX (97* !size/100) (3* !size/100) in
  let vY2 = Mem.sub_vector vY (97* !size/100) (3* !size/100) in

  let inside2 = Vector.create Vector.int32 1 in

  Mem.set inside 0 0l;
  Mem.set inside2 0 0l;

  measure_time (fun () -> 
      Kirc.run piKern (vX1, vY1, inside) (make_bg devices.(0) (97* !size/100)) 0 devices.(0);
      Kirc.run piKern (vX2, vY2, inside2) (make_bg devices.(1) (3* !size/100)) 0 devices.(1);
      Devices.flush devices.(0) ();
      Devices.flush devices.(1) ();
    ) (Printf.sprintf "piCU+piCL on both devices") 1;

  let pi = (float (Int32.to_int (Int32.add (Mem.get inside 0) (Mem.get inside2 0))* 4)) /. (float !size) in
  Printf.printf "PI = %.10g\n" pi;
  
  ()
;;
