(*
         DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
                    Version 2, December 2004 

 Copyright (C) 2004 Sam Hocevar <sam@hocevar.net> 

 Everyone is permitted to copy and distribute verbatim or modified 
 copies of this license document, and changing it is allowed as long 
 as the name is changed. 

            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION 

  0. You just DO WHAT THE FUCK YOU WANT TO.
*)
open Spoc

kernel game : Spoc.Vector.vint32 -> int -> int -> int -> unit = "kernels/Game" "game"

let cpt = ref 0

let tot_time = ref 0.

let measure_time f =
  let t0 = Unix.gettimeofday () in
  let a = f () in
	let t1 = Unix.gettimeofday () in
  Printf.printf "time %d : %Fs\n%!" !cpt (t1 -. t0);
	tot_time := !tot_time +.  (t1 -. t0);
	incr cpt;
  a;;

let measure_time f = f ()

let devices = measure_time(Spoc.Devices.init);;

let measure_time f = f () ;;

let dev1 = ref devices.(0);;

let auto_transfers = ref true;;

let max_gen = ref 3;;

let largeur = ref 1024;;
let hauteur = ref 1024;;




let get_min ar =
	Spoc.Mem.to_cpu ar ();
	let min = ref !max_gen in
	for i = 0 to (Spoc.Vector.length ar - 1) do
		if (Int32.to_int (Spoc.Mem.unsafe_get ar i)) > 1 then
		if (Int32.to_int (Spoc.Mem.unsafe_get ar i)) < !min then
			min :=  (Int32.to_int (Spoc.Mem.unsafe_get ar i))
	done;
	!min

let couleur n =  
		if n = 1 then
			Graphics.red
		else 
			Graphics.blue



let main () = 
	Random.self_init ();
	let arg0 = ("-device" , Arg.Int (fun i  -> dev1 := devices.(i)), "number of the device [0]")
	and arg1 = ("-width" , Arg.Int (fun i  -> largeur := i), "width of the image to compute [1024]")
	and arg2 = ("-height" , Arg.Int (fun i  -> hauteur := i), "height of the image to compute [1024]")
	and arg3 = ("-gens" , Arg.Int (fun b -> max_gen := b), "max number of generations [1024]") in
	Arg.parse ([arg0; arg1;arg2;arg3]) (fun s -> ()) "";
	let b_iter1 = Spoc.Vector.create Spoc.Vector.int32(*`Int32 Bigarray.int32*) ((!largeur)*(!hauteur))
	and img1 = Array.make_matrix !largeur !hauteur Graphics.black
 in
	Printf.printf "Will use device : %s\n" (!dev1).Spoc.Devices.general_info.Spoc.Devices.name;
  Pervasives.flush stdout;
	let threadsPerBlock = match (!dev1).Devices.specific_info with
          | Devices.OpenCLInfo clI -> 
            (match clI.Devices.device_type with
              | Devices.CL_DEVICE_TYPE_CPU -> 1
              | _  ->   256)
          | _  -> 256 in
	let blocksPerGrid = (!largeur* !hauteur + threadsPerBlock -1) / threadsPerBlock in
	let block = {Spoc.Kernel.blockX = threadsPerBlock; Spoc.Kernel.blockY = 1; Spoc.Kernel.blockZ = 1}
	and grid= {Spoc.Kernel.gridX = blocksPerGrid; Spoc.Kernel.gridY = 1; Spoc.Kernel.gridZ = 1} in
	  for i = 0 to Spoc.Vector.length b_iter1 - 1 do
	    (if (Random.int 2) = 1 then
	       b_iter1.[<i>] <- (Random.int32 (Int32.of_int 2))
	     else
	       b_iter1.[<i>] <- (Int32.of_int 0));
	  done;


	  game#compile ~debug:true !dev1;
	  
	  
	  let l = string_of_int !largeur in
	  let h = string_of_int !hauteur in
	  let dim = " " ^l^"x"^h in
	    
	  let draw tab img dev queue= 
	    Spoc.Tools.iteri (fun elt i -> 
				let b = i / !largeur in
				let a = i - (b * !largeur) in
				  img.(b).(a) <-  (couleur (Int32.to_int elt))) tab ;  
  	    Graphics.draw_image (Graphics.make_image img) 0 0;		      
	    Graphics.synchronize();
	  in

	    let compute tab dev queue = 
	      game#run (tab, !max_gen, !largeur, !hauteur) (block, grid) queue  dev;	    
	      game#run (tab, !max_gen, !largeur, !hauteur) (block, grid) queue dev;	    
	    in
	      
	      Spoc.Kernel.run !dev1 (block,grid) game (b_iter1, !max_gen, !largeur, !hauteur);	    
	      Graphics.open_graph dim;
	      
	      for i = 1 to 10_000 do
		
		draw b_iter1 img1 !dev1 1;
		compute b_iter1 !dev1 0;

	  done;
    
	  ignore (Graphics.read_key ());
	  Graphics.close_graph ();;


let measure_time f =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
    Printf.printf "time %d : %Fs\n%!" !cpt (t1 -. t0);
    tot_time := !tot_time +.  (t1 -. t0);
    incr cpt;
    a;;

let measure_time f =
  f()

let _ =
  measure_time(main);
  Printf.printf "Total_time : %g\n%!" !tot_time;;

  
