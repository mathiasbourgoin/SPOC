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
open Unix

  kernel mandelbrot : Spoc.Vector.vint32 -> int -> int -> int -> int -> int -> unit = "kernels/Mandelbrot2" "mandelbrot"

let cpt = ref 0;;

let tot_time = ref 0.

let mesure_time f =
  let t0 = Unix.gettimeofday () in
  let a = f () in
	let t1 = Unix.gettimeofday () in
  Printf.printf "temps %d : %F\n%!" !cpt (t1 -. t0);
	tot_time := !tot_time +.  (t1 -. t0);
	incr cpt;
  a;;



let devices = mesure_time(Spoc.Devices.init)

let mesure_time f = f () ;;

let dev1 = ref devices.(0);;
let dev2 = ref devices.(1);;


let auto_transfers = ref true;;


let largeur = ref 800;;
let hauteur = ref 800;;

let max_iter  = ref 500;;


let b_iter = mesure_time(fun () -> Spoc.Vector.create Spoc.Vector.int32(*`Int32 Bigarray.int32*) ((!largeur)*(!hauteur)));;



let img = mesure_time(fun () -> Array.make_matrix !largeur !hauteur Graphics.black);;




let couleur n = 
  if n =  !max_iter then
    Graphics.black
  else let f n = 
	 let i = float n in 
	 int_of_float (255. *. (0.5 +. 0.5 *. sin(i *. 0.1))) in
       Graphics.rgb (*(f (n + 32))*)0  (f(n + 16))  (f n)


let main () = 
  let arg0 = ("-device1" , Arg.Int (fun i  -> dev1 := devices.(i)), "number of the device [0]")
  and arg1 = ("-device2" , Arg.Int (fun i  -> dev2 := devices.(i)), "number of the device [0]")
  and arg2 = ("-width" , Arg.Int (fun i  -> largeur := i), "width of the image to compute [1024]")
  and arg3 = ("-height" , Arg.Int (fun i  -> hauteur := i), "height of the image to compute [1024]")
  and arg4 = ("-max_iter" , Arg.Int (fun b -> max_iter := b), "max number of iterations [1024]") in
  Arg.parse ([arg0; arg1;arg2;arg3;arg4]) (fun s -> ()) "";
  Printf.printf "Will use device : %s and %s\n%!" (!dev1).Spoc.Devices.general_info.Spoc.Devices.name (!dev2).Spoc.Devices.general_info.Spoc.Devices.name;
  
  let threadsPerBlock dev = match dev.Devices.specific_info with
          | Devices.OpenCLInfo clI -> 
            (match clI.Devices.device_type with
              | Devices.CL_DEVICE_TYPE_CPU -> 1
              | _  ->   256)
          | _  -> 256 in
  let blocksPerGrid dev = (!largeur * !hauteur + (threadsPerBlock dev) -1) / (threadsPerBlock dev) in
  let block dev = {Spoc.Kernel.blockX = threadsPerBlock dev ; Spoc.Kernel.blockY = 1; Spoc.Kernel.blockZ = 1}
  and grid dev = {Spoc.Kernel.gridX = (blocksPerGrid dev)/2; Spoc.Kernel.gridY = 1; Spoc.Kernel.gridZ = 1} in
  let b_iter1 = mesure_time(fun () -> Spoc.Mem.sub_vector b_iter 0 (~ok_rng:(!largeur* !hauteur/2)) (~ko_rng:(!largeur* !hauteur/2)) (!largeur* !hauteur/2))
  and b_iter2 = mesure_time(fun () -> Spoc.Mem.sub_vector b_iter (!largeur* !hauteur/2) (~ok_rng:(!largeur* !hauteur/2)) (~ko_rng:(!largeur* !hauteur/2)) (!largeur* !hauteur/2)) in 
	mesure_time(fun () -> Spoc.Kernel.compile !dev1 mandelbrot);
  mesure_time(fun () -> Spoc.Kernel.compile !dev2 mandelbrot);
  mesure_time(fun () -> Spoc.Kernel.run !dev1 (block !dev1, grid !dev1) mandelbrot  (b_iter1, !max_iter, !largeur, (!hauteur), 0, (!hauteur/2)));



 mesure_time(fun () -> Spoc.Kernel.run !dev2 (block !dev2, grid !dev2) mandelbrot  (b_iter2, !max_iter, !largeur, (!hauteur), (!hauteur/2), (!hauteur)));
  
  let l = string_of_int !largeur in
  let h = string_of_int !hauteur in
  let dim = " " ^l^"x"^h in
  mesure_time(fun () -> Spoc.Mem.to_cpu b_iter1 ());	
  mesure_time(fun () -> Spoc.Mem.to_cpu b_iter2 ());
  mesure_time(fun () -> Spoc.Devices.flush !dev1 ()); 
  mesure_time(fun () -> Spoc.Devices.flush !dev2 ()); 

	for a = 0 to pred (!largeur - 1 ) do
	  for b = 0 to pred (!hauteur - 1 )do
	  img.(a).(b) <-  (couleur (Int32.to_int ((Spoc.Mem.unsafe_get b_iter (a * !hauteur + b) )))) ;
	  done;
	  done;
	  Graphics.open_graph dim;
	  Graphics.draw_image (Graphics.make_image img) 0 0;
 	ignore (Graphics.read_key ());;

let mesure_time f =
  let t0 = Unix.gettimeofday () in
  let a = f () in
	let t1 = Unix.gettimeofday () in
  Printf.printf "temps %d : %F\n%!" !cpt (t1 -. t0);
	tot_time := !tot_time +.  (t1 -. t0);
	incr cpt;
  a;;



let _ = 
	
		mesure_time main;
		Printf.printf "Total_time : %g\n%!" !tot_time;;



