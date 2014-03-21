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

open Kirc


let width = ref (Int32.of_int 1000);;
let height = ref (Int32.of_int 1000);;

let max_iter  = ref (Int32.of_int 50);;

let zoom = ref 1. 
let shiftx = ref Int32.zero
let shifty = ref Int32.zero ;;

let recompile = ref false

let simple = ref true
  
let mandelbrot_recompile = kern img ->
  let open Std in
  let y = thread_idx_y + (block_idx_y * block_dim_y) in
  let x = thread_idx_x + (block_idx_x * block_dim_x) in
  (if (y >= !height) || (x >= !width) then
      return () ;
  );
(*  let shiftx = 0 in
  let shifty = 0 in*)
  let x0 = (x + !shiftx) in
  let y0 = (y + !shifty) in
  let mutable cpt = 0 in 
  let mutable x1 = 0. in
  let mutable y1 = 0. in
  let mutable x2 = 0. in
  let mutable y2 = 0. in
  let a = 4. 
	  *. 
	    ((float32 x0) /. (float32 !width)) /. !zoom  -. 2. in
  let b = 4. *. ((float32 y0) /. (float32 !height)) /. !zoom -. 2. in
 
  let mutable norm = x1 *. x1 +. y1 *. y1
  in
  while ((cpt < !max_iter) && (norm <=. 4.)) do
    cpt := (cpt + 1);
    x2 := (x1 *. x1) -. (y1 *. y1) +. a;
    y2 :=  (2. *. x1 *. y1 ) +. b;
    x1 := x2;
    y1 := y2;
    norm := (x1 *. x1 ) +. ( y1 *. y1);
  done;
  img.[<y * !width + x>] <- cpt
 ;; 


let mandelbrot = kern img shiftx shifty zoom -> 
  let open Std in
  let y = thread_idx_y + (block_idx_y * block_dim_y) in
  let x = thread_idx_x + (block_idx_x * block_dim_x) in
  (if (y >= !height) || (x >= !width) then
      return () ;
  );
  let x0 = (x + shiftx) in
  let y0 = (y + shifty) in
  let mutable cpt = 0 in 
  let mutable x1 = 0. in
  let mutable y1 = 0. in
  let mutable x2 = 0. in
  let mutable y2 = 0. in
  let a = 4. *. ((float32 x0) /. (float32 !width)) /. zoom  -. 2. in
  let b = 4. *. ((float32 y0) /. (float32 !height)) /. zoom -. 2. in
  
  let mutable norm = x1 *. x1 +. y1 *. y1
  in
  while ((cpt < !max_iter) && (norm <=. 4.)) do
    cpt := (cpt + 1);
    x2 := (x1 *. x1) -. (y1 *. y1) +. a;
    y2 :=  (2. *. x1 *. y1 ) +. b;
    x1 := x2;
    y1 := y2;
    norm := (x1 *. x1 ) +. ( y1 *. y1);
  done;
  img.[<y * !width + x>] <- cpt
;;


let mandelbrot_double = kern img shiftx shifty zoom ->
  let open Std in
	let open Math.Float64 in
  let y = thread_idx_y + (block_idx_y * block_dim_y) in
  let x = thread_idx_x + (block_idx_x * block_dim_x) in
  (if (y >= !height) || (x >= !width) then
      return () ;
  );
  let x0 = (x + shiftx) in
  let y0 = (y + shifty) in
  let mutable cpt = 0   in
  let mutable x1 = zero in
  let mutable y1 = zero in
  let mutable x2 = zero in
  let mutable y2 = zero in
  let a = minus (div (mul (of_float 4.)
			(div (float64 x0) (float64 !width)))
		   (of_float zoom)) (of_float 2.) in
  let b = minus (div (mul (of_float 4.)
			(div (float64 y0) (float64 !height)))
		   (of_float zoom)) (of_float 2.) in
  let mutable norm = add (mul x1 x1) (mul y1 y1)
  in
  while ((cpt < !max_iter) && ((to_float norm) <=. 4.)) do
  cpt := (cpt + 1);
  x2 := add (minus (mul x1  x1) (mul y1 y1))  a;
  y2 := add (mul (of_float 2.) (mul x1  y1 )) b;
  x1 := x2;
  y1 := y2;
  norm := add (mul x1  x1 ) (mul  y1  y1);
  done;
  img.[<y * !height + x>] <- cpt
;;


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

let devices = measure_time(Spoc.Devices.init);;

(*let measure_time f = f () ;;*)

let dev = ref devices.(0);;


let auto_transfers = ref true;;



let bench = ref (false,0) ;;


let get_min ar =
  Spoc.Mem.to_cpu ar ();
  let min = ref !max_iter in
  for i = 0 to (Spoc.Vector.length ar - 1) do
    if (Spoc.Mem.unsafe_get ar i) > Int32.one then
      if  (Spoc.Mem.unsafe_get ar i) < !min then
	min :=  (Spoc.Mem.unsafe_get ar i)
  done;
  !min
    
let couleur n = 
  if n =  Int32.to_int !max_iter then
    Graphics.rgb 196 200 200
  else let f n = 
	 let i = float n in 
	 int_of_float (255. *. (0.5 +. 0.5 *. sin(i *. 0.1))) in
       Graphics.rgb (f (n + 32))  (f(n + 16))  (f n)
	 
	 
let main_mandelbrot () = 
  let arg1 = ("-device" , Arg.Int (fun i  -> dev := devices.(i)),
	      "number of the device [0]")
  and arg2 = ("-height" , Arg.Int (fun i  -> height := Int32.of_int i),
	      "height of the image to compute [1000]")
  and arg3 = ("-width" , Arg.Int (fun i  -> width := Int32.of_int i),
	      "width of the image to compute [1000]")
  and arg4 = ("-max_iter" , Arg.Int (fun b -> max_iter := Int32.of_int b),
	      "max number of iterations [50]") 
  and arg5 = ("-recompile" , Arg.Bool (fun b -> recompile := b),
	      "Regenerates kernel at each redraw [false]") 
  and arg6 = ("-bench" , Arg.Int (fun b -> bench := (true,b)),
	      "benchmark (not interactive), number of calculations") 
  and arg7 = ("-double" , Arg.Bool (fun b -> simple := not b),
	      "use double precision [false]") 
in
  Arg.parse ([arg1;arg2;arg3;arg4;arg5;arg6; arg7]) (fun s -> ()) "";
  Printf.printf "Will use device : %s\n%!"
    (!dev).Spoc.Devices.general_info.Spoc.Devices.name;
  let threadsPerBlock = match !dev.Devices.specific_info with
    | Devices.OpenCLInfo clI -> 
      (match clI.Devices.device_type with
      | Devices.CL_DEVICE_TYPE_CPU -> 1
      | _  ->   256)
    | _  -> 256 in

  let b_iter = Spoc.Vector.create Spoc.Vector.int32 (Int32.to_int (Int32.mul !width  !height))
  in
  let sub_b = Spoc.Mem.sub_vector b_iter 0 (Spoc.Vector.length b_iter)
  in
  let img = Array.make_matrix (Int32.to_int !height) (Int32.to_int !width) Graphics.black in
  
  let blocksPerGrid =
    ((Int32.to_int (Int32.mul !width  !height)) + threadsPerBlock -1) / threadsPerBlock
  in
  let block0 = {Spoc.Kernel.blockX = threadsPerBlock;
		Spoc.Kernel.blockY = 1; Spoc.Kernel.blockZ = 1}
  and grid0= {Spoc.Kernel.gridX = blocksPerGrid;
	      Spoc.Kernel.gridY = 1; Spoc.Kernel.gridZ = 1} in

  let threadsPerBlock = match !dev.Devices.specific_info with
    | Devices.OpenCLInfo clI -> 
      (match clI.Devices.device_type with
      | Devices.CL_DEVICE_TYPE_CPU -> 1
      | _  ->   16)
    | _  -> 16  in
  let blocksPerGridx =
    ((Int32.to_int !width) + (threadsPerBlock) -1) / (threadsPerBlock) in
  let blocksPerGridy =
    ((Int32.to_int !height) + (threadsPerBlock) -1) / (threadsPerBlock) in
  let block = {Spoc.Kernel.blockX = threadsPerBlock;
	       Spoc.Kernel.blockY = threadsPerBlock;
	       Spoc.Kernel.blockZ = 1}
  and grid= {Spoc.Kernel.gridX = blocksPerGridx;
	     Spoc.Kernel.gridY = blocksPerGridy;
	     Spoc.Kernel.gridZ = 1} in
    
  if not !recompile then
    if not !simple then
      ignore(Kirc.gen mandelbrot_double)
    else 
      ignore(Kirc.gen mandelbrot);

  let l = string_of_int (Int32.to_int !width) in
  let h = string_of_int (Int32.to_int !height) in
  let dim = " " ^l^"x"^h in
  
  Graphics.open_graph dim;
  
  let running = ref true in
  
  Graphics.auto_synchronize false;
  while !running do
    
    running :=  not ( (fst !bench) && (snd !bench <= 0));
    
    if fst !bench then
      bench := (fst !bench, snd !bench -1);

    if !recompile then
      begin
	Kirc.gen mandelbrot_recompile;

	Kirc.run 
	  mandelbrot_recompile 
	  (sub_b) 
	  (block,grid) 
	  0 
	  !dev;
      end
    else
      begin
	if not !simple then
	  Kirc.run 
	    mandelbrot_double 
	    (sub_b, !shiftx, !shifty, !zoom) 
	    (block,grid) 
	    0 
	    !dev
	else
	  Kirc.run 
	    mandelbrot 
	    (sub_b, !shiftx, !shifty, !zoom) 
	    (block,grid) 
	    0 
	    !dev
      end;
    
    

    Spoc.Mem.to_cpu sub_b (); 
    Spoc.Devices.flush !dev (); 
    for a = 0 to pred (!height )do    
      for b = 0 to pred (!width ) do
	img.(a).(b) <-
	  (couleur (Int32.to_int
		      ((Spoc.Mem.get b_iter (a * !width + b) )))) ;
      done;
    done;
    Graphics.draw_image (Graphics.make_image img) 0 0;
    Graphics.synchronize ();
    
    Gc.full_major ();
    if not (fst !bench) then
      begin
	let key = Graphics.read_key () 
	in
	match key  with
	| 'w' -> shifty := !shifty + 10;
	| 's' -> shifty := !shifty - 10;
	| 'a' -> shiftx := !shiftx + 10;
	| 'd' -> shiftx := !shiftx - 10;
	| '+' -> zoom := !zoom *. 1.1;
	| '!' -> simple := not !simple;
	  shifty := !shifty + ((int_of_float (!zoom *. 10.)) *
				  !height / 200);
	  shiftx := !shiftx + ((int_of_float (!zoom *. 10.)) *
				  !width / 200);
	| '-' -> zoom := !zoom *. 0.9;
	  shifty := !shifty - ((int_of_float (!zoom *. 10.)) *
				  !height / 200);
	  shiftx := !shiftx - ((int_of_float (!zoom *. 10.)) *
				  !width / 200);
	| 'r' -> recompile := not !recompile;
	|_ -> running := false;
      end
  done;
  Graphics.close_graph ();;



let measure_time f =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "time %d : %Fs\n%!" !cpt (t1 -. t0);
  tot_time := !tot_time +.  (t1 -. t0);
  incr cpt;
  a;;

let _ =
  Printf.printf "%s\n" (
    "Interactive commands\n"^
		"  Move                              : WQSD\n"^
		"  Zoom                              : +/-\n"^
		"  Change precision (simple/double)  : !\n"^
		"  Recompile the kernel at each draw : r\n"^
		"  Quit                              : Any other key"
		);

  measure_time(main_mandelbrot);
  Printf.printf "Total_time : %g\n%!" !tot_time;;





