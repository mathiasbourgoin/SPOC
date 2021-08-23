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

[%%kernel external mandelbrot : Spoc.Vector.vint32 -> int -> int -> int -> unit = "kernels/Mandelbrot" "mandelbrot"]

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

let measure_time f = f () ;;

let dev = ref devices.(0);;


let auto_transfers = ref true;;


let largeur = ref 800;;
let hauteur = ref 800;;

let max_iter  = ref 500;;

let get_min ar =
  Spoc.Mem.to_cpu ar ();
  let min = ref !max_iter in
  for i = 0 to (Spoc.Vector.length ar - 1) do
    if (Int32.to_int (Spoc.Mem.unsafe_get ar i)) > 1 then
      if (Int32.to_int (Spoc.Mem.unsafe_get ar i)) < !min then
	min :=  (Int32.to_int (Spoc.Mem.unsafe_get ar i))
  done;
  !min

let couleur n =
  if n =  !max_iter then
    Graphics.black
  else let f n =
	 let i = float n in
	 int_of_float (255. *. (0.5 +. 0.5 *. sin(i *. 0.1))) in
    Graphics.rgb (*(f (n + 32))*)0  (f(n + 16))  (f n)


let main () =
  let arg1 = ("-device" , Arg.Int (fun i  -> dev := devices.(i)), "number of the device [0]")
  and arg2 = ("-width" , Arg.Int (fun i  -> largeur := i), "width of the image to compute [1024]")
  and arg3 = ("-height" , Arg.Int (fun i  -> hauteur := i), "height of the image to compute [1024]")
  and arg4 = ("-max_iter" , Arg.Int (fun b -> max_iter := b), "max number of iterations [1024]") in
  Arg.parse ([arg1;arg2;arg3;arg4]) (fun s -> ()) "";
  Printf.printf "Will use device : %s\n" (!dev).Spoc.Devices.general_info.Spoc.Devices.name;
  (*	let threadsPerBlock = match !dev.Devices.specific_info with
            | Devices.OpenCLInfo clI ->
              (match clI.Devices.device_type with
                | Devices.CL_DEVICE_TYPE_CPU -> 1
                | _  ->   256)
            | _  -> 256 in
    let blocksPerGrid = (!largeur* !hauteur + threadsPerBlock -1) / threadsPerBlock in
    let block = {Spoc.Kernel.blockX = threadsPerBlock; Spoc.Kernel.blockY = 1; Spoc.Kernel.blockZ = 1}
    and grid= {Spoc.Kernel.gridX = blocksPerGrid; Spoc.Kernel.gridY = 1; Spoc.Kernel.gridZ = 1} in
  *)
  let b_iter = Spoc.Vector.create Spoc.Vector.int32(*`Int32 Bigarray.int32*) ~dev:!dev ((!largeur)*(!hauteur)) in


  let img = Array.make_matrix !largeur !hauteur Graphics.black in

  let threadsPerBlock = match !dev.Devices.specific_info with
    | Devices.OpenCLInfo clI ->
      (match clI.Devices.device_type with
       | Devices.CL_DEVICE_TYPE_CPU -> 1
       | _  ->   8)
    | _  -> 8 in
  let blocksPerGridx = (!largeur + (threadsPerBlock) -1) / (threadsPerBlock) in
  let blocksPerGridy = (!hauteur + (threadsPerBlock) -1) / (threadsPerBlock) in
  let block = {Spoc.Kernel.blockX = threadsPerBlock; Spoc.Kernel.blockY = threadsPerBlock; Spoc.Kernel.blockZ = 1}
  and grid= {Spoc.Kernel.gridX = blocksPerGridx;   Spoc.Kernel.gridY = blocksPerGridy; Spoc.Kernel.gridZ = 1} in


  let sub_b = Spoc.Mem.sub_vector b_iter 0 (Spoc.Vector.length b_iter) in

  Spoc.Kernel.compile !dev mandelbrot;

  Graphics.auto_synchronize false;
  let l = string_of_int !largeur in
  let h = string_of_int !hauteur in
  let dim = " " ^l^"x"^h in
  Graphics.open_graph dim;

  for i = 0 to 9 do

    Spoc.Kernel.run !dev (block,grid) mandelbrot (sub_b, !max_iter, !largeur, !hauteur);

    Spoc.Mem.to_cpu sub_b ();
    Spoc.Devices.flush !dev ();


    for a = 0 to pred (!largeur - 1 ) do
      for b = 0 to pred (!hauteur - 1 )do
	img.(b).(a) <-  (couleur (Int32.to_int ((Spoc.Mem.unsafe_get b_iter (b * !hauteur + a) )))) ;
      done;
    done;
    Graphics.draw_image (Graphics.make_image img) 0 0;
    Graphics.synchronize ();
    Gc.full_major();
  done;;
Printf.printf "Press any key to close\n%!";

(*	ignore (Graphics.read_key ());*)
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
  measure_time(main);
  Printf.printf "Total_time : %g\n%!" !tot_time;;
