---
layout : index_sample
---

# Mandelbrot


``` ocaml
open Spoc

open Kirc

let width = ref 1024;;
let height = ref 1024;;

let max_iter  = ref 50;;

let mandelbrot = kern img -> 
  let open Std in
  let y = thread_idx_y + (block_idx_y * block_dim_y) in
  let x = thread_idx_x + (block_idx_x * block_dim_x) in
  if (y < !height) && (x < !width) then
     begin  
       let x0 = x  in
       let y0 = y  in
       let mutable cpt = 0 in 
       let mutable x1 = 0. in
       let mutable y1 = 0. in
       let mutable x2 = 0. in
       let mutable y2 = 0. in
       let a = 4. *. ((float x0) /. (float !width))   -. 2. in
       let b = 4. *. ((float y0) /. (float !height)) -. 2. in
       
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
     end
;;


let append_text e s = Dom.appendChild e (document##createTextNode (Js.string s))

let button action = 
  let b = createInput ~_type:(Js.string "button")  document in
  b##value <- (Js.string "Go");
  b##onclick <- handler action;
  b
;;  

let measure_time s f =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "Time %s : %Fs\n%!" s (t1 -. t0);
  a;;

let color n = 
  if n =  !max_iter then
    (196, 200, 200)
  else 
    let f n = 
      let i = float n in 
      int_of_float (255. *. (0.5 +. 0.5 *. sin(i *. 0.1))) in
    ((f (n + 32)),  (f(n + 16)),  (f n))
      

let compute devid devs data imageData c= 
  let dev = devs.(devid) in
  Printf.printf "Will use device : %s!"
    (dev).Spoc.Devices.general_info.Spoc.Devices.name;
  let gpu_vect = Spoc.Vector.create Vector.int32 (!width * !height)
  in
  Random.self_init ();

  let threadsPerBlock = match dev.Devices.specific_info with
    | Devices.OpenCLInfo clI -> 
      (match clI.Devices.device_type with
      | Devices.CL_DEVICE_TYPE_CPU -> 1
      | _  ->   16)
    | _  -> 16  in
  let blocksPerGridx =
    (!width + (threadsPerBlock) -1) / (threadsPerBlock) in
  let blocksPerGridy =
    (!height + (threadsPerBlock) -1) / (threadsPerBlock) in
  let block = {Spoc.Kernel.blockX = threadsPerBlock;
         Spoc.Kernel.blockY = threadsPerBlock;
	        Spoc.Kernel.blockZ = 1}
  and grid= {Spoc.Kernel.gridX = blocksPerGridx;
       Spoc.Kernel.gridY = blocksPerGridy;
            Spoc.Kernel.gridZ = 1} in

  measure_time "" 
    (fun () -> Kirc.run mandelbrot (gpu_vect) (block,grid) 0 dev;
      for i = 0 to Vector.length gpu_vect - 1 do
        let t = Int32.to_int gpu_vect.[<i>] in 
        let r,g,b = (color t) in 
        pixel_set data (i*4) r; 
        pixel_set data (i*4+1) g; 
        pixel_set data (i*4+2) b; 
        pixel_set data (i*4+3) 255; 
      done;
    );
  c##putImageData (imageData, 0., 0.);
;;


let f select_devices devs data imageData c= 
  (fun _ ->
     let select = select_devices##selectedIndex + 0 in
     compute select devs data imageData c;
     Js._true)
;;

let newLine _ = Dom_html.createBr document


let nodeJsText t =
  let sp = Dom_html.createSpan document in
  Dom.appendChild sp (document##createTextNode (t)) ;
  sp

let nodeText t =
  nodeJsText (Js.string t)

open Spoc

let go _ =


  let body =
    Js.Opt.get (document##getElementById (Js.string "section1"))
      (fun () -> assert false) in

  Dom.appendChild body (newLine ());
  let select_devices = createSelect document in
  Dom.appendChild body (nodeText "Choose a computing device : ");
  select_devices##style##margin <- Js.string "10px";


  Dom.appendChild body select_devices;

  let a = createA document in
  Dom.appendChild body a;
  Dom.appendChild body (newLine ());


  let canvas = createCanvas document in
  canvas##width <- !width;
  canvas##height <- !height;

  let c = canvas##getContext (Dom_html._2d_) in
  Dom.appendChild body (newLine ());
  Dom.appendChild body canvas;
  let devs =
    Devices.init ~only:Devices.OpenCL () in	     
          ignore(Kirc.gen ~only:Devices.OpenCL
	    mandelbrot);
 
 let imageData = c##getImageData (0., 0., (float !width), (float !height)) in
 let data = imageData##data in

 Dom.appendChild body (newLine ());

        Array.iter
	  (fun (n) ->
	       let option = createOption document in
	            append_text option n.Devices.general_info.Devices.name;
		         Dom.appendChild select_devices  option)
			   devs;

			   Dom.appendChild a (button (f select_devices devs data 
                                        imageData c));

  Js._false


let _ = 
  window##onload <- handler go
```
