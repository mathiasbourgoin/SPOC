---
layout : index
---

# Image Filter


``` ocaml
open Spoc

open Kirc

let (>>=) = Lwt.bind

let gpu_to_gray = kern v ->
  let open Std in
  let tid = thread_idx_x + block_dim_x * block_idx_x in
  if tid > (512*512) then
    ()
  else
    (
      let i = (tid*4) in
      let res = (v.[<i>] + v.[<i+1>] + v.[<i+2>]) / 3 in
      v.[<i>] <- res;
      v.[<i+1>] <- res;
      v.[<i+2>] <- res
    )

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
  

let compute devid devs data imageData c= 
  let dev = devs.(devid) in
  Printf.printf "Will use device : %s!"
  		(dev).Spoc.Devices.general_info.Spoc.Devices.name;
  let gpu_vect = Spoc.Vector.create Vector.int32 (512*512*4)
  in
  Random.self_init ();
  for i = 0 to Vector.length gpu_vect - 1 do
    gpu_vect.[<i>] <- Int32.of_int (pixel_get data i);
  done;
 
  let threadsPerBlock = match dev.Devices.specific_info with
    | Devices.OpenCLInfo clI -> 
      (match clI.Devices.device_type with
       | Devices.CL_DEVICE_TYPE_CPU -> 1
       | _  ->   256)
    | _  -> 256 in
  let blocksPerGrid =
    ((512*512) + threadsPerBlock -1) / threadsPerBlock
  in
  let block0 = {Spoc.Kernel.blockX = threadsPerBlock;
      Spoc.Kernel.blockY = 1; Spoc.Kernel.blockZ = 1}
  and grid0= {Spoc.Kernel.gridX = blocksPerGrid;
        Spoc.Kernel.gridY = 1; Spoc.Kernel.gridZ = 1} in
  ignore(Kirc.gen ~only:Devices.OpenCL
  		    gpu_to_gray);
  measure_time "" 
         (fun () -> Kirc.run gpu_to_gray (gpu_vect) (block0,grid0) 0 dev);
  
  for i = 0 to Vector.length gpu_vect - 1 do
    let t = Int32.to_int gpu_vect.[<i>] in 
    pixel_set data i t
  done;
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


  
  Dom.appendChild body select_devices;
  Dom.appendChild body (newLine ());


  let canvas = createCanvas document in
  canvas##width <- 512;
  canvas##height <- 512;
  
  let image : imageElement Js.t = createImg document in
  image##src <- Js.string "lena.png";

  let c = canvas##getContext (Dom_html._2d_) in
  image##onload <- 
    handler (fun _ -> 
         c##drawImage (image, 0., 0.); 
         Dom.appendChild body (newLine ());
	 Dom.appendChild body canvas;
	 let devs =
 	 Devices.init ~only:Devices.OpenCL () in	     
         let imageData = c##getImageData (0., 0., 512., 512.) in
         let data = imageData##data in
	 Dom.appendChild body (newLine ());
	 Array.iter
	 (fun (n) ->
	 let option = createOption document in
	 append_text option n.Devices.general_info.Devices.name;
	 Dom.appendChild select_devices  option)
	 devs;
	 
	 Dom.appendChild body (button (f select_devices devs data imageData c)
	 );Js._true);
  
  Js._false
    
    
let _ = 
  window##onload <- handler go
```