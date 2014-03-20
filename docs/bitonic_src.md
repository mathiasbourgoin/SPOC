---
layout : index_sample
---

# Bitonic Sort


``` ocaml
open Spoc

open Kirc



let gpu_bitonic = kern v j k ->
  let open Std in
  let i = thread_idx_x + block_dim_x * block_idx_x in
  let ixj = Math.xor i j in
  let mutable temp = 0. in
  if ixj < i then
    () else
    begin
      if (Math.logical_and i k) = 0  then
        (
          if  v.[<i>] >. v.[<ixj>] then
            (temp := v.[<ixj>];
             v.[<ixj>] <- v.[<i>];
             v.[<i>] <- temp)
        )
      else 
      if v.[<i>] <. v.[<ixj>] then
        (temp := v.[<ixj>];
         v.[<ixj>] <- v.[<i>];
         v.[<i>] <- temp);
    end

let append_text e s = Dom.appendChild e (document##createTextNode 
    		      (Js.string s))

let button action = 
  let b = createInput ~_type:(Js.string "button")  document in
  b##value <- (Js.string "Go");
  b##onclick <- handler action;
  b
;;  



let text  name default cols = 
  let b = createInput ~_type:(Js.string "text")  document in
  b##value <- (Js.string default);
  b##size <-  4;
  b

let cpt = ref 0

let tot_time = ref 0.

let measure_time s f =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "time %s : %Fs\n%!" s (t1 -. t0);
  tot_time := !tot_time +.  (t1 -. t0);
  incr cpt;
  a;;
  

let compute size devid devs = 
  let dev = devs.(devid) in
  Printf.printf "Will use device : %s  to sort %d floats\n%!"
    (dev).Spoc.Devices.general_info.Spoc.Devices.name size;
  
  let gpu_vect = Spoc.Vector.create Vector.float32 size
  and base_vect = Spoc.Vector.create Vector.float32 size
  and vect_as_array = Array.create size 0.
  in
  Random.self_init ();
  (* fill vectors with randmo values... *)
  for i = 0 to Vector.length gpu_vect - 1 do
    let v = Random.float 255. in
    gpu_vect.[<i>] <- v;
    base_vect.[<i>] <- v;
    vect_as_array.(i) <- v;
  done;
  

  begin
    measure_time "Sequential Array.sort" 
      (fun () -> Array.sort Pervasives.compare vect_as_array);
  end;
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
  ignore(Kirc.gen ~only:Devices.OpenCL
           gpu_bitonic);
  let j,k = ref 0,ref 2 in
  measure_time "Parallel Bitonic" (fun () ->
      while !k <= size do
        j := !k lsr 1;
        while !j > 0 do
          Kirc.run gpu_bitonic (gpu_vect,!j,!k) (block0,grid0) 
	  	   	       	0 dev;
          j := !j lsr 1;
        done;
        k := !k lsl 1 ;
      done;
    );

  (
    let r = ref (-. infinity) in
    for i = 0 to size - 1 do
      if gpu_vect.[<i>] < !r then
        failwith (Printf.sprintf "error, %g <  %g" 
		  gpu_vect.[<i>] !r)
      else
        r := gpu_vect.[<i>]; 
    done;
    Printf.printf "Check OK\n";
  )
;;


let pow2 n = 
  let rec aux acc n = 
    if n = 1 then 
      acc
    else
      aux (acc*2) (n-1)
  in
  aux 2 n

let f size_text select_devices devs = 
  (fun _ ->
     let size = 
       pow2 (int_of_string (Js.to_string size_text##value))
     in
     let select = select_devices##selectedIndex + 0 in
     compute size select devs;
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
  let devs = Devices.init ~only:Devices.OpenCL () in

  let body =
    Js.Opt.get (document##getElementById (Js.string "BitonicSort"))
      (fun () -> assert false) in

  Dom.appendChild body (nodeText 
  	"This sample computes a bitonic sort over a"^
	" vector of float");
  Dom.appendChild body (newLine ());
  Dom.appendChild body (newLine ());
  let select_devices = createSelect document in
  Dom.appendChild body (nodeText "Choose a computing device : ");

  Array.iter
    (fun (n) ->
       let option = createOption document in
       append_text option n.Devices.general_info.Devices.name;
       Dom.appendChild select_devices  option)
    devs;
  
  Dom.appendChild body select_devices;
  Dom.appendChild body (newLine ());

  let size_text = (text "size" "10" 4) in
  Dom.appendChild body (nodeText "Vector size :  2^"); 
  Dom.appendChild body size_text;


  Dom.appendChild body (newLine ());
  Dom.appendChild body (button (f size_text select_devices devs));


  Js._false

    
let _ = 
  window##onload <- handler go
```
