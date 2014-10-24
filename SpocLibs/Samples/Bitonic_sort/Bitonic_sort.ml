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

ktype color = Spades | Hearts | Diamonds | Clubs 

ktype colval = {c:color; v : int32} 

ktype card = 
  Ace of color
  | King of color
  | Queen of color
  | Jack of color 
  | Other of colval

      
let gpu_bitonic = kern v j k trump ->
  let value = fun a trump ->
    match a with 
    | Ace c -> 11
    | King c -> 4
    | Queen c -> 3
    | Jack c -> if c = trump then 20 else 2
    | Other cv ->
      if cv.v = 10 then 10 else if (cv.c = trump) && (cv.v = 9) then 14 else 0 
  in
  let open Std in
  let i = thread_idx_x + block_dim_x * block_idx_x in
  let ixj = Math.xor i j in
  let mutable temp = v.[<0>] in
  if ixj < i then
    () else
    begin
      if (Math.logical_and i k) = 0  then
        (
          if  (value v.[<i>] trump.[<0>]) >
               (value v.[<ixj>] trump.[<0>]) then
            (temp := v.[<ixj>];
             v.[<ixj>] <- v.[<i>];
             v.[<i>] <- temp)
        )
      else 
      if (value v.[<i>] trump.[<0>]) < (value v.[<ixj>] trump.[<0>]) then
        (temp := v.[<ixj>];
         v.[<ixj>] <- v.[<i>];
         v.[<i>] <- temp);
    end
    (*  else
        v.[<i>] <- 0.
*)
  

;;



let exchange (v : (float, Bigarray.float32_elt) Spoc.Vector.vector) i j : unit =
  let t : float = v.[<i>] in
  v.[<i>] <- v.[<j>];
  v.[<j>] <- t
;;

let rec sortup v 
    m n : unit = 
  if n <> 1 then
    begin
      sortup v m (n/2);
      sortdown v (m+n/2) (n/2);
      mergeup v m (n/2); 
    end

and sortdown v 
    m n : unit =
  if n <> 1 then
    begin
      sortup v m (n/2);
      sortdown v (m+n/2) (n/2);
      mergedown v m (n/2);
    end

and mergeup v 
    (m:int) (n:int) : unit =
  if n <> 0 then
    begin
      for i = 0 to n - 1 do
        if v.[<m+i>] > v.[<m+i+n>] then
          exchange v  (m+i) (m+i+n);
      done;
      mergeup v  m (n/2);
      mergeup v  (m+n) (n/2)
    end

and mergedown v 
    m n =
  if n <> 0 then
    begin
      for i = 0 to n - 1 do
        if v.[<m+i>] < v.[<m+i+n>] then
          exchange v (m+i) (m+i+n);
      done;
      mergedown v m (n/2);
      mergedown v (m+n) (n/2)
    end
;;

  

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
  


let () = 
  let devid = ref 1 
  and size = ref (1024*2*2*2*2*2*2*2*2*2*2*2*2*2*2*2)
  and check = ref true
  and compare = ref true
  in

  let arg1 = ("-device" , Arg.Int (fun i  -> devid := i),
	      "number of the device [0]")
  and arg2 = ("-size" , Arg.Int (fun i  -> size := i),
	      "size of the vector to sort [1024]")
  and arg3 = ("-bench" , Arg.Bool (fun b  -> compare := b),
	      "compare time with sequential computation [true]")
  and arg4 = ("-check" , Arg.Bool (fun b  -> check := b),
	      "verify results [true]")
  in
  Arg.parse ([arg1;arg2; arg3; arg4]) (fun s -> ()) "";
  let devs = Spoc.Devices.init () in
  let dev = ref devs.(!devid) in
  Printf.printf "Will use device : %s  to sort %d floats\n%!"
    (!dev).Spoc.Devices.general_info.Spoc.Devices.name !size;
  let size = !size 
  and check = !check 
  and compare = !compare 
  in    
  let cards_c  = Array.create size (King Hearts) in
(*  let cards = Spoc.Vector.create (Vector.Custom customCard)  size
  and trump = Spoc.Vector.create (Vector.Custom customColor)  1
  in*)
  Random.self_init ();
  (* fill vectors with randmo values... *)
  for i = 0 to Array.length cards_c - 1 do
    let v = Random.float 255. in
        let c =  (let j = Random.int 12 + 1  in
                     let c = 
                       match Random.int 3 with
                       | 0 -> Spades
                       | 1 -> Hearts 
                       | 2 -> Diamonds 
                       | 3 -> Clubs
                       | _ -> assert false in
                     match j with
                     | 11 -> Jack c
                     | 12 -> Queen c
                     | 13 ->  King c 
                     | 1 -> Ace c 
                     | a -> Other {c = c; v = Int32.of_int a}) in
(*    Mem.set cards i c;*)
    cards_c.(i) <- c;
  done;
(*  Mem.set trump 0 Spades;*)

  if compare then
    begin
(*      
      for i = 0 to Vector.length cards - 1 do

      done;*)
      let value = fun a ->
        let trump = Spades in
        match a with 
        | Ace c -> 11
        | King c -> 4
        | Queen c -> 3
        | Jack c -> if c = trump then 20 else 2
        | Other cv ->
          if cv.v = 10l then 10 else if (cv.c = trump) && (cv.v = 9l) then 14 else 0 
      in
      measure_time "Sequential Array.sort" 
        (fun () -> Array.sort (fun a b -> Pervasives.compare (value a) (value b)) cards_c);
    end;
(*  let threadsPerBlock = match !dev.Devices.specific_info with
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
          Kirc.run gpu_bitonic (cards,!j,!k, trump) (block0,grid0) 0 !dev;
          j := !j lsr 1;
        done;
        k := !k lsl 1 ;
      done;
    );
*)
;;
