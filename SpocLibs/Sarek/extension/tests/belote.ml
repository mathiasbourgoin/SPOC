open Spoc

ktype color = Spades | Hearts | Diamonds | Clubs ;;

ktype colval = {c:color; v : int32} ;;

ktype card = 
  Ace of color
  | King of color
  | Queen of color
  | Jack of color 
  | Other of colval;;

let  compute  = kern cards trump values n ->
  let value = fun a trump->
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
  if i < n then
   values.[<i>] <- value  cards.[<i>] trump.[<0>]
      

let n = 5_000_0000


let cpt = ref 0

let tot_time = ref 0.

let measure_time f s =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "%s : time %d : %Fs\n%!" s !cpt (t1 -. t0);
  tot_time := !tot_time +.  (t1 -. t0);
  incr cpt;
  a;;


let _ = 
  Random.self_init ();
  let devs = Spoc.Devices.init () in
  let dev = devs.(2) in
  let cards =  Vector.create (Vector.Custom customCard) n in
  let values =  Vector.create Vector.int32 n in
  let trump = Vector.create (Vector.Custom customColor) 1 in
(*  let cards_c = Array.create n (King Spades) 
  and values_c = Array.create n 0 in*)
  for i = 0 to n - 1 do
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
    Mem.set cards i c;
(*    cards_c.(i) <- c;*)
  done;
  Mem.set trump 0 Spades;
  ignore(Kirc.gen ~only:Devices.OpenCL compute);
  let threadsPerBlock = match dev.Devices.specific_info with
    | Devices.OpenCLInfo clI ->
      (match clI.Devices.device_type with
       | Devices.CL_DEVICE_TYPE_CPU -> 1
       | _ -> 256)
    | _ -> 256
  in
  let blocksPerGrid = (n + threadsPerBlock -1) / threadsPerBlock in
  let block = {Spoc.Kernel.blockX = threadsPerBlock; 
               Spoc.Kernel.blockY = 1;
               Spoc.Kernel.blockZ = 1;} in
  let grid = {Spoc.Kernel.gridX = blocksPerGrid; 
              Spoc.Kernel.gridY = 1;
              Spoc.Kernel.gridZ = 1;} in

  let name = dev.Spoc.Devices.general_info.Spoc.Devices.name in
  measure_time (fun () ->
      for i = 0 to 9 do
        Kirc.run compute (cards, trump, values, n) (block,grid) 0 dev;
        Mem.to_cpu values ();
        Devices.flush dev ();
      done) ("GPU "^name);
(*  let string_of_card c =
    let string_of_color c =
      match c with
      | Spades -> "Spades"
      | Hearts -> "Hearts"
      | Diamonds -> "Diamonds"
      | Clubs -> "Clubs"
    in
    match c with 
    | Ace c -> "Ace of "^string_of_color c
    | King c -> "King of " ^string_of_color c
    | Queen c -> "Queen of "^string_of_color c
    | Jack c -> "Jack of "^string_of_color c
    | Other {c;v} -> ("Other of "^string_of_color c^" of val "^Int32.to_string v)
  in 
  for i = 0 to 100 do
    Printf.printf "%s val : %ld\n" (string_of_card (Mem.get cards i)) (Mem.get values i)
  done
*)
(*
  let compute cards trump =
    let value = fun a ->
      match a with 
      | Ace c -> 11
      | King c -> 4
      | Queen c -> 3
      | Jack c -> if c = trump then 20 else 2
      | Other cv ->
        if cv.v = 10l then 10 else if (cv.c = trump) && (cv.v = 9l) then 14 else 0 
    in
    for i = 0 to n -1 do
      values_c.(i) <-  value cards.(i)
    done;
  in
  measure_time (fun () -> 
      for i = 0 to 9 do
        ignore(compute cards_c Spades);
      done) "CPU";
          
*)
