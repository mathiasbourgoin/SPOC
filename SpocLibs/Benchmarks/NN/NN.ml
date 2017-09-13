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
let cpt = ref 0

let tot_time = ref 0.

let measure_time f  s =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "%s time %d : %Fs\n%!" s !cpt (t1 -. t0);
  tot_time := !tot_time +.  (t1 -. t0);
  incr cpt;
  a;;


let devices = measure_time (Spoc.Devices.init ) "init"

ktype latLong = {
                lat : float32;
                lng : float32;
              }

let euclid = kern locations distances numRecords lat lng ->
  let open Std in
  let open Math.Float32 in
  let globalId = block_dim_x * (grid_dim_x * block_idx_y+block_idx_x) + thread_idx_x in
  if (globalId < numRecords) then
    let latLong = locations.[<globalId>] in
    distances.[<globalId>] <-  sqrt (add (mul  (minus lat latLong.lat) (minus lat latLong.lat))
                                         (mul (minus lng  latLong.lng) (minus lng  latLong.lng)))
                                    
                                    
type record = {
    recString : string;
  mutable distance : float;
}


let loadData fname records locations =
  let flist = open_in fname in
  let dbname = ref "" in
  let recNum = ref 0 in
  (try
    while true do
      dbname :=
        Pervasives.input_line flist;
         
      if !dbname = "" then
        raise End_of_file;
      let fp =
        try (open_in !dbname)
        with | _ -> failwith ("Error opening a db: " ^ !dbname)
      in
      (try
         
         while true do
           let s = input_line fp in

           let lat = float_of_string (String.sub s 28 4)
           and lng = float_of_string (String.sub s 33 4) 
           in locations := {lat; lng} :: !locations;
           
           records := {recString = s;
                       distance = 0.} :: !records;
           incr recNum;
         done;
       with | End_of_file -> (
           Printf.printf "read %s, found %d records in total\n%!" !dbname !recNum;
           close_in fp););
      
    done;
   with
   | End_of_file -> close_in flist;);
  !recNum
  
let findLowest records distances numRecords topN =
  for i = 0 to topN - 1 do
    let minLoc = ref i in
    let tempRec = ref {
        recString ="";
        distance = 0.;
      }
    in
    let tempval = ref 0. in
    
    for j = i to numRecords - 1 do
      tempval := Mem.unsafe_get distances j;
      if !tempval < Mem.unsafe_get distances !minLoc then
        minLoc := j
    done;
    
    tempRec := records.(i);
    records.(i) <- records.(!minLoc);
    records.(!minLoc) <- !tempRec;

    let tempDist = Mem.get distances i in
    Mem.unsafe_set distances i (Mem.unsafe_get distances !minLoc);
    Mem.unsafe_set distances !minLoc tempDist;

    records.(i).distance <- Mem.unsafe_get distances i;
    
    
  done

  
let _ = 

  let i = 0 in
  let filename = ref "" in
  let resultsCount = ref 10 in
  let lat = ref 0. and lng = ref 0. in
  let devId = ref 0 in
  let arg0 = ("-f", Arg.String (fun s -> filename := s), "the filename that lists the data input files") 
  and arg1 = ("-r", Arg.Int (fun i -> resultsCount := i) , "the number of records to return (default: 10)")
  and arg2 = ("-lat", Arg.Float (fun f -> lat := f), "the latitude for nearest neighbors (default: 0)")
  and arg3 = ("-lng", Arg.Float (fun f -> lng := f), "the longitude for nearest neighbors (default: 0)")
  and arg4 = ("-d", Arg.Int (fun i -> devId := i), "Choose the device (default: 0)")
  in
  Arg.parse [arg0; arg1; arg2; arg3; arg4] (fun s -> ()) "";

  if !filename = "" then
    Arg.usage [arg0; arg1; arg2; arg3; arg4]  "Nearest Neighbor Usage\n NN.asm [filname] -r [int] -lat [int] -lng [int] -d [int]";
  
  let records = ref [] in
  let locs = ref [] in
  let numRecords : int = loadData !filename records locs in
  if (!resultsCount > numRecords) then
    resultsCount := numRecords;
  let dev = devices.(!devId) in
  
  let distances = Vector.create Vector.float32 ~dev numRecords 
  and locations = Vector.create (Vector.Custom customLatLong) numRecords
  in
  
  List.iteri (fun i a -> Mem.unsafe_set locations i a) !locs;
  
  
  
  let nIter = 10 in
  let elapsed_time = ref 0.0 in
  
  let threadsPerBlock = 1024 in
  let ceilDiv  a b = (a + b -1) / b in
  let blocks= ceilDiv  numRecords  threadsPerBlock in
  let gridY = ceilDiv blocks 65535 in
  let gridX = ceilDiv  blocks gridY in
  
  let block = {Kernel.blockX = threadsPerBlock; Kernel.blockY = 1; Kernel.blockZ = 1}
  and grid = {Kernel.gridX = gridX; Kernel.gridY = gridY; Kernel.gridZ = 1} in

  Printf.printf "nR : %d - blockX : %d, gridX : %d, gridY : %d : %d \n" numRecords threadsPerBlock gridX gridY blocks;
  let recArray = (Array.of_list (!records)) in  

  for dev_id = 0 to 2 do
    let dev = devices.(dev_id) in

    measure_time (fun () ->
        let kind = match dev.Devices.specific_info with
          | Devices.OpenCLInfo clI -> Devices.OpenCL
          | _ -> Devices.Cuda
        in
        ignore(Kirc.gen ~only:kind euclid dev);
      ) "Code generation";
    
    
    Mem.to_device locations dev;
    for  iter = 0 to nIter do
      
      let t0 = Unix.gettimeofday () in  
      
      Kirc.run euclid (locations, distances, numRecords, !lat, !lng) (block,grid) 0 dev;
      Devices.flush ~queue_id:0 dev ();
      
      
      if iter> 0 then
        elapsed_time := !elapsed_time +. (Unix.gettimeofday() -. t0);
      
      findLowest recArray  distances numRecords !resultsCount;
      Mem.to_device distances dev;
      Devices.flush dev ();
      
    done;
    
    Printf.printf "Dev : -> %s has computed %d times in %fs\n" dev.Devices.general_info.Devices.name (nIter) !elapsed_time;
    for i = 0 to !resultsCount -1 do
      Printf.printf "%s ---> Distance = %g\n" recArray.(i).recString recArray.(i).distance;
    done;
    print_endline "----------------------------";
  done;

  
  


