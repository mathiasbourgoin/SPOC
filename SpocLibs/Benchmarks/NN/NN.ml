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

let euclid = kern location distances numRecords lat lng ->
  let open Std in
  let open Math.Float32 in
  let globalId = block_dim_x * (grid_dim_x * block_idx_y+block_idx_x) + thread_idx_x in
  if (globalId < numRecords) then
    let latLong = location.[< globalId >] in
    let distance = fun lat lng latLong ->
      let open Math.Float32 in
      add (mul  (minus lat latLong.lat) (minus lat latLong.lat))
        (mul (minus lng  latLong.lng) (minus lng  latLong.lng))
    in
    distances.[<globalId>] <- sqrt (
        (distance lat lng latLong))
        

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
           let latLong,recS =
             let s = input_line fp in
             Scanf.sscanf s "%i %i %i %i %i %s %f %f %i %i "
               (fun y _ _ _ _ n lat lng _ _ ->
                  { lat; lng},(Printf.sprintf "%i %s %f %f" y n lat lng))
           in locations := latLong :: !locations;

           records := {recString = recS;
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
      tempval := Mem.get distances j;
      if !tempval < Mem.get distances !minLoc then
        minLoc := j
    done;
    
    let tempRec = records.(i) in
    records.(i) <- records.(!minLoc);
    records.(!minLoc) <- tempRec;

    let tempDist = Mem.get distances i in
    Mem.set distances i (Mem.get distances !minLoc);
    Mem.set distances !minLoc tempDist;

    records.(i).distance <- Mem.get distances i;
    
    
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
  let quiet = 0 and timing = 0 and platform = 0 and device = 0 in
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
  
  measure_time (fun () ->
      let kind = match dev.Devices.specific_info with
        | Devices.OpenCLInfo clI -> Devices.OpenCL
        | _ -> Devices.Cuda
      in
      ignore(Kirc.gen ~only:kind euclid dev);
    ) "Code generation";
  
  
  let nIter = 10 in
  let elapsed_time = ref 0.0 in
  
  let threadsPerBlock = 256 in
  let ceilDiv  a b = (a + b -1) / b in
  let blocks= ceilDiv  numRecords  threadsPerBlock in
  let gridY = ceilDiv blocks 65535 in
  let gridX = ceilDiv  blocks gridY in
  
  let block = {Kernel.blockX = threadsPerBlock; Kernel.blockY = 1; Kernel.blockZ = 1}
  and grid = {Kernel.gridX = gridX; Kernel.gridY = gridY; Kernel.gridZ = 1} in

  Printf.printf "nR : %d - blockX : %d, gridX : %d, gridY : %d : %d \n" numRecords threadsPerBlock gridX gridY blocks;
  let recArray = (Array.of_list (!records)) in  

  for  iter = 0 to nIter do
    
    let t0 = Unix.gettimeofday () in  
    
    Kirc.run euclid (locations, distances, numRecords, !lat, !lng) (block,grid) 0 dev;
    Devices.flush ~queue_id:0 dev ();


    findLowest recArray  distances numRecords !resultsCount;
    
    if iter> 0 then
      elapsed_time := !elapsed_time +. (Unix.gettimeofday() -. t0);

  done;

  Printf.printf "Computed %d times in %fs\n" (nIter) !elapsed_time;

  for i = 0 to !resultsCount -1 do
    Printf.printf "%s ---> Distance=%f\n" recArray.(i).recString recArray.(i).distance;
  done
  
  


  
(*ktype node = {
                mutable starting : int32;
                mutable no_of_edges : int32;
              }


  let max_threads_per_block = ref 512l

  let bfs_kern1 = kern g_graph_nodes g_graph_edges g_graph_mask g_updating_graph_mask g_graph_visited g_cost no_of_nodes ->
  let open Std in
  let tid = (block_idx_x * @max_threads_per_block) + thread_idx_x in
  if (tid < no_of_nodes && (1 = g_graph_mask.[<tid>]) ) then
    (
      g_graph_mask.[<tid>] <- 0;
      for i = g_graph_nodes.[<tid>].starting to (g_graph_nodes.[<tid>].no_of_edges + g_graph_nodes.[<tid>].starting -1) do

        let id = g_graph_edges.[<i>] in
        if !(1 = g_graph_visited.[<id>] ) then
          (
            g_cost.[<id>] <- g_cost.[<tid>] + 1;
            g_updating_graph_mask.[<id>] <- 1;
          )
      done;
    )

  let bfs_kern2 = kern g_graph_mask g_updating_graph_mask g_graph_visited g_over no_of_nodes ->
  let open Std in
  let tid = (block_idx_x * @max_threads_per_block) + thread_idx_x in
    if( tid<no_of_nodes && (g_updating_graph_mask.[<tid>] = 1) ) then
      (
        g_graph_mask.[<tid>] <- 1;
        g_graph_visited.[<tid>] <- 1;
        g_over.[<0>] <- 1;
        g_updating_graph_mask.[<tid>] <- 0;
      )


  let usage () =
  Printf.eprintf "Usage: %s <input_file> [?device_id] \n%!" (Sys.argv.(0))


  let bfs_graph () =

  if (Array.length Sys.argv) != 2 && (Array.length Sys.argv) != 3 then
   ( usage();
     exit 0;
   );
  print_endline "Reading File";


  let dev =
    if (Array.length Sys.argv) = 3 then
      devices.(int_of_string Sys.argv.(2))
    else devices.(0)
  in


  let fs =
    try
      Scanf.Scanning.open_in Sys.argv.(1)
    with | Sys_error _  -> failwith ("Error reading graph file : "^Sys.argv.(1))
  in


  (*let fs = Scanf.Scanning.from_channel fp in*)

  let no_of_nodes = Scanf.bscanf fs "%d " (fun i -> i) in

  let num_of_blocks = ref 1. in
  let num_of_threads_per_block = ref no_of_nodes in

  if no_of_nodes > (Int32.to_int !max_threads_per_block) then
    (
      num_of_blocks := ceil ((float no_of_nodes) /. (Int32.to_float !max_threads_per_block));
      num_of_threads_per_block := Int32.to_int !max_threads_per_block;
    );


  (*  let graph_nodes = Vector.create (Vector.Custom customNode) no_of_nodes *)
  let graph_mask = Vector.create Vector.int32 no_of_nodes 
  and updating_graph_mask = Vector.create Vector.int32 no_of_nodes
  and graph_visited = Vector.create Vector.int32 no_of_nodes in

  let graph_nodes = Tools.map (fun _ ->
      Scanf.bscanf fs "%d %d "
        (fun a b ->
           {starting = Int32.of_int a;
            no_of_edges = Int32.of_int b}
        )) (Vector.Custom customNode) graph_mask in

  for i = 0 to no_of_nodes - 1 do
    Mem.unsafe_set graph_mask i  0l;
    Mem.unsafe_set updating_graph_mask i 0l;
    Mem.unsafe_set graph_visited i 0l;
  done;




  let source = ref (Scanf.bscanf fs "%d " (fun i -> i)) in
  source := 0;
  Mem.unsafe_set graph_mask !source 1l;
  Mem.unsafe_set graph_visited !source 1l;

  let edge_list_size = Scanf.bscanf fs "%d " (fun i -> i) in

  let graph_edges = Vector.create Vector.int32 edge_list_size in

  for i = 0 to edge_list_size - 1 do
    Scanf.bscanf fs "%d " (fun a -> Mem.unsafe_set graph_edges i (Int32.of_int a));
    Scanf.bscanf fs "%d " (fun a -> ());
    (* if i = 4 || i = 118 then *)
    (*   Printf.printf "\n\n %ld  \n\n" (Mem.get graph_edges i) *)
  done;

  Scanf.Scanning.close_in fs;

  print_endline "Read file";

  let cost = Vector.create Vector.int32 no_of_nodes in
  for i = 0 to no_of_nodes - 1 do
    Mem.unsafe_set cost i (-1l)
  done;
  Mem.set cost 0 0l;

  let over = Vector.create Vector.int32 1 in
  Mem.set over 0 0l;

  let grid =
    {Kernel.gridX =  int_of_float !num_of_blocks;
     Kernel.gridY = 1; Kernel.gridZ = 1;}
  and block = {Kernel.blockX = !num_of_threads_per_block;
               Kernel.blockY = 1; Kernel.blockZ = 1} in






  Mem.to_device graph_nodes dev;
  Mem.to_device graph_edges dev;
  Mem.to_device graph_mask dev;
  Mem.to_device updating_graph_mask dev;
  Mem.to_device graph_visited dev;
  Mem.to_device cost dev;
  Spoc.Devices.flush dev ();

  print_endline ("Copied Everything to GPU memory");

  print_endline "Start traversing the tree";
  let k = ref 0 in


  Mem.set over 0 1l;


  measure_time (fun () ->
      let kind = match dev.Devices.specific_info with
        | Devices.OpenCLInfo clI -> Devices.OpenCL
        | _ -> Devices.Cuda
      in
      Kirc.gen ~only:kind bfs_kern1 dev;
      Kirc.gen ~only:kind bfs_kern2 dev;
    ) "Code generation";


  let elapsed_time = ref 0. in

  while Mem.get over 0 = 1l  do

    let t0 = Unix.gettimeofday () in  
    Mem.set over 0 0l;  

    Kirc.run bfs_kern1 (graph_nodes, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes) (block,grid) 0 dev;

    Kirc.run bfs_kern2 (graph_mask, updating_graph_mask, graph_visited, over, no_of_nodes) (block,grid) 0 dev;    

    incr k;
    if !k> 1 then
      elapsed_time := !elapsed_time +. (Unix.gettimeofday() -. t0);

  done;

  Printf.printf "Kernel Executed %d times in %f\n" !k !elapsed_time;

  let fpo = open_out "result.txt" in
  for i = 0 to no_of_nodes - 1 do
    Printf.fprintf fpo "%d) cost:%ld\n" i (Mem.get cost i);
  done;
  close_out fpo;
  print_endline "Result stored in result.txt";  
  ;;


  let _ =
    bfs_graph ()

*)
