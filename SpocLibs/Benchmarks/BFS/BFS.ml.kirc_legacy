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
open Sarek
open Kirc
let cpt = ref 0

let tot_time = ref 0.

let measure_time f  s =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "%s time %d : %Fs\n%!" s !cpt ((t1 -. t0)/.10.);
  tot_time := !tot_time +.  (t1 -. t0);
  incr cpt;
  a;;


let devices = measure_time (Spoc.Devices.init ) "init"


type node = {
  mutable starting : int32;
  mutable no_of_edges : int32;
} [@@sarek.type]


let max_threads_per_block = ref 512l

let bfs_kern1 () = [%kernel fun (g_graph_nodes : node vector) (g_graph_edges : int32 vector) (g_graph_mask : int32 vector) (g_updating_graph_mask : int32 vector) (g_graph_visited : int32 vector) (g_cost : int32 vector) (no_of_nodes : int32) ->
  let open Std in
  let tid = (block_idx_x * [%global max_threads_per_block]) + thread_idx_x in
  if (tid < no_of_nodes && (1 = g_graph_mask.%[tid]) ) then
    (
      g_graph_mask.%[tid] <- 0;
      for i = g_graph_nodes.%[tid].starting to (g_graph_nodes.%[tid].no_of_edges + g_graph_nodes.%[tid].starting - 1) do

        let id = g_graph_edges.%[i] in
        if not (1 = g_graph_visited.%[id]) then
          (
            g_cost.%[id] <- g_cost.%[tid] + 1;
            g_updating_graph_mask.%[id] <- 1;
          )
      done;
    )
]

let bfs_kern2 = [%kernel fun (g_graph_mask : int32 vector) (g_updating_graph_mask : int32 vector) (g_graph_visited : int32 vector) (g_over : int32 vector) (no_of_nodes : int32) ->
  let open Std in
  let tid = (block_idx_x * [%global max_threads_per_block]) + thread_idx_x in
  if( tid<no_of_nodes && (g_updating_graph_mask.%[tid] = 1) ) then
    (
      g_graph_mask.%[tid] <- 1;
      g_graph_visited.%[tid] <- 1;
      g_over.%[0] <- 1;
      g_updating_graph_mask.%[tid] <- 0;
    )
]


let usage () =
  Printf.eprintf "Usage: %s <input_file> [?device_id] \n%!" (Sys.argv.(0))


let bfs_graph () =

  if (Array.length Sys.argv) != 2 && (Array.length Sys.argv) != 3 then
    ( usage();
      exit 0;
    );

  (*print_endline "Reading File";*)


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


  (*  let graph_nodes = Vector.create (Vector.Custom node_custom) no_of_nodes *)
  let graph_mask = Vector.create Vector.int32 no_of_nodes
  and updating_graph_mask = Vector.create Vector.int32 no_of_nodes
  and graph_visited = Vector.create Vector.int32 no_of_nodes in

  let graph_nodes = Tools.map (fun _ ->
      Scanf.bscanf fs "%d %d "
        (fun a b ->
           {starting = Int32.of_int a;
            no_of_edges = Int32.of_int b}
        )) (Vector.Custom node_custom) graph_mask in

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

  (*  print_endline "Read file";*)

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

  (*print_endline ("Copied Everything to GPU memory");*)

  (*print_endline "Start traversing the tree";*)
  let k = ref 0 in


  Mem.set over 0 1l;


  let kern1 = bfs_kern1 () in
  measure_time (fun () ->
      let kind = match dev.Devices.specific_info with
        | Devices.OpenCLInfo clI -> Devices.OpenCL
        | _ -> Devices.Cuda
      in
      for i = 1 to 10 do
        ignore(Kirc.gen ~only:kind kern1 dev);
        ignore(Kirc.gen ~only:kind bfs_kern2 dev)
      done;
    ) "Code generation";


  let elapsed_time = ref 0. in

  while Mem.get over 0 = 1l  do

    let t0 = Unix.gettimeofday () in
    Mem.set over 0 0l;

    Kirc.run kern1 (graph_nodes, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes) (block,grid) 0 dev;

    Kirc.run bfs_kern2 (graph_mask, updating_graph_mask, graph_visited, over, no_of_nodes) (block,grid) 0 dev;

    incr k;
    if !k> 1 then
      elapsed_time := !elapsed_time +. (Unix.gettimeofday() -. t0);

  done;

  Printf.printf "Kernel Executed with %d times in %f : %d nodes \n---------------------\n" !k !elapsed_time no_of_nodes;

  (*
  let fpo = open_out "result.txt" in
  for i = 0 to no_of_nodes - 1 do
    Printf.fprintf fpo "%d) cost:%ld\n" i (Mem.get cost i);
  done;
  close_out fpo;
  print_endline "Result stored in result.txt";  *)
;;


let _ =
  bfs_graph ()
