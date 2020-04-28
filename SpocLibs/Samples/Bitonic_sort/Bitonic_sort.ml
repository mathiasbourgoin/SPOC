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


let gpu_bitonic = kern v j k ->
  let open Std in
  let i = thread_idx_x + block_dim_x * block_idx_x in
  let ixj = Math.xor i j in
  let mutable temp = 0. in
  if ixj >= i then
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

let nearest_pow2 i  =
  let rec aux acc =
    let res = acc*2 in
    if res < i then
      aux res
    else acc in
  let r = aux 1 in
  if r <> i then
    Printf.printf "Changed size value to a power of two (%d -> %d)\n" i r;
  r

let () =
  let devid = ref 0
  and size = ref 1024 (*(1024*2*2*2*2*2*2*2*2*2*2*2*2*2*2*2)*)
  and check = ref true
  and compare = ref true
  in

  let arg1 = ("-device" , Arg.Int (fun i  -> devid := i),
	      "number of the device [0]")
  and arg2 = ("-size" , Arg.Int (fun i  -> size := nearest_pow2 i),
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
  and compare = !compare in
  let seq_vect  = Spoc.Vector.create Vector.float32 size

  and gpu_vect = Spoc.Vector.create Vector.float32 size
  and base_vect = Spoc.Vector.create Vector.float32 size
  and vect_as_array = Array.make size 0.
  in
  Random.self_init ();
  (* fill vectors with random values... *)
  for i = 0 to Vector.length seq_vect - 1 do
    let v = Random.float 255. in
    seq_vect.[<i>] <- v;
    gpu_vect.[<i>] <- v;
    base_vect.[<i>] <- v;
    vect_as_array.(i) <- v;
  done;

  if compare then
   begin
      measure_time "Sequential bitonic"
        (fun () -> Mem.unsafe_rw true; sortup seq_vect 0 (Vector.length seq_vect); Mem.unsafe_rw false);
      measure_time "Sequential Array.sort"
        (fun () -> Array.sort Stdlib.compare vect_as_array);
    end;
  let threadsPerBlock = match !dev.Devices.specific_info with
    | Devices.OpenCLInfo clI ->
      (match clI.Devices.device_type with
       | Devices.CL_DEVICE_TYPE_CPU -> 1
       | _  ->   1)
    | _  -> 1 in
  let blocksPerGrid =
    (size + threadsPerBlock -1) / threadsPerBlock
  in
  let block0 = {Spoc.Kernel.blockX = threadsPerBlock;
		Spoc.Kernel.blockY = 1; Spoc.Kernel.blockZ = 1}
  and grid0= {Spoc.Kernel.gridX = blocksPerGrid;
	      Spoc.Kernel.gridY = 1; Spoc.Kernel.gridZ = 1} in
  ignore(Kirc.gen gpu_bitonic devs.(!devid));

  let j,k = ref 0,ref 2 in
  let first = ref true in
  measure_time "Parallel Bitonic" (fun () ->
      while !k <= size do
        j := !k lsr 1;
        while !j > 0 do
          if !first then
            (Kirc.run gpu_bitonic (gpu_vect,!j,!k) (block0,grid0) 0 !dev;
             first := false);
          Kirc.run gpu_bitonic (gpu_vect,!j,!k) (block0,grid0) 0 !dev;
          j := !j lsr 1;
        done;
        k := !k lsl 1 ;
      done;
      Mem.to_cpu gpu_vect ();
      Devices.flush !dev ();
    );

  if check then
    (
      let r = ref (-. infinity) in
      let ok = ref true in
      for i = 0 to size - 1 do
        if gpu_vect.[<i>] < !r then
          ( Printf.printf "%d -> %s\n" i (Printf.sprintf "error, %g <  %g" gpu_vect.[<i>] !r);
            r := Mem.get gpu_vect i;
            ok := false ;)
        else (
          r := gpu_vect.[<i>]);
      done;
      if !ok then
        Printf.printf "Check OK\n"
      else
        Printf.printf "Check KO\n";
    )
;;
