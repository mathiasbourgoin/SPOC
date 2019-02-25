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

let measure_time f  s iter =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "%s time %d : %Fs -> average %Fs \n%!" s !cpt (t1 -. t0) ((t1 -. t0) /. (float iter));
  tot_time := !tot_time +.  (t1 -. t0);
  incr cpt;
  a;;


let devices = measure_time Spoc.Devices.init "init" 1

let tile_dim = ref 32l
let block_rows = ref 8l

let transpose = kern idata odata ->
 let open Std in
 let mutable x = thread_idx_x + block_idx_x * @tile_dim in
 let mutable y = thread_idx_y + block_idx_y * @tile_dim in
 let width = grid_dim_x * @tile_dim in
 
 (* [ dim ][ dim + 1] helps avoid shared memory banks conflicts *)
 let ( tile : float32 array ) = create_array ( @tile_dim * ( @tile_dim +1) ) in
 let mutable j = 0 in
   
 (* fill the shared array in parallel *)
 while j < @tile_dim do
   tile.( ( thread_idx_y + j )* @tile_dim + thread_idx_x) <>- (idata.[<( y +j )* width + x >] : float32);
   j := j + @block_rows ;
 done;
 (* block - level synchronization *)
 block_barrier () ;
 
 x := thread_idx_x + block_idx_y * @tile_dim;
 y := thread_idx_y + block_idx_x * @tile_dim;
 j := 0;
 while j < @tile_dim do
   odata .[<( y + j) * width + x >] <- tile .(thread_idx_x * @tile_dim + (thread_idx_y +j) );
   j := j + @block_rows ;
 done

let _ =
  let nx = ref 1024
  and ny = ref 1024 in
  let dev_id = ref 0 in

  let arg0 = ("-d", Arg.Int (fun i -> dev_id := i), "Choose the device (default: 0)") in
  let arg1 = ("-nx", Arg.Int (fun i -> nx := i), "Matrix width (default 1024)") in
  let arg2 = ("-ny", Arg.Int (fun i -> ny := i), "Matrix height (default 1024)") in
  Arg.parse [arg0; arg1; arg2] (fun s -> ()) "";
  
  let m1 = Vector.create Vector.float32 (!nx * !ny)
  and m2 = Vector.create Vector.float32 (!nx * !ny)
  in
  
  let dev = devices.(!dev_id)   in
  measure_time (fun () ->
      let kind = match dev.Devices.specific_info with
        | Devices.OpenCLInfo clI -> Devices.OpenCL
        | _ -> Devices.Cuda
      in
      for i = 1 to 10 do
        ignore(Kirc.gen ~only:kind transpose dev);
      done;
    ) "Code generation" 10;


  for i = 0 to (!nx  -1 )* (!ny -1) do
    Mem.set m1 i (float i);
  done;
  Mem.to_device m1 dev;
  Mem.to_device m2 dev;


 
  let open Kernel in 
  let block = {blockX = Int32.to_int !tile_dim;
               blockY = Int32.to_int !block_rows;
               blockZ = 1}
  and grid = {gridX = !nx / (Int32.to_int !tile_dim);
              gridY = !ny / (Int32.to_int !tile_dim);
              gridZ = 1}
  in
  
  Kirc.run transpose (m1, m2) (block,grid) 0 dev;

  Devices.flush dev ();
  Printf.printf "Starting iterations\n%!";

  
  measure_time (fun () ->
  for iter = 1 to 10 do
    Kirc.run transpose (m1, m2) (block,grid) 0 dev;
    Devices.flush ~queue_id:0 dev ();
  done;) "Transposition kernel" 10;


               
     (*let blockSize = ref 64l
let softening = ref 1e-9

ktype float4 = {
                  mutable x :float32;
                  mutable y:float32;
                  mutable z : float32;
                  w: float32;}
                
let randomizeBodies data n =
  for i = 0 to n - 1 do

    let x = (2.0 *. (Random.float 1.) -. 1.);
    and y = (2.0 *. (Random.float 1.) -. 1.);
    and z = (2.0 *. (Random.float 1.) -. 1.);
    and w = 0. in
    Mem.unsafe_set data i {x; y; z; w;}
    ;
  done
  
let bodyForce = kern p v dt n ->
  let open Std in
  let open Math in
  let open Float32 in
  let i = block_idx_x * block_dim_x + thread_idx_x in
  if (i < n) then
    
    let mutable fx = zero in
    let mutable fy = zero in
    let mutable fz = zero in

    let dist = fun a b c ->
      let open Math in let open Float32 in 
      add (add (mul a a) (mul b b)) (mul c c) in
      
    for tile = 0 to grid_dim_x - 1 do
      let (spos : float32 array) = create_array (3* @blockSize ) in
      let tpos = p.[< tile * block_dim_x + thread_idx_x >] in
      spos.(3*thread_idx_x) <>- tpos.x;
      spos.(3*thread_idx_x+1) <>- tpos.y;
      spos.(3*thread_idx_x+2) <>- tpos.z;
      block_barrier ();
      
      
      pragma "unroll"
      for  j =  0 to @blockSize - 1 do
        let dx = minus (spos.(3*j))  p.[<i>].x in
        let dy = minus (spos.(3*j+1))  p.[<i>].y in
        let dz = minus (spos.(3*j+2))  p.[<i>].z in
        let distSqr = add (dist dx dy dz)  @softening in
        let (invDist:float32) = rsqrt distSqr  in
        let invDist3 = mul  (mul invDist invDist) invDist in
        
        fx := add fx (mul dx invDist3);
        fy := add fy (mul dy invDist3);
        fz := add fz (mul dz invDist3);
      done;
      block_barrier ();
      
    done;
    v.[<i>].x <- add v.[<i>].x (mul dt fx);
    v.[<i>].y <- add v.[<i>].y (mul dt fy);
    v.[<i>].z <- add v.[<i>].z (mul dt fz)
;;    


let _ =
  let devid = ref 0 in


 let nBodies = ref 30_000 in
 let nIters = ref 10 in

 let arg1 = ("-nBodies", Arg.Int (fun i -> nBodies := i), "number of bodies")
 and arg2 = ("-nIters", Arg.Int (fun i -> nIters := i), "number of iteration")
 and arg3 = ("-device" , Arg.Int (fun i  -> devid := i),
	     "number of the device [0]")
            
  in
  Arg.parse [arg1; arg2; arg3] (fun s -> ()) "";

  let dev = devices.(!devid) in
  let dt = 0.01 in

  Printf.printf "Will use device : %s\n%!"
    (dev).Spoc.Devices.general_info.Spoc.Devices.name;
  
  let bodiesPos = Vector.create (Vector.Custom customFloat4) !nBodies in
  let bodiesVel = Vector.create (Vector.Custom customFloat4) !nBodies in
  
  randomizeBodies bodiesPos !nBodies;
  randomizeBodies bodiesVel !nBodies;
  
  let blockSize = Int32.to_int !blockSize in
  let blocksPerGrid =
    (!nBodies + blockSize - 1) / blockSize in
  let block = {Kernel.blockX = blockSize; Kernel.blockY= 1; Kernel.blockZ = 1} 
  and grid = {Kernel.gridX = blocksPerGrid; Kernel.gridY = 1; Kernel.gridZ = 1}
  in
  (*measure_time (fun () -> 
      for iter = 1 to !nIters do
        ignore(Kirc.gen ~only:Devices.Cuda ~nvrtc_options:[|"-ftz=true"|] bodyForce dev)
              
      done;) "CUDA Code generation";*)
  
  measure_time (fun () ->
      for iter = 1 to !nIters do
        ignore(Kirc.gen ~only:Devices.OpenCL bodyForce dev)
              
      done;) "OpenCL Code generation";
  
  

  let tot_time = ref 0.0 in

  for iter = 1 to !nIters do
    
    let t0 = Unix.gettimeofday () in
    
    Kirc.run bodyForce (bodiesPos, bodiesVel, dt, !nBodies) (block,grid) 0 dev;    
    
    
    
    
    for i = 0 to !nBodies - 1 do
      let bP,bV = (Mem.get bodiesPos i), (Mem.get bodiesVel i) in
      Mem.set bodiesPos i
              {x = bP.x +. bV.x *. dt;
               y = bP.y +. bV.y *. dt;
               z = bP.z +. bV.z *. dt;
               w = 0.};     
    done;
    
    let tElapsed = (Unix.gettimeofday() -. t0) in
    
    
    if iter > 1 then
      tot_time := !tot_time +. tElapsed;
    
    Printf.printf "Iteration %d: %.3f seconds\n%!"  iter  tElapsed;                                                                                                                                                        

*  done;
  Printf.printf "Total time : %f\n" !tot_time;




;;

 *)
