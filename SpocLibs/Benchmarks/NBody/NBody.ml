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


let devices = measure_time Spoc.Devices.init "init"
    
let blockSize = ref 256l
let softening = ref 1e-9

ktype float4 = {
                  mutable x :float64;
                mutable y:float64;
                mutable z : float64;
                w: float64;}
              
let randomizeBodies data n =
  for i = 0 to n - 1 do
    
    Mem.set data i {x= (2.0 *. (Random.float 1. /. max_float) -. 1.);
                    y= (2.0 *. (Random.float 1. /. max_float) -. 1.);
                    z= (2.0 *. (Random.float 1. /. max_float) -. 1.);
                    w= (2.0 *. (Random.float 1. /. max_float) -. 1.);
                   }
                    ;
  done
  
let bodyForce = kern p v dt n ->
  let open Std in
  let open Math in
  let open Float64 in
  let i = block_idx_x * block_dim_x + thread_idx_x in
  if (i < n) then
    
    let mutable fx = zero in
    let mutable fy = zero in
    let mutable fz = zero in
    
    for tile = 0 to grid_dim_x - 1 do
      let (spos : float64 array) = create_array (3* @blockSize ) in
      let tpos = p.[< tile * block_dim_x + thread_idx_x >] in
      spos.(3*thread_idx_x) <>- (tpos.x);
      
      spos.(3*thread_idx_x+1) <>- tpos.y;
      spos.(3*thread_idx_x+2) <>- tpos.z;
      block_barrier ();

      (*pragma "unroll"*)
      for  j =  0 to @blockSize - 1 do
        let dx = minus (spos.(3*j))  p.[<i>].x in
        let dy = minus (spos.(3*j+1))  p.[<i>].y in
        let dz = minus (spos.(3*j+2))  p.[<i>].z in
        let distSqr = add (add (mul dx  dx)  (mul dy dy)) (add (mul dz dz) @softening) in
        let (invDist:float64) = rsqrt distSqr  in
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

 let arg1 = ("nBodies", Arg.Int (fun i -> nBodies := i), "number of bodies")
 and arg2 = ("nIters", Arg.Int (fun i -> nIters := i), "number of iteration")
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
  measure_time (fun () -> 
      for iter = 1 to !nIters do
        ignore(Kirc.gen ~only:Devices.Cuda bodyForce dev)
          
      done;) "CUDA Code generation";
  
  measure_time (fun () ->
      for iter = 1 to !nIters do
        ignore(Kirc.gen ~only:Devices.OpenCL bodyForce dev)
          
      done;) "OpenCL Code generation";
  
  measure_time (fun () ->
      for iter = 1 to !nIters do
        
        (*  List.iter (Printf.printf "%s\n")((fst bodyForce)#get_cuda_sources ());*)
        Kirc.run bodyForce (bodiesPos, bodiesVel, dt, !nBodies) (block,grid) 0 dev;
        
        for i = 0 to !nBodies - 1 do
          (Mem.get bodiesPos i).x <- (Mem.get bodiesPos i).x +.  (Mem.get bodiesVel i).x *. dt;
          (Mem.get bodiesPos i).y <- (Mem.get bodiesPos i).y +. (Mem.get bodiesVel i).y *. dt;
          (Mem.get bodiesPos i).z <- (Mem.get bodiesPos i).z +. (Mem.get bodiesVel i).z *. dt;
        done;
        
      done;) "NBODY Computation";
  ()
  

;;

