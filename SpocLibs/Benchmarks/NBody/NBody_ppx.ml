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

type float32 = float

let cpt = ref 0

let tot_time = ref 0.

let measure_time f s =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "%s time %d : %Fs\n%!" s !cpt (t1 -. t0) ;
  tot_time := !tot_time +. (t1 -. t0) ;
  incr cpt ;
  a

let devices = measure_time Spoc.Devices.init "init"

let blockSize = ref 64l

let softening = ref 1e-9

(* PPX version: ktype -> [@@sarek.type] *)
type float4 = {
  mutable x : float32;
  mutable y : float32;
  mutable z : float32;
  w : float32;
}
[@@sarek.type]

let randomizeBodies data n =
  for i = 0 to n - 1 do
    let x = (2.0 *. Random.float 1.) -. 1.
    and y = (2.0 *. Random.float 1.) -. 1.
    and z = (2.0 *. Random.float 1.) -. 1.
    and w = 0. in
    Mem.unsafe_set data i {x; y; z; w}
  done

(* PPX version: kern -> [%kernel fun ...]
   Use a function to avoid value restriction issues with kernel objects
   that contain mutable state (hashtables for binaries, sources, etc.) *)
let bodyForce () =
  [%kernel
    fun (p : float4 vector) (v : float4 vector) (dt : float32) (n : int32) ->
      let open Std in
      let i = (block_idx_x * block_dim_x) + thread_idx_x in
      if i < n then (
        let fx = mut 0.0 in
        let fy = mut 0.0 in
        let fz = mut 0.0 in

        for tile = 0 to grid_dim_x - 1 do
          let spos : float32 array =
            create_array (3 * [%global blockSize]) Shared
          in
          let tpos = p.((tile * block_dim_x) + thread_idx_x) in
          spos.(3 * thread_idx_x) <- tpos.x ;
          spos.((3 * thread_idx_x) + 1) <- tpos.y ;
          spos.((3 * thread_idx_x) + 2) <- tpos.z ;
          block_barrier () ;

          pragma
            ["unroll"]
            (for j = 0 to [%global blockSize] - 1 do
               let dx = spos.(3 * j) -. p.(i).x in
               let dy = spos.((3 * j) + 1) -. p.(i).y in
               let dz = spos.((3 * j) + 2) -. p.(i).z in
               let distSqr =
                 (dx *. dx) +. (dy *. dy) +. (dz *. dz) +. [%global softening]
               in
               let invDist : float32 = rsqrt distSqr in
               let invDist3 = invDist *. invDist *. invDist in
               fx := fx +. (dx *. invDist3) ;
               fy := fy +. (dy *. invDist3) ;
               fz := fz +. (dz *. invDist3)
             done) ;
          block_barrier ()
        done ;
        v.(i).x <- v.(i).x +. (dt *. fx) ;
        v.(i).y <- v.(i).y +. (dt *. fy) ;
        v.(i).z <- v.(i).z +. (dt *. fz))]

let () =
  let devid = ref 0 in
  let nBodies = ref 30_000 in
  let nIters = ref 10 in

  let arg1 = ("-nBodies", Arg.Int (fun i -> nBodies := i), "number of bodies")
  and arg2 = ("-nIters", Arg.Int (fun i -> nIters := i), "number of iteration")
  and arg3 =
    ("-device", Arg.Int (fun i -> devid := i), "number of the device [0]")
  in
  Arg.parse [arg1; arg2; arg3] (fun _ -> ()) "" ;

  let dev = devices.(!devid) in
  let dt = 0.01 in

  Printf.printf
    "Will use device : %s\n%!"
    dev.Spoc.Devices.general_info.Spoc.Devices.name ;
  let bodiesPos = Vector.create (Vector.Custom float4_custom) !nBodies in
  let bodiesVel = Vector.create (Vector.Custom float4_custom) !nBodies in

  randomizeBodies bodiesPos !nBodies ;
  randomizeBodies bodiesVel !nBodies ;

  let blockSize = Int32.to_int !blockSize in
  let blocksPerGrid = (!nBodies + blockSize - 1) / blockSize in
  let block = Kernel.{blockX = blockSize; blockY = 1; blockZ = 1}
  and grid = Kernel.{gridX = blocksPerGrid; gridY = 1; gridZ = 1} in
  measure_time
    (fun () ->
      for _iter = 1 to !nIters do
        ignore
          (Sarek.Kirc.gen
             ~only:Devices.Cuda
             ~nvrtc_options:[|"-ftz=true"|]
             (bodyForce ())
             dev)
      done)
    "CUDA Code generation" ;

  measure_time
    (fun () ->
      for _iter = 1 to !nIters do
        ignore (Sarek.Kirc.gen ~only:Devices.OpenCL (bodyForce ()) dev)
      done)
    "OpenCL Code generation" ;

  let tot_time = ref 0.0 in

  for iter = 1 to !nIters do
    let t0 = Unix.gettimeofday () in
    Sarek.Kirc.run
      (bodyForce ())
      (bodiesPos, bodiesVel, dt, !nBodies)
      (block, grid)
      0
      dev ;
    for i = 0 to !nBodies - 1 do
      let bP, bV = (Mem.get bodiesPos i, Mem.get bodiesVel i) in
      Mem.set
        bodiesPos
        i
        {
          x = bP.x +. (bV.x *. dt);
          y = bP.y +. (bV.y *. dt);
          z = bP.z +. (bV.z *. dt);
          w = 0.;
        }
    done ;
    let tElapsed = Unix.gettimeofday () -. t0 in
    if iter > 1 then tot_time := !tot_time +. tElapsed ;
    Printf.printf "Iteration %d: %.3f seconds\n%!" iter tElapsed
  done ;
  Printf.printf "Total time : %f\n" !tot_time
