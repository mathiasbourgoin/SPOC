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

module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer
module Std = Sarek_stdlib.Std

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

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

let blockSize = ref 64l

let softening = ref 1e-9

(* Custom type for float4 *)
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
    V2_Vector.set data i {x; y; z; w}
  done

(* Kernel wrapped in a function to avoid value restriction *)
let bodyForce () =
  [%kernel
    fun (p : float4 vector) (v : float4 vector) (dt : float32) (n : int32) ->
      let open Std in
      let i = (block_idx_x * block_dim_x) + thread_idx_x in
      if i < n then (
        let fx = mut 0.0 in
        let fy = mut 0.0 in
        let fz = mut 0.0 in

        for tile = 0l to grid_dim_x - 1l do
          let spos : float32 array =
            create_array (Int32.to_int (3l * [%global blockSize])) Shared
          in
          let tpos = p.((tile * block_dim_x) + thread_idx_x) in
          spos.(3l * thread_idx_x) <- tpos.x ;
          spos.((3l * thread_idx_x) + 1l) <- tpos.y ;
          spos.((3l * thread_idx_x) + 2l) <- tpos.z ;
          block_barrier () ;

          pragma
            ["unroll"]
            (for j = 0l to [%global blockSize] - 1l do
               let dx = spos.(3l * j) -. p.(i).x in
               let dy = spos.((3l * j) + 1l) -. p.(i).y in
               let dz = spos.((3l * j) + 2l) -. p.(i).z in
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

  let devices =
    measure_time
      (fun () -> V2_Device.init ~frameworks:["CUDA"; "OpenCL"] ())
      "init"
  in
  if Array.length devices = 0 then begin
    Printf.eprintf "No devices found\n%!" ;
    exit 1
  end ;
  let dev = devices.(!devid) in
  let dt = 0.01 in

  Printf.printf
    "Will use device : %s (%s)\n%!"
    dev.V2_Device.name
    dev.V2_Device.framework ;

  (* Create V2 vectors with custom type *)
  let bodiesPos = V2_Vector.create_custom float4_custom_v2 !nBodies in
  let bodiesVel = V2_Vector.create_custom float4_custom_v2 !nBodies in

  randomizeBodies bodiesPos !nBodies ;
  randomizeBodies bodiesVel !nBodies ;

  let blockSize = Int32.to_int !blockSize in
  let blocksPerGrid = (!nBodies + blockSize - 1) / blockSize in
  let block = Sarek.Execute.dims1d blockSize in
  let grid = Sarek.Execute.dims1d blocksPerGrid in

  (* Get V2 IR *)
  let _, kirc = bodyForce () in
  let ir =
    match kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  let tot_time = ref 0.0 in

  for iter = 1 to !nIters do
    let t0 = Unix.gettimeofday () in
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        [
          Sarek.Execute.Vec bodiesPos;
          Sarek.Execute.Vec bodiesVel;
          Sarek.Execute.Float32 dt;
          Sarek.Execute.Int32 (Int32.of_int !nBodies);
        ]
      ~block
      ~grid
      () ;
    V2_Transfer.flush dev ;

    (* Update positions on CPU *)
    for i = 0 to !nBodies - 1 do
      let bP = V2_Vector.get bodiesPos i in
      let bV = V2_Vector.get bodiesVel i in
      V2_Vector.set
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
