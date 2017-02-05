(*let mandelbrot = kern  img  ->
  let open Std in

  let y = thread_idx_y + (block_idx_y * block_dim_y) in
  let x = thread_idx_x + (block_idx_x * block_dim_x) in
  (if (y >= @height) || (x >= @width) then
      return () ;
  );
  let x0 = (x + @shiftx) in
  let y0 = (y + @shifty) in
  let mutable cpt = 0 in
  let mutable x1 = 0. in
  let mutable y1 = 0. in
  let mutable x2 = 0. in
  let mutable y2 = 0. in
  let a = 4. *. ((float x0) /. (float @width)) /. @zoom  -. 2. in
  let b = 4. *. ((float y0) /. (float @height)) /. @zoom -. 2. in

  let normalize = fun x y ->
    let pow2 = fun x -> x *. x in
    (pow2 x) +. (pow2 y) in

  let mutable norm = normalize x1  y1


  in
  while ((cpt < @max_iter) && (norm <=. 4.)) do
    cpt := (cpt + 1);
    x2 := (x1 *. x1) -. (y1 *. y1) +. a;
    y2 :=  (2. *. x1 *. y1 ) +. b;
    x1 := x2;
    y1 := y2;
    norm := (x1 *. x1 ) +. ( y1 *. y1);
  done;
  img.[<y * @width + x>] <- cpt
*)
(*
open Spoc
open Kirc
open Spoc.Vector

*)

(*let f = kern (a:float vector) x ->
  let i = Std.global_thread_id in
  a.[<i>] <- Math.Float32.sin x
*)

(*
ktype point = {x:float;y:float}

klet id = fun p -> {x=p.x;y=p.y}

let toy = kern c n -> let open Std in let i = thread_idx_x + block_dim_x * block_idx_x in if i < n then c.[<i>] <- id {x=1.;y=2.}

let _ =
  Spoc.Devices.init () in
Kirc.gen ~only:Devices.OpenCL toy;
List.iter (Printf.printf  "%s\n")((fst toy)#get_opencl_sources ())
*)

open Spoc
open Kirc

    ktype point = {x:float;y:float}

(*klet id = fun p -> {x=p.x;y=p.y}*)

let toy = kern c n ->
  let open Std in
  let i = thread_idx_x + block_dim_x * block_idx_x in
  if i < n then c.[<i>] <- (*id*)  {x=1.;y=2.}

let () =
  let devs = Spoc.Devices.init () in
  let dev = devs.(0) in
  let n = 100 in
  let c =  Vector.create (Vector.Custom customPoint) n in
  ignore(Kirc.gen ~only:Devices.OpenCL toy);
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
  Kirc.run toy ( c, n) (block,grid) 0 dev;
  let i = 0 in
  let p = Spoc.Mem.get c i in
    Printf.printf "\n%f %f" p.x p.y;

    
