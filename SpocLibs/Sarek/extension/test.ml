open Spoc
open Kirc
open Spoc.Vector


    
    (*let height = ref 512
let width = ref 512l
let shiftx = ref 0l
let shifty = ref 0l
let zoom = ref 1.
let max_iter = ref 10000l

let mandelbrot = kern  img  ->
  let open Std in

  let y = thread_idx_y + (block_idx_y * block_dim_y) in
  let x = thread_idx_x + (block_idx_x * block_dim_x) in
  (if (y >= @height) || (x >= @width) then
     return () ;
   else return ();
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
  img.[<y * @width + x>] <- cpt;;


let _ =
  let devs = Spoc.Devices.init  ()
  in
  Kirc.gen ~profile:true  mandelbrot devs.(0);
  Printf.printf "Here\n%!";
  List.iter (Printf.printf "%s\n")((fst mandelbrot)#get_cuda_sources ())
    
*)



(* let k = kern a b i -> *)
(*   let open Std in  *)
(*   let tid = thread_idx_x + (block_idx_x * block_dim_x) in *)
(*   if (tid < i) then *)
(*     b.[<tid>] <- f (a.[<tid>]) *)


(* let _ = *)
(*   let d = Devices.init () in *)
(*   let v1 = (Vector.create Vector.int32 1024) in *)
(*   let v2 = (Vector.create Vector.int32 1024) in *)
(*   for i = 0 to 1023 do *)
(*     Mem.set v1 i  (Int32.of_int i); *)
(*     Mem.set v2 i  (Int32.of_int (1023 - i)) *)
(*   done; *)
  
(*   let res = Transform.zip ~dev:d.(0) (kern a b -> *)
(*                                       a + b *)
(*                                    )   v1 v2 in *)
(*   for i = 0 to 1023 do *)
(*     Printf.printf "%ld + %ld -> %ld\n" (Mem.get v1 i) (Mem.get v2 i) (Mem.get res  i) *)
(*   done *)




(* let k = kern a b -> *)
(*   let open Std in *)
(*   let f = (fun a b -> (a + b)) in *)
(*   reduce f  a b   *)
(*   ;; *)


(* let _ =  *)
(*   let d = Devices.init () in *)
(*   let dev = d.(0) in *)
(*   let v1 = (Vector.create Vector.float32 1024) in *)
(*   let v2 = (Vector.create Vector.int32 1024) in *)
(*   for i = 0 to 1023 do *)
(*     Mem.set v1 i  (float_of_int i); *)
(*   done; *)
(*   Kirc.gen k dev; *)
(*   let threadsPerBlock = match dev.Devices.specific_info with *)
(*     | Devices.OpenCLInfo clI -> *)
(*       (match clI.Devices.device_type with *)
(*        | Devices.CL_DEVICE_TYPE_CPU -> 1 *)
(*        | _ -> 256) *)
(*     | _ -> 256 *)
(*   in *)
(*   let blocksPerGrid = (1024 + threadsPerBlock -1) / threadsPerBlock in *)
(*   let block = { Spoc.Kernel.blockX = threadsPerBlock; Spoc.Kernel.blockY = 1 ; Spoc.Kernel.blockZ = 1;} in *)
(*   let grid = { Spoc.Kernel.gridX = blocksPerGrid; Spoc.Kernel.gridY = 1 ; Spoc.Kernel.gridZ = 1;} in *)
(*   List.iter (Printf.printf "%s\n")((fst k)#get_cuda_sources ()); *)
(*   Kirc.run k (v1,v2) (block,grid) 0 dev; *)
(*   for i = 0 to 1023 do *)
(*     Printf.printf "%g -> %ld\n" (Mem.get v1 i) (Mem.get v2 i) *)
(*   done *)


open Spoc
open Kirc

let smemSize = ref 0l
let blockSize = ref 0l

let k = kern (res:int32 vector) (a:int32 vector) nIsPow2 n ->
  let open Std in
  
  let (sdata: int32 array) = create_array @smemSize in
  let tid = thread_idx_x in
  let mutable i = block_idx_x * @blockSize*2 + thread_idx_x in
  let gridsize = @blockSize*2*grid_dim_x in
  
  let mutable acc = 0 in
  while (i < n) do
    acc :=  acc + a.[<i>];
    if (nIsPow2 = 1) || (i + @blockSize) < n then
      acc := acc + a.[<i + @blockSize>];
    
    i := i + grid_dim_x;
    
  done;
  
  sdata.(tid) <>- acc;
  block_barrier();
  
  if ( @blockSize >= 512) && (tid < 256) then
    (acc := acc + sdata.(tid + 256);
     sdata.(tid) <>- acc;
    );
  
  block_barrier();
  
  if ( @blockSize >= 256) && (tid < 128) then
    (acc := acc + sdata.(tid + 128);
     sdata.(tid) <>- acc;
    );
  block_barrier();
  
  if ( @blockSize >= 128) && (tid < 64) then
    (acc := acc + sdata.(tid + 64);
     sdata.(tid) <>- acc;
    );
  block_barrier();

  if ( @blockSize >= 64) && (tid < 32) then
    (acc := acc + sdata.(tid + 32);
     sdata.(tid) <>- acc;
    );
  block_barrier();
  
  if ( @blockSize >= 32) && (tid < 16) then
    (acc := acc + sdata.(tid + 16);
     sdata.(tid) <>- acc;
    );
  block_barrier();

    if ( @blockSize >= 16) && (tid < 8) then
    (acc := acc + sdata.(tid + 8);
     sdata.(tid) <>- acc;
    );
  block_barrier();
  
  if ( @blockSize >= 8) && (tid < 4) then
    (acc := acc + sdata.(tid + 4);
     sdata.(tid) <>- acc;
    );
  block_barrier();
  
  if ( @blockSize >= 4) && (tid < 2) then
    (acc := acc + sdata.(tid + 2);
     sdata.(tid) <>- acc;
    );
  block_barrier();
  
  
  if ( @blockSize >= 2) && (tid < 1) then
    (acc := acc + sdata.(tid + 1);
     sdata.(tid) <>- acc;
    );
  
  block_barrier();
  
  if (tid = 0) then
    res.[<block_idx_x>] <- acc;
  
;;
  

let reduce v dev threadsPerBlock =
  Kirc.gen k dev;
  let blocksPerGrid = ((Vector.length v) + threadsPerBlock -1) / threadsPerBlock in
  let block = { Spoc.Kernel.blockX = threadsPerBlock; Spoc.Kernel.blockY = 1 ; Spoc.Kernel.blockZ = 1;} in
  let grid = { Spoc.Kernel.gridX = blocksPerGrid; Spoc.Kernel.gridY = 1 ; Spoc.Kernel.gridZ = 1;} in
  let res = Vector.create Vector.int32 blocksPerGrid in
  for i = 0 to blocksPerGrid - 1 do
    Mem.set res i 0l;
  done;
  blockSize := Int32.of_int threadsPerBlock;
  Kirc.run ~recompile:true k (res, v, 1, Vector.length v) (block,grid)  0 dev;

  let r = ref 0 in
  for i = 0 to blocksPerGrid - 1 do
    r := !r + Int32.to_int (Mem.get res i);
  done;
  !r

  

    
let _ =
  let devs = Spoc.Devices.init  ()
  in
  let v1 = Vector.create Vector.int32 1024 in
  for i= 0 to Vector.length v1 - 1  do
    Mem.set v1 i 1l;
  done;

  let threads dev =
    match dev.Devices.specific_info with
    | Devices.OpenCLInfo clI ->
      (match clI.Devices.device_type with
       | Devices.CL_DEVICE_TYPE_CPU -> 1
       | _ -> 256)
    | _ -> 256
  in
  
  
  let threadsPerBlock = threads devs.(0) in
  smemSize :=
    Int32.of_int (if threadsPerBlock <= 32 then
                    2 * threadsPerBlock
                  else threadsPerBlock);
  let r = reduce v1 devs.(0) threadsPerBlock in
  Printf.printf "%s -> reduction : %d\n"  (devs.(0).Devices.general_info.Devices.name) r;
  (*List.iter (Printf.printf "%s\n") ((fst k)#get_cuda_sources ());*)
  
  let threadsPerBlock = threads devs.(3) in
  smemSize := Int32.of_int (
      if threadsPerBlock <= 32 then
        2 * threadsPerBlock
      else threadsPerBlock);
  let r = reduce v1 devs.(1) threadsPerBlock in
  Printf.printf "%s -> reduction : %d\n" (devs.(1).Devices.general_info.Devices.name) r;
  (*List.iter (Printf.printf "%s\n") ((fst k)#get_opencl_sources ())*)
