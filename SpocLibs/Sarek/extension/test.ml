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




let k = kern a b ->
  let open Std in
  let f = (fun a b -> (a + b)) in
  reduce f  a b  
  ;;


let _ = 
  let d = Devices.init () in
  let dev = d.(0) in
  let v1 = (Vector.create Vector.float32 1024) in
  let v2 = (Vector.create Vector.int32 1024) in
  for i = 0 to 1023 do
    Mem.set v1 i  (float_of_int i);
  done;
  Kirc.gen k dev;
  let threadsPerBlock = match dev.Devices.specific_info with
    | Devices.OpenCLInfo clI ->
      (match clI.Devices.device_type with
       | Devices.CL_DEVICE_TYPE_CPU -> 1
       | _ -> 256)
    | _ -> 256
  in
  let blocksPerGrid = (1024 + threadsPerBlock -1) / threadsPerBlock in
  let block = { Spoc.Kernel.blockX = threadsPerBlock; Spoc.Kernel.blockY = 1 ; Spoc.Kernel.blockZ = 1;} in
  let grid = { Spoc.Kernel.gridX = blocksPerGrid; Spoc.Kernel.gridY = 1 ; Spoc.Kernel.gridZ = 1;} in
  List.iter (Printf.printf "%s\n")((fst k)#get_cuda_sources ());
  Kirc.run k (v1,v2) (block,grid) 0 dev;
  for i = 0 to 1023 do
    Printf.printf "%g -> %ld\n" (Mem.get v1 i) (Mem.get v2 i)
  done
