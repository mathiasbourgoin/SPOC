open Spoc

ktype t1 = X | Y of int32

ktype t2 = 
{
  mutable x : t1;
  y : int32;
}

ktype t3 = 
 A
| B of int32
| C of t2

  ;;

let f = kern a b  ->
  let open Std in
  let i = thread_idx_x + block_dim_x * block_idx_x in
  b.[<i>] <- {x = Y 3l; y = i*3l};  
  (if (i mod 3l) = 0l then
    b.[<i>].x <- Y (b.[<i>].y));
   a.[<i>] <-
     (if (i mod 2l) = 0l then
      X
    else
      Y i);
  a.[<i>] <- 
    (match a.[<i>] with
     | X ->   Y i
     | Y x ->  Y (x*2))
;;

let _ =
  let devs = Spoc.Devices.init () in

  let x = Vector.create (Custom customT1) 1024
  and y =  Vector.create (Custom customT2) 1024
  and dev = 
    let i = 
      try int_of_string (Sys.argv.(1)) with | _ -> 0
    in devs.(i)
  in
  ignore(Kirc.gen ~only:Devices.OpenCL f);
  Printf.printf "%s\n%!" (List.hd ((fst f)#get_opencl_sources ()));
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
  for i = 0 to 1023 do
    let t = (if i mod 2 = 0 then X else Y  (Int32.of_int i)) in
    Mem.set x i t;
  done;
  Kirc.run f (x,y) (block,grid) 0 dev;
  for i = 0 to 1023 do
    Printf.printf "%d \n%!" i;
    let t = Mem.get x i in
    begin
      match t with
      | X -> print_endline "X";
      | Y i -> print_endline ("Y of "^(Int32.to_string i))
    end;
  done;
  Printf.printf "%s\n%!" (List.hd ((fst f)#get_opencl_sources ()));

  

