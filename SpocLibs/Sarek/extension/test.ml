open Spoc

ktype t1 = X | Y of int

ktype t2 = 
{
  x : t1;
  y : int;
}


ktype t3 = 
 A
| B of int
| C of t2

  ;;




let f = kern a ->
  let open Std in
  let i = thread_idx_x + block_dim_x * block_idx_x in
  a.[<i>] <- X;;




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
    let t = (if i mod 2 = 0 then X else Y i) in
    Mem.set x i t;
  done;
  Kirc.run f (x) (block,grid) 0 dev;
(*  for i = 0 to 1023 do
    let t = (if i mod 2 = 0 then X else Y i) in
    Mem.set x i t;
    Mem.set y i {x = t; y = i*i}; 
  done;
  Mem.to_device x dev;
  Devices.flush dev (); *)
  for i = 0 to 1023 do
    Printf.printf "%d \n%!" i;
    let t = Mem.get x i in
    begin
      match t with
      | X -> print_endline "X";
      | Y i -> print_endline ("Y of "^(string_of_int i))
    end;
  done;
  

  

