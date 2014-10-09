open Spoc

ktype cell = 
| Alive
| Dead


let step = kern field next_field width height ->
  let max = fun a  b ->
    if a > b then a else b 
  in
  let min = fun a b ->
    if a < b then a else b
  in
  let open Std in
  let i = thread_idx_x + block_dim_x * block_idx_x in
  let y = i / height in
  let x = i - (y*width) in
  if y < height && x < width then
    begin
      let left = max 0  (x - 1) in
      let right = min (width-1) (x+1) in
      let top = max 0 (y-1) in
      let bottom = min (height-1) (y +1) in
      let mutable cpt = 0 in
      for h = top to bottom do
        for w = left to right do
          match field.[<h*width+w>] with
          | Alive -> cpt := cpt + 1
          | Dead -> ()
        done;
      done;
      next_field.[<i>] <-
        match field.[<i>] with
        | Dead -> if cpt = 3 then Alive else Dead
        | Alive -> if cpt < 3 || cpt > 4 then Dead else Alive
    end
  

let couleur n =  
  match n with
  | Alive ->
    Graphics.red
  | Dead ->
    Graphics.blue
     

let _ = 
  Random.self_init ();
  let devid = ref 0
  and w = ref 1024
  and h = ref 1024 
  and nb_iter = ref 10_000 
  and bench = ref false in
  Arg.parse [
    ("-device" , Arg.Int (fun i  -> devid := i), "number of the device [0]");
    ("-height" , Arg.Int (fun i  -> h := i), "field height [1024]");
    ("-width" , Arg.Int (fun i  ->  w := i), "field width [1024]");
    ("-iter" , Arg.Int (fun i  ->  nb_iter := i), "nb steps [10 000]");
    ("-bench" , Arg.Bool (fun b  ->  bench := b), "run as benchmark [1024]")] 
    (fun s -> ()) "";
  let devs = Spoc.Devices.init ~only:Devices.OpenCL() in
  let dev = devs.(!devid) in
  let x1 = Vector.create (Custom customCell) (!h * !w) 
  and x2 = Vector.create (Custom customCell) (!h * !w) in
  ignore(Kirc.gen ~only:Devices.OpenCL step);
  (*Printf.printf "%s\n%!" (List.hd ((fst step)#get_opencl_sources ()));*)
  let threadsPerBlock = match dev.Devices.specific_info with
    | Devices.OpenCLInfo clI ->
      (match clI.Devices.device_type with
       | Devices.CL_DEVICE_TYPE_CPU -> 1
       | _ -> 256)
    | _ -> 256
  in
  let blocksPerGrid = ((!h * !w) + threadsPerBlock -1) / threadsPerBlock in
  let block = { Spoc.Kernel.blockX = threadsPerBlock; Spoc.Kernel.blockY = 1 ; Spoc.Kernel.blockZ = 1;} in
  let grid = { Spoc.Kernel.gridX = blocksPerGrid; Spoc.Kernel.gridY = 1 ; Spoc.Kernel.gridZ = 1;} in
  for i = 0 to !h-1 do
    for j = 0 to !w-1 do
      Mem.set x1 (i * !w + j) ( 
        if Random.int 255 mod 2 = 0 then 
          Alive else Dead)
    done;
  done;
  
  let l_ = string_of_int !w in
  let h_ = string_of_int !h in
  let dim = " " ^l_^"x"^h_ in
  let draw tab img = 
    Spoc.Tools.iteri (fun elt i -> 
	let b = i / !h in
 let a = i - (b * !w) in
 img.(b).(a) <-  (couleur elt)) tab ;  
    Graphics.draw_image (Graphics.make_image img) 0 0;		      
    Graphics.synchronize();
  in

  let img1 = Array.make_matrix 1024 1024 Graphics.black in
  Graphics.open_graph dim;
  draw x1 img1;
  for i = 0 to 10_000 do
    Kirc.run step (x1,x2,!h,!w) (block,grid) 0 dev;
    Devices.flush dev ();
    Kirc.run step (x2,x1,!h,!w) (block,grid) 0 dev;

    draw x1 img1;
  done;
  ignore (Graphics.read_key ());
  Graphics.close_graph ();;

 

