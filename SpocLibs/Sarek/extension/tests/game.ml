open Spoc
open Tsdl

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
  


let cpt = ref 0

let tot_time = ref 0.

let measure_time f s =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "%s : time %d : %Fs\n%!" s !cpt (t1 -. t0);
  tot_time := !tot_time +.  (t1 -. t0);
  incr cpt;
  a;;


let couleur n r =  
  match n with
  | Alive ->
    Sdl.set_render_draw_color r 255 0 0 255       
  | Dead ->
    Sdl.set_render_draw_color r 0 0 255 255 

      


    
let init id h w bench = 
  let devs = Spoc.Devices.init ~only:Devices.OpenCL () in
  let dev = devs.(id) in
  let x1 = Vector.create (Custom customCell) (h * w) 
  and x2 = Vector.create (Custom customCell) (h * w) 
  and x1_cpu = Array.create (h * w) Dead  
  and x2_cpu = Array.create (h * w) Dead  in
  for i = 0 to h-1 do
    for j = 0 to w-1 do
      let c =  
        if Random.int 255 mod 2 = 0 then 
          Alive else Dead
      in
      Mem.set x1 (i * w + j) c;
      x1_cpu.(i * w + j) <- c;
    done;
  done;
    let win = match Sdl.init Sdl.Init.video with 
      | `Error e -> Sdl.log "Init error: %s" e; exit 1
      | `Ok () -> 
        match Sdl.create_window ~w:w ~h:h "SDL OpenGL" Sdl.Window.opengl with 
        | `Error e -> Sdl.log "Create window error: %s" e; exit 1
        | `Ok w -> w in
    if bench then 
      (Sdl.destroy_window win;
       Sdl.quit (););
    x1,x2,x1_cpu,x2_cpu,dev,win


let gpu_compute x1 x2 dev h w nb_iter bench win = 
  Printf.printf "Will use device : %s\n%!" dev.Spoc.Devices.general_info.Spoc.Devices.name;
  ignore(Kirc.gen ~only:Devices.OpenCL step);
  (*Printf.printf "%s\n%!" (List.hd ((fst step)#get_opencl_sources ()));*)
  let threadsPerBlock = match dev.Devices.specific_info with
    | Devices.OpenCLInfo clI ->
      (match clI.Devices.device_type with
       | Devices.CL_DEVICE_TYPE_CPU -> 1
       | _ -> 256)
    | _ -> 256
  in
  let blocksPerGrid = ((h * w) + threadsPerBlock -1) / threadsPerBlock in
  let block = { Spoc.Kernel.blockX = threadsPerBlock; Spoc.Kernel.blockY = 1 ; Spoc.Kernel.blockZ = 1;} in
  let grid = { Spoc.Kernel.gridX = blocksPerGrid; Spoc.Kernel.gridY = 1 ; Spoc.Kernel.gridZ = 1;} in

  
  let l_ = string_of_int w in
  let h_ = string_of_int h in
  let dim = " " ^l_^"x"^h_ in
  if not bench then
    (
      let r1 = match Sdl.create_renderer ~flags:Sdl.Renderer.accelerated win with
        | ` Ok r -> r
        | `Error s  -> failwith s
      in
      
      let draw tab r = 
        Devices.flush dev ();
        for i = 0 to h*w -10 do
          let b = i / h in
          let a = i - (b * w) in
          (couleur (Mem.get tab i) r);
          Sdl.render_draw_point r b a;
        done;
        Sdl.render_present r;
      in
      
      for i = 0 to nb_iter/2 do
        Kirc.run step (x1,x2,h,w) (block,grid) 0 dev;
        Kirc.run step (x2,x1,h,w) (block,grid) 0 dev;
        if not bench then
          ( draw x1 r1;
          );
      done;
      Sdl.quit ();
    )
  else
    for i = 0 to nb_iter/2 do
      Kirc.run step (x1,x2,h,w) (block,grid) 0 dev;
      Kirc.run step (x2,x1,h,w) (block,grid) 0 dev;
    done;
  
  Devices.flush dev ();;

  (*if not bench then
    ignore (Graphics.read_key ());*)

let cpu_compute x1 x2 h w nb_iter = 
  let compute t1 t2 = 
    let cpt = ref 0 in
    for y = 0 to h - 1 do
      for x = 0 to w - 1 do
        let i = y * w + x in
        begin
          let left = max 0 (x - 1) in
          let right = min (w-1) (x+1) in
          let top = max 0 (y-1) in
          let bottom = min (h-1) (y +1) in
          cpt := 0;
          for h_ = top to bottom do
            for w_ = left to right do
              match x1.(h_ * w + w_) with
              | Alive -> cpt := !cpt + 1
              | Dead -> ()
            done;
          done;
          x2.(i) <-
            match x1.(i) with
            | Dead -> if !cpt = 3 then Alive else Dead
            | Alive -> if !cpt < 3 || !cpt > 4 then Dead else Alive
        end
      done;
    done
  in
  for i = 0 to nb_iter/2 do
    compute x1 x2;
    compute x2 x1;
  done

  let _ = 
    Random.self_init ();
    let devid = ref 0
    and w = ref 512
    and h = ref 512 
    and nb_iter = ref 10_000 
    and bench = ref false 
    and cpu = ref false
    in     
    let parse_args () = 
      Arg.parse [
        ("-device" , Arg.Int (fun i  -> devid := i), "number of the device [0]");
        ("-height" , Arg.Int (fun i  -> h := i), "field height [1024]");
        ("-width" , Arg.Int (fun i  ->  w := i), "field width [1024]");
        ("-iter" , Arg.Int (fun i  ->  nb_iter := i), "nb steps [10 000]");
        ("-bench" , Arg.Unit (fun ()  ->  bench := true), "run as benchmark [false]");
        ("-cpu" , Arg.Unit (fun ()  ->  cpu := true), "compare with CPU  [false]")] 
        (fun s -> ()) ""
    in
    parse_args ();
    let x1,x2,x1c, x2c, dev, window = init !devid !h !w !bench in
    let name = dev.Spoc.Devices.general_info.Spoc.Devices.name in
    measure_time (fun () -> gpu_compute x1 x2 dev !h !w !nb_iter !bench window) ("GPU "^name);
    if !cpu then
      measure_time (fun () -> cpu_compute x1c x2c !h !w !nb_iter) "CPU";
