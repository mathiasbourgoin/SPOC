(******************************************************************************
 * Mandelbrot Sarek (PPX) - headless PPM output
 ******************************************************************************)

open Spoc

let make_mandelbrot () =
  [%kernel fun (img : int32 vector)
                (width : int)
                (height : int)
                (max_iter : int)
                (shiftx : int)
                (shifty : int)
                (zoom : float32) ->
    let x = thread_idx_x + block_idx_x * block_dim_x in
    let y = thread_idx_y + block_idx_y * block_dim_y in
    if (x < width) && (y < height) then begin
      let x0 = x + shiftx in
      let y0 = y + shifty in
      let a = 4.0 *. (float x0 /. float width) /. zoom -. 2.0 in
      let b = 4.0 *. (float y0 /. float height) /. zoom -. 2.0 in
      let cpt = mut 0l in
      let x1 = mut 0.0 in
      let y1 = mut 0.0 in
      let norm = mut 0.0 in
      while (cpt < max_iter) && (norm <= 4.0) do
        cpt := cpt + 1l;
        let x2 = x1 *. x1 -. y1 *. y1 +. a in
        let y2 = 2.0 *. x1 *. y1 +. b in
        x1 := x2;
        y1 := y2;
        norm := x1 *. x1 +. y1 *. y1
      done;
      img.(y * width + x) <- cpt
    end
  ]

let color_rgb n max_iter =
  if n = max_iter then
    (0, 0, 0)
  else
    let f v =
      let i = float_of_int v in
      int_of_float (255. *. (0.5 +. 0.5 *. sin (i *. 0.1)))
    in
    (f (n + 16), f (n + 32), f n)

let write_ppm filename width height data max_iter =
  let oc = open_out_bin filename in
  Printf.fprintf oc "P6\n%d %d\n255\n" width height;
  for y = 0 to pred height do
    for x = 0 to pred width do
      let idx = (y * width) + x in
      let v = Int32.to_int (Spoc.Mem.unsafe_get data idx) in
      let r, g, b = color_rgb v max_iter in
      output_byte oc r;
      output_byte oc g;
      output_byte oc b;
    done;
  done;
  close_out oc

let open_ppm filename =
  let cmd = Printf.sprintf "xdg-open %s" (Filename.quote filename) in
  match Sys.command cmd with
  | 0 -> ()
  | rc ->
    Printf.eprintf "xdg-open failed (%d): %s\n%!" rc filename

let () =
  let output_file = ref "/tmp/mandelbrot.ppm" in
  let width = ref 800 in
  let height = ref 800 in
  let max_iter = ref 512 in
  let shiftx = ref 0 in
  let shifty = ref 0 in
  let zoom = ref 1.0 in
  let dev_id = ref 0 in
  let open_file = ref false in
  let dump_opencl = ref None in
  let args = [
    ("-ppm", Arg.String (fun s -> output_file := s), "output PPM file");
    ("-open", Arg.Unit (fun () -> open_file := true), "open output PPM with xdg-open");
    ("-dump-opencl", Arg.String (fun s -> dump_opencl := Some s), "write OpenCL kernel to file");
    ("-width", Arg.Int (fun v -> width := v), "image width [800]");
    ("-height", Arg.Int (fun v -> height := v), "image height [800]");
    ("-max_iter", Arg.Int (fun v -> max_iter := v), "max iterations [512]");
    ("-shiftx", Arg.Int (fun v -> shiftx := v), "x shift [0]");
    ("-shifty", Arg.Int (fun v -> shifty := v), "y shift [0]");
    ("-zoom", Arg.Float (fun v -> zoom := v), "zoom [1.0]");
    ("-device", Arg.Int (fun v -> dev_id := v), "device id [0]");
  ] in
  Arg.parse args (fun _ -> ()) "";

  let devs = Devices.init () in
  if Array.length devs = 0 then begin
    Printf.eprintf "No GPU devices found\n%!";
    exit 1
  end;
  let dev = devs.(!dev_id) in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name;

  let w = !width in
  let h = !height in
  let total = w * h in
  let img = Spoc.Vector.create Spoc.Vector.int32 ~dev total in

  let threads =
    match dev.Devices.specific_info with
    | Devices.OpenCLInfo clI ->
      (match clI.Devices.device_type with
       | Devices.CL_DEVICE_TYPE_CPU -> 1
       | _ -> 16)
    | _ -> 16
  in
  let grid_x = (w + threads - 1) / threads in
  let grid_y = (h + threads - 1) / threads in
  let block = { Spoc.Kernel.blockX = threads; Spoc.Kernel.blockY = threads; Spoc.Kernel.blockZ = 1 } in
  let grid = { Spoc.Kernel.gridX = grid_x; Spoc.Kernel.gridY = grid_y; Spoc.Kernel.gridZ = 1 } in

  let mandelbrot = make_mandelbrot () in
  (match !dump_opencl with
   | None -> ()
   | Some path ->
     let src = Sarek.Kirc.opencl_source mandelbrot dev in
     let oc = open_out path in
     output_string oc src;
     close_out oc);
  let mandelbrot = Sarek.Kirc.gen mandelbrot dev in
  Sarek.Kirc.run mandelbrot
    (img,
     w,
     h,
     !max_iter,
     !shiftx,
     !shifty,
     !zoom)
    (block, grid) 0 dev;

  Spoc.Mem.to_cpu img ();
  Spoc.Devices.flush dev ();
  write_ppm !output_file w h img !max_iter;
  if !open_file then open_ppm !output_file;
  Printf.printf "Wrote %s\n%!" !output_file
