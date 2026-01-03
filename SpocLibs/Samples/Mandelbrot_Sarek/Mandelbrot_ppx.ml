(******************************************************************************
 * Mandelbrot Sarek (PPX) - headless PPM output
 * V2 runtime only.
 ******************************************************************************)

module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer
module Std = Sarek_stdlib.Std

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init () ;
  Sarek_native.Native_plugin_v2.init () ;
  Sarek_interpreter.Interpreter_plugin_v2.init ()

let make_mandelbrot () =
  [%kernel
    fun (img : int32 vector)
        (width : int)
        (height : int)
        (max_iter : int)
        (shiftx : int)
        (shifty : int)
        (zoom : float32) ->
      let open Std in
      let x = global_idx_x in
      let y = global_idx_y in
      if x < width && y < height then begin
        let x0 = x + shiftx in
        let y0 = y + shifty in
        let a = (4.0 *. (float x0 /. float width) /. zoom) -. 2.0 in
        let b = (4.0 *. (float y0 /. float height) /. zoom) -. 2.0 in
        let cpt = mut 0l in
        let x1 = mut 0.0 in
        let y1 = mut 0.0 in
        let norm = mut 0.0 in
        while cpt < max_iter && norm <= 4.0 do
          cpt := cpt + 1l ;
          let x2 = (x1 *. x1) -. (y1 *. y1) +. a in
          let y2 = (2.0 *. x1 *. y1) +. b in
          x1 := x2 ;
          y1 := y2 ;
          norm := (x1 *. x1) +. (y1 *. y1)
        done ;
        img.((y * width) + x) <- cpt
      end]

let color_rgb n max_iter =
  if n = max_iter then (0, 0, 0)
  else
    let f v =
      let i = float_of_int v in
      int_of_float (255. *. (0.5 +. (0.5 *. sin (i *. 0.1))))
    in
    (f (n + 16), f (n + 32), f n)

let write_ppm filename width height
    (data : (int32, Bigarray.int32_elt) V2_Vector.t) max_iter =
  let oc = open_out_bin filename in
  Printf.fprintf oc "P6\n%d %d\n255\n" width height ;
  for y = 0 to pred height do
    for x = 0 to pred width do
      let idx = (y * width) + x in
      let v = Int32.to_int (V2_Vector.get data idx) in
      let r, g, b = color_rgb v max_iter in
      output_byte oc r ;
      output_byte oc g ;
      output_byte oc b
    done
  done ;
  close_out oc

let open_ppm filename =
  let cmd = Printf.sprintf "xdg-open %s" (Filename.quote filename) in
  match Sys.command cmd with
  | 0 -> ()
  | rc -> Printf.eprintf "xdg-open failed (%d): %s\n%!" rc filename

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
  let args =
    [
      ("-ppm", Arg.String (fun s -> output_file := s), "output PPM file");
      ( "-open",
        Arg.Unit (fun () -> open_file := true),
        "open output PPM with xdg-open" );
      ("-width", Arg.Int (fun v -> width := v), "image width [800]");
      ("-height", Arg.Int (fun v -> height := v), "image height [800]");
      ("-max_iter", Arg.Int (fun v -> max_iter := v), "max iterations [512]");
      ("-shiftx", Arg.Int (fun v -> shiftx := v), "x shift [0]");
      ("-shifty", Arg.Int (fun v -> shifty := v), "y shift [0]");
      ("-zoom", Arg.Float (fun v -> zoom := v), "zoom [1.0]");
      ("-device", Arg.Int (fun v -> dev_id := v), "device id [0]");
    ]
  in
  Arg.parse args (fun _ -> ()) "" ;

  let devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then begin
    Printf.eprintf "No GPU devices found\n%!" ;
    exit 1
  end ;
  let dev = devs.(!dev_id) in
  Printf.printf
    "Using device: %s (%s)\n%!"
    dev.V2_Device.name
    dev.V2_Device.framework ;

  let w = !width in
  let h = !height in
  let total = w * h in
  let img = V2_Vector.create V2_Vector.int32 total in

  (* Get V2 IR *)
  let _, kirc = make_mandelbrot () in
  let ir =
    match kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  let threads = if dev.V2_Device.capabilities.is_cpu then 1 else 16 in
  let grid_x = (w + threads - 1) / threads in
  let grid_y = (h + threads - 1) / threads in
  let block = Sarek.Execute.dims2d threads threads in
  let grid = Sarek.Execute.dims2d grid_x grid_y in

  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec img;
        Sarek.Execute.Int w;
        Sarek.Execute.Int h;
        Sarek.Execute.Int !max_iter;
        Sarek.Execute.Int !shiftx;
        Sarek.Execute.Int !shifty;
        Sarek.Execute.Float32 !zoom;
      ]
    ~block
    ~grid
    () ;
  V2_Transfer.flush dev ;

  write_ppm !output_file w h img !max_iter ;
  if !open_file then open_ppm !output_file ;
  Printf.printf "Wrote %s\n%!" !output_file
