open Spoc
open Kirc
open Spoc.Vector

let height = ref 0l
let width = ref 0l
let shiftx = ref 0l
let shifty = ref 0l
let zoom = ref 0.
let max_iter = ref 0l

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
  let devs = Spoc.Devices.init ~only:Devices.OpenCL ()
  in
  Kirc.gen ~profile:true ~only:Devices.OpenCL mandelbrot devs.(0);
  Printf.printf "Here\n%!";
  List.iter (Printf.printf "%s\n")((fst mandelbrot)#get_opencl_sources ())
    
