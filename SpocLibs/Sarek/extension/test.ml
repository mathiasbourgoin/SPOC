open Spoc
open Vector 

(* not very original... *)
let mandelbrot = kern img  -> 
  let open Std in
  (* 2D kernel *)
  let y = thread_idx_y + (block_idx_y * block_dim_y) in
  let x = thread_idx_x + (block_idx_x * block_dim_x) in
  if (y < @height) && (x < @width) then 
     begin  
       let x0 = x  in
       let y0 = y  in
       let mutable cpt = 0 in 
       let mutable x1 = 0. in
       let mutable y1 = 0. in
       let mutable x2 = 0. in
       let mutable y2 = 0. in
       let a = 4. *. ((float x0) /. (float @width))   -. 2. in
       let b = 4. *. ((float y0) /. (float @height)) -. 2. in
       let normal = fun x y ->
           x *. x +. y *. y in
       let mutable norm = 0. 
       in
       while ((cpt < @max_iter) && (norm <=. 4.)) do
         cpt := (cpt + 1);
         x2 := (x1 *. x1) -. (y1 *. y1) +. a;
         y2 :=  (2. *. x1 *. y1 ) +. b;
         x1 := x2;
         y1 := y2;
         norm := normal x1 y1;
       done;
       img.[<y * @width + x>] <- cpt
     end