  (* img shiftx shifty zoom ->  *)
  (* let open Std in *)

  (* let y = thread_idx_y + (block_idx_y * block_dim_y) in *)
  (* let x = thread_idx_x + (block_idx_x * block_dim_x) in *)
  (* (if (y >= @height) || (x >= @width) then *)
  (*     return () ; *)
  (* ); *)
  (* let x0 = (x + shiftx) in *)
  (* let y0 = (y + shifty) in *)
  (* let mutable cpt = 0 in  *)
  (* let mutable x1 = 0. in *)
  (* let mutable y1 = 0. in *)
  (* let mutable x2 = 0. in *)
  (* let mutable y2 = 0. in *)
  (* let a = 4. *. ((float x0) /. (float @width)) /. zoom  -. 2. in *)
  (* let b = 4. *. ((float y0) /. (float @height)) /. zoom -. 2. in *)

  (* let normalize = fun x y ->  *)
  (*   let pow2 = fun x -> x *. x in *)
  (*   (pow2 x) +. (pow2 y) in   *)

  (* let mutable norm = normalize x1  y1 *)


  (* in *)
  (* while ((cpt < @max_iter) && (norm <=. 4.)) do *)
  (*   cpt := (cpt + 1); *)
  (*   x2 := (x1 *. x1) -. (y1 *. y1) +. a; *)
  (*   y2 :=  (2. *. x1 *. y1 ) +. b; *)
  (*   x1 := x2; *)
  (*   y1 := y2; *)
  (*   norm := (x1 *. x1 ) +. ( y1 *. y1); *)
  (* done; *)
  (* img.[<y * @width + x>] <- cpt *)


open Spoc

let height = ref 0l
let width = ref 0l
let max_iter = ref 0l


let incrv = kern v length ->
  for i = 0 to length do
    if (i mod 2) = 0 then
      v.[<i>] <- v.[<i>] +. 1.
    else
      v.[<i>] <- v.[<i>] -. 1.;
  done



let _ =
  let module IncrV =
    (val (Transform.to_fast_flow  (snd incrv)))
  in
  Printf.printf "%s\n" IncrV.source;
  IncrV.create_accelerator 24;
  IncrV.run_accelerator ();
