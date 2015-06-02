open Spoc
open Vector 

let gpu_gray = kern v width height ->
    let gray = fun r g b ->
    let open Std in
      int_of_float
       ((0.21 *. (float r)) +. 
             (0.71 *. (float g)) +. 
             (0.07 *. (float b)) ) 
    in
    let tid = Std.global_thread_id in 
    if tid <= (width*height) then (
    let i = (tid*4) in
    let gr =  gray v.[<i>] v.[<i+1>] v.[<i+2>] in
    v.[<i>] <- gr; 
    v.[<i+1>] <-gr; 
    v.[<i+2>] <-  gr)