(*
         DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
                    Version 2, December 2004 

 Copyright (C) 2004 Sam Hocevar <sam@hocevar.net> 

 Everyone is permitted to copy and distribute verbatim or modified 
 copies of this license document, and changing it is allowed as long 
 as the name is changed. 

            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION 

  0. You just DO WHAT THE FUCK YOU WANT TO.
*)	
open Spoc
open Kernel

kernel k_init   : Spoc.Vector.vfloat64 -> Spoc.Vector.vfloat64 -> Spoc.Vector.vfloat64 -> int -> int  -> unit= "kernels/Puissance" "init"
kernel k_divide : Spoc.Vector.vfloat64 -> Spoc.Vector.vfloat64 -> int -> unit = "kernels/Puissance" "divide"
kernel k_norme  : Spoc.Vector.vfloat64 -> Spoc.Vector.vfloat64 -> Spoc.Vector.vfloat64 -> int -> unit = "kernels/Puissance" "norme"

let size = ref 1024
let eps = 0.01

let cpt = ref 0

let tot_time = ref 0.

let measure_time f =
  let t0 = Unix.gettimeofday () in
  let a = f () in
	let t1 = Unix.gettimeofday () in
  Printf.printf "temps %d : %F\n%!" !cpt (t1 -. t0);
	tot_time := !tot_time +.  (t1 -. t0);
	incr cpt;
  a;;

let max_iter = 100

let main a v n () = 
  let iter = ref 0 in
  
  let dev = Devices.init () 
  and vn = Vector.create Vector.float64 n
  and max = (Vector.create Vector.float64 1)
  and max2 = (Vector.create Vector.float64 1)
    
  and v_norme = Vector.create Vector.float64 n
  and norme = ref 1.; 
  in
  while (!norme > eps && !iter < max_iter) do
    incr iter;
    max.[<0>] <- 0.;
    ignore(let open Compose in 
	   (Compose.run  
	      (pipe 
		 (map k_init vn (a, v, vn, n, n))
		 (reduce spoc_max max (vn, max,n))) dev.(0) a));
    norme := 0.;
    
    let open Compose in
    norme := 	(Compose.run 
		   (pipe 
		      (pipe 
			 (map k_divide vn (vn, max, n))
			 (map k_norme v_norme (vn, v, v_norme, n))) 
		      (reduce spoc_max max2 (v_norme, max2, n))) dev.(0) vn ).[<0>]; 
  done;
  
  max.[<0>], !iter
    
let seq_iterate a v n () =
  let iter = ref 0
  and vn = Array.create n 0.
  and norme = ref 1. in 
  let s = ref 0. in
  let max = ref 0. in
  while !norme > eps && !iter < max_iter do
    incr iter;
    
    for i = 0 to n - 1 do
      s := 0.;
      for j = 0 to n - 1 do
	s:= !s +. a.(i * n + j) *. v.(j)
      done;
      vn.(i) <- !s
    done;
    
    
    max := 0.;
    Array.iter (fun a -> if (abs_float a) > !max then max := (abs_float a) ) vn;	
    Array.iteri (fun i a -> vn.(i) <- a /. !max) vn;   
    
    
    norme := 0.;
    for i = 0 to n  - 1 do
      let diff = abs_float(vn.(i) -. v.(i)) in
      (if !norme < diff then
	  norme := diff
      );
      v.(i) <- vn.(i)	
	
    done;
    
  done;
  !max, !iter
    
let external_iterate a v n () =
	let iter = ref 0
	and vn = Array.create n 0.
	and norme = ref 1. in 
	let s = ref 0. in
	let max = ref 0. in
	while !norme > eps && !iter < max_iter do
	  incr iter;
	  
	  for i = 0 to n - 1 do
	    s := 0.;
	    for j = 0 to n - 1 do
	      s:= !s +. a.(i * n + j) *. v.(j)
	    done;
	    vn.(i) <- !s
	  done;
	  
	  
	  max := 0.;
	  Array.iter (fun a -> if (abs_float a) > !max then max := (abs_float a) ) vn;	
	  Array.iteri (fun i a -> vn.(i) <- a /. !max) vn;   
	  
	  
	  norme := 0.;
	  for i = 0 to n  - 1 do
	    let diff = abs_float(vn.(i) -. v.(i)) in
	    (if !norme < diff then
		norme := diff
	    );
	    v.(i) <- vn.(i)	
	      
	  done;
	  
	done;
	!max, !iter
	  
	  
let _ = 
  Random.self_init ();
  let a = Vector.create Vector.float64 (!size * !size)
  and v = Vector.create Vector.float64 !size in
  
  let seq_a = Array.create (!size * !size) 0.
  and seq_v = Array.create (!size) 0. in
  
  for i = 0 to !size - 1 do
    for j = 0 to !size  - 1 do
      let r =  (Random.float 100.) -. 50. in
      a.[<i * !size + j>] <- r;
      seq_a.(i * !size + j) <- r;
    done;
  done;	    
  for k = 0 to !size - 1 do 
    v.[<k>] <- 0.;
    seq_v.(k) <- 0.;
  done;
  v.[<0>] <- 1.;
  seq_v.(0) <-1.;
  Printf.printf "Starting\n"; flush stdout;
  let max,iter = (measure_time (main a v !size)) in
  Printf.printf "GPU MAX = %g in %i iterations\n" max iter;
  flush stdout;
  let max,iter = (measure_time (seq_iterate seq_a seq_v !size)) in
  Printf.printf "CPU MAX = %g in %i iterations\n" max iter;
  
