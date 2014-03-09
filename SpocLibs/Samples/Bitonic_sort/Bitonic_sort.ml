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

open Kirc


let gpu_bitonic = kern v j k ->
  let open Std in
  let i = thread_idx_x + block_dim_x * block_idx_x in
  let ixj = Math.pow i j in
  let mutable temp = 0. in
  if ixj < i then
    begin
      if (Math.logical_and i k) = 0  then
        (
          if  v.[<i>] >. v.[<ixj>] then
            (temp := v.[<ixj>];
             v.[<ixj>] <- v.[<i>];
             v.[<i>] <- temp)
        )
      else 
      if  v.[<i>] <. v.[<ixj>] then
        (temp := v.[<ixj>];
         v.[<ixj>] <- v.[<i>];
         v.[<i>] <- temp);
    end
  
  

;;





(*let (%) f g x = f (g x) ;;
let id x = x ;;
let ($) f x = f x ;;

let swap (x, y) = if x <= y then x, y else y, x
let unit x = [x]
let join = List.concat
let rec pairup = function
  | [], l | l, [] -> List.map unit l
  | a :: l, b :: r -> [a; b] :: pairup (l, r)
let zip p = join % pairup $ p
let lift2 f = function [x; y] -> f (x, y) | l -> l
let inj2 (x, y) = [x; y]
let exchange p = join % List.map (lift2 $ inj2 % swap) % pairup $ p
let rec unzip = function
  | []           -> [], []
  | [x]          -> [x], []
  | x :: y :: xs ->
    let l, r = unzip xs in
  x :: l, y :: r
let (>>>) f (x, y) = (f x, f y)
let rec bitonic = function
  | []  -> []
  | [x] -> [x]
  | l   -> exchange (bitonic >>> unzip l)
let merge (p, q) = bitonic (List.rev_append p q)

let rec sort = function
  | []  -> []
  | [x] -> [x]
  | l   -> merge (sort >>> unzip l)

let rec bin_sequences n =
  if n = 0 then [[]] else
    let s = bin_sequences (n - 1) in
  List.map (fun l -> 0 :: l) s @ List.map (fun l -> 1 :: l) s

let rec is_ascending = function
  | [] | [_] -> true
  | x :: y :: l -> x <= y && is_ascending (y :: l)
*)

let exchange (v : (float, Bigarray.float32_elt) Spoc.Vector.vector) i j : unit =
  let t : float = v.[<i>] in
  v.[<i>] <- v.[<j>];
  v.[<j>] <- t
;;

let rec sortup v 
    m n : unit = 
  if n <> 1 then
    begin
      sortup v m (n/2);
      sortdown v (m+n/2) (n/2);
      mergeup v m (n/2); 
    end

and sortdown v 
    m n : unit =
  if n <> 1 then
    begin
      sortup v m (n/2);
      sortdown v (m+n/2) (n/2);
      mergedown v m (n/2);
    end

and mergeup v 
    (m:int) (n:int) : unit =
  if n <> 0 then
    begin
      for i = 0 to n - 1 do
        if v.[<m+i>] > v.[<m+i+n>] then
          exchange v  (m+i) (m+i+n);
      done;
      mergeup v  m (n/2);
      mergeup v  (m+n) (n/2)
    end

and mergedown v 
    m n =
  if n <> 0 then
    begin
      for i = 0 to n - 1 do
        if v.[<m+i>] < v.[<m+i+n>] then
          exchange v (m+i) (m+i+n);
      done;
      mergedown v m (n/2);
      mergedown v (m+n) (n/2)
    end
;;

  

let cpt = ref 0

let tot_time = ref 0.

let measure_time f =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "time %d : %Fs\n%!" !cpt (t1 -. t0);
  tot_time := !tot_time +.  (t1 -. t0);
  incr cpt;
  a;;
  


let () = 
  let _ = Spoc.Devices.init () in
  let size = try int_of_string Sys.argv.(1) with _ -> 1024 in

  let seq_vect  = Spoc.Vector.create Vector.float32 size
      
  and gpu_vect = Spoc.Vector.create Vector.float32 size
  in
  Random.self_init ();
  (* fill vectors with randmo values... *)
  for i = 0 to Vector.length seq_vect - 1 do
    let v = Random.float 255. in
    seq_vect.[<i>] <- v;
    gpu_vect.[<i>] <- v;
  done;
  

  measure_time (fun () -> sortup seq_vect 0 (Vector.length seq_vect));
;;
