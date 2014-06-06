let test = kern v ->
 ( v + 2 ) mod v 

(*let filter = kern v ->  let open Std in
  let tid = thread_idx_x + block_dim_x * block_idx_x in
  let tab = make_shared 32 in
  tab.(0) <- v.[<0>];
  if tid <= (512*512) then (
    let i = (tid*4) in
    let res = (int_of_float ((0.21 *. (float (v.[<i>]))) +.
                             (0.71 *. (float (v.[<i+1>]))) +.
                             (0.07 *. (float (v.[<i+2>]))) )) + 3 in
    v.[<i>] <- res;
    v.[<i+1>] <- res;
    v.[<i+2>] <- res )
  



let demo = kern a b c n -> 
    let open Std in
    let tab = make_shared 8 in
    tab.(0) <- a.[<i>];
    let i = global_thread_id in
    if i < n then
        c.[<i>] <- a.[<i>] + b.[<i>]*)
