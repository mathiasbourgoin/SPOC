let filter = kern v ->
  let open Std in
  let tab = Std.make_shared 32 in
  tab.(0) <- 1;
  let tid = thread_idx_x + block_dim_x * block_idx_x in
  if tid <= (512*512) then (
    let i = (tid*4) in
    let res = int_of_float ((0.21 *. (float (v.[<i>]))) +.
                           (0.71 *. (float (v.[<i+1>]))) +.
                           (0.07 *. (float (v.[<i+2>]))) ) in
                           
    v.[<i>] <- res;
    v.[<i+1>] <- res;
    v.[<i+2>] <- res )

(*let gpu_bitonic = kern v j k ->
  let open Std in
  let i = thread_idx_x + block_dim_x * block_idx_x in
  let ixj = Math.xor i j in
  let mutable temp = 0. in
  if ixj < i then
    () 
  else
    begin
      if (Math.logical_and i k) = 0  then
        (
          if  v.[<i>] >. v.[<ixj>] then
            (temp := v.[<ixj>];
             v.[<ixj>] <- v.[<i>];
             v.[<i>] <- temp)
        )
      else 
      if v.[<i>] <. v.[<ixj>] then
        (temp := v.[<ixj>];
         v.[<ixj>] <- v.[<i>];
         v.[<i>] <- temp);
    end

    *)
(*let test = kern v n ->
  let open Std in
  let tid = thread_idx_x + block_dim_x * block_idx_x in
  let tab = make_shared 32 in
  tab.(0) <- v.[<0>];
  v.[<0>] <- tab.(0);
  if tid <= (512*512) then (
    let i = (tid*4) in
    let res = (int_of_float ((0.21 *. (float (v.[<i>]))) +.
                             (0.71 *. (float (v.[<i+1>]))) +.
                             (0.07 *. (float (v.[<i+2>]))) )) + 3 in
    v.[<i>] <- res;
    v.[<i+1>] <- res;
    v.[<i+2>] <- res 
  )
*)

(*( v +. 2. ) *. v
  else
    ( v +. 2. ) *. v
*)
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
