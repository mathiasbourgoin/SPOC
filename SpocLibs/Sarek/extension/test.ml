let gpu_to_gray = kern v ->
		  let open Std in
		  let i = thread_idx_x + block_dim_x * block_idx_x in
  if i > (512*512) then
    ()
  else
    (
      let real_i = (i*4) in
      let res = (v.[<i>] + v.[<i+1>] + v.[<i+2>]) / 3 in
      v.[<i>] <- res;
      v.[<i+1>] <- res;
      v.[<i+2>] <- res
    )
