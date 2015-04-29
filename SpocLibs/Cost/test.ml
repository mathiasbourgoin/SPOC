open Spoc
open Cost

let gpu_bitonic = kern v j k ->
  let open Std in
  let i = thread_idx_x + block_dim_x * block_idx_x in
  let ixj = Math.xor i j in
  let mutable temp = 0. in
  if ixj >= i then
    begin
      if (Math.logical_and i k) = 0 then
        (
          if v.[<i>] >. v.[<ixj>] then
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


let _ =
  let open Kirc in
  let kir,k =  gpu_bitonic in
  let kernel_body = k.body in
  Cost.eval_cost kernel_body
  
