open Spoc
open Vector 

let devices = Spoc.Devices.init ~only:Devices.OpenCL ()
let dev = devices.(1)
       
let double v1 v2  =
  Kirc.map2 ~dev:dev (kern a b -> a + b) v1 v2

let v1 = Vector.create Vector.int32 1024
let v2 = Vector.create Vector.int32 1024



let _ = 
  let v2 =
    for i = 0 to 1023 do
      Mem.set v1 i (Int32.of_int i);
      Mem.set v2 i (Int32.of_int (1023 -i));
    done;
    double v1 v2
  in
  for i = 0 to 100 do
    Printf.printf "%ld\n" (Mem.get v2 i);
  done
