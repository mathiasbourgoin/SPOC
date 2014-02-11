open Spoc
open Kirc


let _ = 
  let dev = Spoc.Devices.init () in
  let v1 = Vector.create Vector.int32 1024
  and v2 = Vector.create Vector.int32 1024 in 
  for i = 0 to Vector.length v1 - 1 do
    Mem.unsafe_set v1 i (Int32.of_int (Random.int 255));
    Mem.unsafe_set v2 i (Int32.of_int (Random.int 255));
  done;
	let res = ref v1 in
	for i = 0 to 10 - 1 do
 		res := map2 (kern a b -> a * b) ~dev:dev.(0) v1 v2;
	Mem.to_cpu !res ();
	Devices.flush dev.(0)
	done;

  for i = 0 to 10 do
    Printf.printf "input : %ld - %ld, output : %ld, expecting %ld \n" (Mem.get v1 i) (Mem.get v2 i) (Mem.get !res i) (Int32.mul (Mem.get v1 i) (Mem.get v2 i));
  done;
  res;;


