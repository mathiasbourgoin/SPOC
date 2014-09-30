open Spoc

ktype t1 = X | Y of int

ktype t2 = 
{
  x : t1;
  y : int;
}

(*
ktype t3 = 
 A
| B of int
| C of t2

  ;;

*)

let (x : ((t1,t1) Vector.vector)) = Vector.create (Custom customT1) 1024
let (y : ((t2,t2) Vector.vector)) = Vector.create (Custom customT2) 1024

let _ =
  let devs = Spoc.Devices.init () in
  for i = 0 to 1023 do
    let t = (if i mod 2 = 0 then X else Y i) in
    Mem.set x i t;
    Mem.set y i {x = t; y = i}; 
  done;
  Mem.to_device x devs.(1);
  Devices.flush devs.(1) ();
  for i = 0 to 1024 do
    Printf.printf "%d \n%!" i;
    let t = Mem.get x i in
    begin
      match t with
      | X -> print_endline "X";
      | Y i -> print_endline ("Y of "^(string_of_int i))
    end;
    let t = Mem.get y i in
    print_endline (string_of_int t.y);
  done;
  

  
