open Spoc

ktyp k = {
  x : float;
  y : int}



let test_elt = 
  {
    x = 0.;
    y = 1
  }
;;

let v1 = Spoc.Vector.create (Spoc.Vector.Ccustom custom_k) 10;;

(*let _ =
v1.[<0>].x <~ 0.
*)
