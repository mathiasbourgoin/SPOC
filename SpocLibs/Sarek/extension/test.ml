open Spoc

ktyp k = {
  x : float;
  y : int}

let v1 = Spoc.Vector.create (Spoc.Vector.Ccustom k) 10;;


let _ =
v1.[<0>].x <~ 0.

