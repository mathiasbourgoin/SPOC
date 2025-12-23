(* Test that convention values are accessible from external module *)
open Sarek_geometry

let () =
  (* These should all be resolved by OCaml's type system *)
  let _ = Geometry_lib.sarek_type_point in
  let _ = Geometry_lib.sarek_get_point_x in
  let _ = Geometry_lib.sarek_get_point_y in
  let _ = Geometry_lib.sarek_fun_distance in

  (* Test that accessors have correct types *)
  let p : Geometry_lib.point = {x = 1.0; y = 2.0} in
  let x : float = Geometry_lib.sarek_get_point_x p in
  let y : float = Geometry_lib.sarek_get_point_y p in
  Printf.printf "Convention test:\n" ;
  Printf.printf "  point.x = %f, point.y = %f\n" x y ;
  Printf.printf
    "  sarek_type_point length = %d\n"
    (String.length Geometry_lib.sarek_type_point) ;
  Printf.printf
    "  sarek_fun_distance length = %d\n"
    (String.length Geometry_lib.sarek_fun_distance) ;
  print_endline "Convention test PASSED"
