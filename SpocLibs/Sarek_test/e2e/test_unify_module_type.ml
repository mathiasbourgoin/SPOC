(* Test: type unification with module-qualified types *)
open Spoc
open Sarek_geometry

let () =
  let kernel =
    [%kernel
      fun (points : Geometry_lib.point vector)
          (out : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then
          let p = points.(tid) in
          (* Create a new point with module-qualified type annotation *)
          let q : Geometry_lib.point = {x = p.x +. 1.0; y = p.y +. 1.0} in
          out.(tid) <- q.x]
  in
  let _, kirc = kernel in
  Sarek.Kirc.print_ast kirc.Sarek.Kirc.body ;
  print_endline "Module type unification PASSED"
