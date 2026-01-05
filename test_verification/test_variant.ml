open Spoc_core
open Sarek

type float32 = float

(* Define a variant with multiple arguments *)
type complex_variant =
  | Point of float32 * float32
  | Circle of float32 * float32 * float32
  | Rect of float32 * float32 * float32 * float32
[@@sarek.type]

let test_multi_arg_variant =
  [%kernel
    fun (out : float32 vector) ->
      let p = Point ((1.0 : float32), (2.0 : float32)) in
      let c = Circle ((3.0 : float32), (4.0 : float32), (5.0 : float32)) in
      let r =
        Rect ((6.0 : float32), (7.0 : float32), (8.0 : float32), (9.0 : float32))
      in

      (* Simple check to ensure values are preserved *)
      (match p with
      | Point (x, y) ->
          if x = (1.0 : float32) && y = (2.0 : float32) then
            Vector.set out 0 (1.0 : float32)
          else Vector.set out 0 (0.0 : float32)
      | _ -> Vector.set out 0 (-1.0 : float32)) ;

      (match c with
      | Circle (x, y, r) ->
          if x = (3.0 : float32) && y = (4.0 : float32) && r = (5.0 : float32)
          then Vector.set out 1 (1.0 : float32)
          else Vector.set out 1 (0.0 : float32)
      | _ -> Vector.set out 1 (-1.0 : float32)) ;

      match r with
      | Rect (x, y, w, h) ->
          if
            x = (6.0 : float32)
            && y = (7.0 : float32)
            && w = (8.0 : float32)
            && h = (9.0 : float32)
          then Vector.set out 2 (1.0 : float32)
          else Vector.set out 2 (0.0 : float32)
      | _ -> Vector.set out 2 (-1.0 : float32)]

let () =
  let dev = Kirc.Devices.init ~kind:Kirc.Devices.Vulkan () |> List.hd in
  let out = Vector.create Bigarray.float32 3 in

  Kirc.gen ~target:dev test_multi_arg_variant ;
  test_multi_arg_variant#run ~block:(1, 1, 1) ~grid:(1, 1, 1) (out, 3) ;

  Printf.printf "Point check: %f\n" (Vector.get out 0) ;
  Printf.printf "Circle check: %f\n" (Vector.get out 1) ;
  Printf.printf "Rect check: %f\n" (Vector.get out 2)
