(******************************************************************************
 * Sarek PPX - Test for Polymorphic Module Functions
 *
 * Tests that [@sarek.module] functions can be polymorphic and used at
 * multiple types within kernels.
 ******************************************************************************)

open Spoc
open Sarek

(* Define a polymorphic identity function in module scope *)
let[@sarek.module] identity (x : 'a) : 'a = x

let test_poly_identity () =
  let test_kernel =
    [%kernel
      fun (src_i : int32 vector) (src_f : float32 vector)
          (dst_i : int32 vector) (dst_f : float32 vector) ->
        let open Std in
        let idx = global_idx_x in
        (* Use identity at int32 *)
        dst_i.(idx) <- identity src_i.(idx) ;
        (* Use identity at float32 *)
        dst_f.(idx) <- identity src_f.(idx)]
  in
  let _, kirc = test_kernel in
  print_endline "=== Polymorphic module identity ===" ;
  Kirc.print_ast kirc.Kirc.body ;
  print_endline "====================================" ;

  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No devices - skipping runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "Testing on: %s\n%!" dev.Devices.general_info.Devices.name ;

    let n = 1024 in
    let src_i = Vector.create Vector.int32 n in
    let src_f = Vector.create Vector.float32 n in
    let dst_i = Vector.create Vector.int32 n in
    let dst_f = Vector.create Vector.float32 n in

    for i = 0 to n - 1 do
      Mem.set src_i i (Int32.of_int i) ;
      Mem.set src_f i (float_of_int i)
    done ;

    let block = {Kernel.blockX = 256; blockY = 1; blockZ = 1} in
    let grid = {Kernel.gridX = n / 256; gridY = 1; gridZ = 1} in
    Kirc.run test_kernel (src_i, src_f, dst_i, dst_f) (block, grid) 0 dev ;
    Mem.to_cpu dst_i () ;
    Mem.to_cpu dst_f () ;
    Spoc.Devices.flush dev () ;

    let ok = ref true in
    for i = 0 to n - 1 do
      let expected_i = Int32.of_int i in
      let expected_f = float_of_int i in
      if Mem.get dst_i i <> expected_i then (
        Printf.printf
          "FAIL: dst_i[%d] = %ld, expected %ld\n"
          i
          (Mem.get dst_i i)
          expected_i ;
        ok := false) ;
      if abs_float (Mem.get dst_f i -. expected_f) > 0.001 then (
        Printf.printf
          "FAIL: dst_f[%d] = %f, expected %f\n"
          i
          (Mem.get dst_f i)
          expected_f ;
        ok := false)
    done ;
    if !ok then print_endline "PASS: Polymorphic module identity" ;
    !ok

let () =
  print_endline "=== Polymorphic Module Function Tests ===" ;
  print_endline "" ;

  let t1 = test_poly_identity () in
  print_endline "" ;

  print_endline "=== Summary ===" ;
  Printf.printf "Polymorphic module identity: %s\n" (if t1 then "PASS" else "FAIL") ;

  if t1 then (
    print_endline "\nAll tests passed!" ;
    exit 0)
  else (
    print_endline "\nSome tests failed!" ;
    exit 1)
