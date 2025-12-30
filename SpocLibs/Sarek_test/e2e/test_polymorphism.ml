(******************************************************************************
 * Sarek PPX - End-to-End Test for Module Functions and Polymorphism
 *
 * Tests:
 * 1. Basic module functions with concrete types
 * 2. Polymorphic module functions used at multiple types
 ******************************************************************************)

open Spoc
open Sarek

(** Test 1: Basic module function with concrete type *)
let test_basic_module_fun () =
  let test_kernel =
    [%kernel
      let add_one (x : int32) : int32 = x + 1l in
      fun (src : int32 vector) (dst : int32 vector) ->
        let open Std in
        let idx = global_idx_x in
        dst.(idx) <- add_one src.(idx)]
  in
  let _, kirc = test_kernel in
  print_endline "=== Basic module function ===" ;
  Kirc.print_ast kirc.Kirc.body ;
  print_endline "==============================" ;

  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No devices - skipping runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "Testing on: %s\n%!" dev.Devices.general_info.Devices.name ;

    let n = 1024 in
    let src = Vector.create Vector.int32 n in
    let dst = Vector.create Vector.int32 n in

    for i = 0 to n - 1 do
      Mem.set src i (Int32.of_int i)
    done ;

    let block = {Kernel.blockX = 256; blockY = 1; blockZ = 1} in
    let grid = {Kernel.gridX = n / 256; gridY = 1; gridZ = 1} in
    Kirc.run test_kernel (src, dst) (block, grid) 0 dev ;
    Mem.to_cpu dst () ;
    Spoc.Devices.flush dev () ;

    let ok = ref true in
    for i = 0 to n - 1 do
      let expected = Int32.of_int (i + 1) in
      if Mem.get dst i <> expected then (
        Printf.printf
          "FAIL: dst[%d] = %ld, expected %ld\n"
          i
          (Mem.get dst i)
          expected ;
        ok := false)
    done ;
    if !ok then print_endline "PASS: Basic module function" ;
    !ok

(** Test 2: Multiple module functions - times_two *)
let test_multiple_module_funs () =
  let test_kernel =
    [%kernel
      let times_two (x : int32) : int32 = x + x in
      fun (src : int32 vector) (dst : int32 vector) ->
        let open Std in
        let idx = global_idx_x in
        dst.(idx) <- times_two src.(idx)]
  in
  let _, kirc = test_kernel in
  print_endline "=== Multiple module functions (times_two) ===" ;
  Kirc.print_ast kirc.Kirc.body ;
  print_endline "==============================================" ;

  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No devices - skipping runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "Testing on: %s\n%!" dev.Devices.general_info.Devices.name ;

    let n = 1024 in
    let src = Vector.create Vector.int32 n in
    let dst = Vector.create Vector.int32 n in

    for i = 0 to n - 1 do
      Mem.set src i (Int32.of_int i)
    done ;

    let block = {Kernel.blockX = 256; blockY = 1; blockZ = 1} in
    let grid = {Kernel.gridX = n / 256; gridY = 1; gridZ = 1} in
    Kirc.run test_kernel (src, dst) (block, grid) 0 dev ;
    Mem.to_cpu dst () ;
    Spoc.Devices.flush dev () ;

    let ok = ref true in
    for i = 0 to n - 1 do
      (* double(i) = 2*i *)
      let expected = Int32.of_int (2 * i) in
      if Mem.get dst i <> expected then (
        Printf.printf
          "FAIL: dst[%d] = %ld, expected %ld\n"
          i
          (Mem.get dst i)
          expected ;
        ok := false)
    done ;
    if !ok then print_endline "PASS: Multiple module functions (times_two)" ;
    !ok

(** Test 3: Polymorphic identity function used at different types *)
let test_polymorphic_identity () =
  let test_kernel =
    [%kernel
      let identity (x : 'a) : 'a = x in
      fun (src_i : int32 vector)
          (src_f : float32 vector)
          (dst_i : int32 vector)
          (dst_f : float32 vector)
        ->
        let open Std in
        let idx = global_idx_x in
        (* Use identity at int32 *)
        dst_i.(idx) <- identity src_i.(idx) ;
        (* Use identity at float32 *)
        dst_f.(idx) <- identity src_f.(idx)]
  in
  let _, kirc = test_kernel in
  print_endline "=== Polymorphic identity ===" ;
  Kirc.print_ast kirc.Kirc.body ;
  print_endline "=============================" ;

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
    if !ok then print_endline "PASS: Polymorphic identity" ;
    !ok

let () =
  print_endline "=== Module Functions & Polymorphism Tests ===" ;
  print_endline "" ;

  let t1 = test_basic_module_fun () in
  print_endline "" ;

  let t2 = test_multiple_module_funs () in
  print_endline "" ;

  let t3 = test_polymorphic_identity () in
  print_endline "" ;

  print_endline "=== Summary ===" ;
  Printf.printf "Basic module function: %s\n" (if t1 then "PASS" else "FAIL") ;
  Printf.printf
    "Multiple module functions: %s\n"
    (if t2 then "PASS" else "FAIL") ;
  Printf.printf "Polymorphic identity: %s\n" (if t3 then "PASS" else "FAIL") ;

  if t1 && t2 && t3 then (
    print_endline "\nAll tests passed!" ;
    exit 0)
  else (
    print_endline "\nSome tests failed!" ;
    exit 1)
