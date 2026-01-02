(******************************************************************************
 * Sarek PPX - Test for Polymorphic Module Functions
 *
 * Tests that [@sarek.module] functions can be polymorphic and used at
 * multiple types within kernels.
 * Supports both SPOC and V2 runtime paths.
 ******************************************************************************)

(* Module aliases to avoid conflicts *)
module Spoc_Vector = Spoc.Vector
module Spoc_Devices = Spoc.Devices
module Spoc_Mem = Spoc.Mem
module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer

open Sarek

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_opencl.Opencl_plugin_v2.init () ;
  Sarek_native.Native_plugin.init () ;
  Sarek_native.Native_plugin_v2.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin_v2.init ()

(* Define a polymorphic identity function in module scope *)
let[@sarek.module] identity (x : 'a) : 'a = x

let test_kernel =
  [%kernel
    fun (src_i : int32 vector)
        (src_f : float32 vector)
        (dst_i : int32 vector)
        (dst_f : float32 vector) ->
      let open Std in
      let idx = global_idx_x in
      (* Use identity at int32 *)
      dst_i.(idx) <- identity src_i.(idx) ;
      (* Use identity at float32 *)
      dst_f.(idx) <- identity src_f.(idx)]

(* === SPOC Path === *)

let test_poly_identity_spoc () =
  let _, kirc = test_kernel in
  print_endline "=== SPOC: Polymorphic module identity ===" ;
  Kirc.print_ast kirc.Kirc.body ;
  print_endline "==========================================" ;

  let devs = Spoc_Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No SPOC devices - skipping runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf
      "Testing on: %s\n%!"
      dev.Spoc_Devices.general_info.Spoc_Devices.name ;

    let n = 1024 in
    let src_i = Spoc_Vector.create Spoc_Vector.int32 n in
    let src_f = Spoc_Vector.create Spoc_Vector.float32 n in
    let dst_i = Spoc_Vector.create Spoc_Vector.int32 n in
    let dst_f = Spoc_Vector.create Spoc_Vector.float32 n in

    for i = 0 to n - 1 do
      Spoc_Mem.set src_i i (Int32.of_int i) ;
      Spoc_Mem.set src_f i (float_of_int i)
    done ;

    let block = {Spoc.Kernel.blockX = 256; blockY = 1; blockZ = 1} in
    let grid = {Spoc.Kernel.gridX = n / 256; gridY = 1; gridZ = 1} in
    Kirc.run test_kernel (src_i, src_f, dst_i, dst_f) (block, grid) 0 dev ;
    Spoc_Mem.to_cpu dst_i () ;
    Spoc_Mem.to_cpu dst_f () ;
    Spoc_Devices.flush dev () ;

    let ok = ref true in
    for i = 0 to n - 1 do
      let expected_i = Int32.of_int i in
      let expected_f = float_of_int i in
      if Spoc_Mem.get dst_i i <> expected_i then (
        Printf.printf
          "FAIL: dst_i[%d] = %ld, expected %ld\n"
          i
          (Spoc_Mem.get dst_i i)
          expected_i ;
        ok := false) ;
      if abs_float (Spoc_Mem.get dst_f i -. expected_f) > 0.001 then (
        Printf.printf
          "FAIL: dst_f[%d] = %f, expected %f\n"
          i
          (Spoc_Mem.get dst_f i)
          expected_f ;
        ok := false)
    done ;
    if !ok then print_endline "PASS: SPOC Polymorphic module identity" ;
    !ok

(* === V2 Path === *)

let test_poly_identity_v2 () =
  print_endline "=== V2: Polymorphic module identity ===" ;

  let devs = V2_Device.all () in
  if Array.length devs = 0 then (
    print_endline "No V2 devices - skipping runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "Testing on: %s\n%!" dev.V2_Device.name ;

    let _, kirc = test_kernel in
    let ir =
      match kirc.Kirc.body_v2 with
      | Some ir -> ir
      | None -> failwith "Kernel has no V2 IR"
    in

    let n = 1024 in
    let src_i = V2_Vector.create V2_Vector.int32 n in
    let src_f = V2_Vector.create V2_Vector.float32 n in
    let dst_i = V2_Vector.create V2_Vector.int32 n in
    let dst_f = V2_Vector.create V2_Vector.float32 n in

    for i = 0 to n - 1 do
      V2_Vector.set src_i i (Int32.of_int i) ;
      V2_Vector.set src_f i (float_of_int i)
    done ;

    let block = Execute.dims1d 256 in
    let grid = Execute.dims1d (n / 256) in
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        [
          Execute.Vec src_i;
          Execute.Vec src_f;
          Execute.Vec dst_i;
          Execute.Vec dst_f;
        ]
      ~block
      ~grid
      () ;
    V2_Transfer.flush dev ;

    let ok = ref true in
    for i = 0 to n - 1 do
      let expected_i = Int32.of_int i in
      let expected_f = float_of_int i in
      let got_i = V2_Vector.get dst_i i in
      let got_f = V2_Vector.get dst_f i in
      if got_i <> expected_i then (
        Printf.printf "FAIL: dst_i[%d] = %ld, expected %ld\n" i got_i expected_i ;
        ok := false) ;
      if abs_float (got_f -. expected_f) > 0.001 then (
        Printf.printf "FAIL: dst_f[%d] = %f, expected %f\n" i got_f expected_f ;
        ok := false)
    done ;
    if !ok then print_endline "PASS: V2 Polymorphic module identity" ;
    !ok

let () =
  print_endline "=== Polymorphic Module Function Tests ===" ;
  print_endline "" ;

  let t1_spoc = test_poly_identity_spoc () in
  print_endline "" ;

  let t1_v2 = test_poly_identity_v2 () in
  print_endline "" ;

  print_endline "=== Summary ===" ;
  Printf.printf
    "Polymorphic module identity (SPOC): %s\n"
    (if t1_spoc then "PASS" else "FAIL") ;
  Printf.printf
    "Polymorphic module identity (V2): %s\n"
    (if t1_v2 then "PASS" else "FAIL") ;

  if t1_spoc && t1_v2 then (
    print_endline "\nAll tests passed!" ;
    exit 0)
  else (
    print_endline "\nSome tests failed!" ;
    exit 1)
