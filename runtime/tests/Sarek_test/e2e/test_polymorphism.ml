(******************************************************************************
 * Sarek PPX - End-to-End Test for Module Functions and Polymorphism
 *
 * Tests:
 * 1. Basic module functions with concrete types
 * 2. Polymorphic module functions used at multiple types
 * V2 runtime only.
 ******************************************************************************)

module Std = Sarek_stdlib.Std
open Sarek

(* V2 module aliases *)
module V2_Device = Spoc_core.Device
module V2_Vector = Spoc_core.Vector
module V2_Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init () ;
  Sarek_native.Native_plugin.init () ;
  Sarek_native.Native_plugin_v2.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin_v2.init ()

let basic_kernel =
  [%kernel
    let add_one (x : int32) : int32 = x + 1l in
    fun (src : int32 vector) (dst : int32 vector) ->
      let open Std in
      let idx = global_idx_x in
      dst.(idx) <- add_one src.(idx)]

let times_two_kernel =
  [%kernel
    let times_two (x : int32) : int32 = x + x in
    fun (src : int32 vector) (dst : int32 vector) ->
      let open Std in
      let idx = global_idx_x in
      dst.(idx) <- times_two src.(idx)]

let identity_kernel =
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

(* ========== V2 Tests ========== *)

let test_basic_v2 () =
  let _, kirc = basic_kernel in
  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length v2_devs = 0 then (
    print_endline "V2: No device - SKIPPED" ;
    true)
  else
    let dev = v2_devs.(0) in
    match kirc.Sarek.Kirc_types.body_v2 with
    | None ->
        print_endline "V2: No V2 IR - SKIPPED" ;
        true
    | Some ir ->
        let n = 1024 in
        let src = V2_Vector.create V2_Vector.int32 n in
        let dst = V2_Vector.create V2_Vector.int32 n in
        for i = 0 to n - 1 do
          V2_Vector.set src i (Int32.of_int i)
        done ;

        let block = Execute.dims1d 256 in
        let grid = Execute.dims1d (n / 256) in
        Execute.run_vectors
          ~device:dev
          ~ir
          ~args:[Execute.Vec src; Execute.Vec dst]
          ~block
          ~grid
          () ;
        V2_Transfer.flush dev ;

        let ok = ref true in
        for i = 0 to n - 1 do
          let expected = Int32.of_int (i + 1) in
          if V2_Vector.get dst i <> expected then (
            Printf.printf
              "V2 FAIL: dst[%d] = %ld, expected %ld\n"
              i
              (V2_Vector.get dst i)
              expected ;
            ok := false)
        done ;
        if !ok then print_endline "PASS: Basic module function (V2)" ;
        !ok

let test_times_two_v2 () =
  let _, kirc = times_two_kernel in
  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length v2_devs = 0 then (
    print_endline "V2: No device - SKIPPED" ;
    true)
  else
    let dev = v2_devs.(0) in
    match kirc.Sarek.Kirc_types.body_v2 with
    | None ->
        print_endline "V2: No V2 IR - SKIPPED" ;
        true
    | Some ir ->
        let n = 1024 in
        let src = V2_Vector.create V2_Vector.int32 n in
        let dst = V2_Vector.create V2_Vector.int32 n in
        for i = 0 to n - 1 do
          V2_Vector.set src i (Int32.of_int i)
        done ;

        let block = Execute.dims1d 256 in
        let grid = Execute.dims1d (n / 256) in
        Execute.run_vectors
          ~device:dev
          ~ir
          ~args:[Execute.Vec src; Execute.Vec dst]
          ~block
          ~grid
          () ;
        V2_Transfer.flush dev ;

        let ok = ref true in
        for i = 0 to n - 1 do
          let expected = Int32.of_int (2 * i) in
          if V2_Vector.get dst i <> expected then (
            Printf.printf
              "V2 FAIL: dst[%d] = %ld, expected %ld\n"
              i
              (V2_Vector.get dst i)
              expected ;
            ok := false)
        done ;
        if !ok then print_endline "PASS: Times two (V2)" ;
        !ok

let test_identity_v2 () =
  let _, kirc = identity_kernel in
  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length v2_devs = 0 then (
    print_endline "V2: No device - SKIPPED" ;
    true)
  else
    let dev = v2_devs.(0) in
    match kirc.Sarek.Kirc_types.body_v2 with
    | None ->
        print_endline "V2: No V2 IR - SKIPPED" ;
        true
    | Some ir ->
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
          if V2_Vector.get dst_i i <> expected_i then (
            Printf.printf
              "V2 FAIL: dst_i[%d] = %ld, expected %ld\n"
              i
              (V2_Vector.get dst_i i)
              expected_i ;
            ok := false) ;
          if abs_float (V2_Vector.get dst_f i -. expected_f) > 0.001 then (
            Printf.printf
              "V2 FAIL: dst_f[%d] = %f, expected %f\n"
              i
              (V2_Vector.get dst_f i)
              expected_f ;
            ok := false)
        done ;
        if !ok then print_endline "PASS: Polymorphic identity (V2)" ;
        !ok

(* ========== Interpreter Tests ========== *)

let test_basic_interpreter () =
  let _, kirc = basic_kernel in
  match kirc.Sarek.Kirc_types.body_v2 with
  | None ->
      print_endline "Interpreter: No V2 IR - SKIPPED" ;
      true
  | Some ir ->
      let n = 1024 in
      let src =
        Array.init n (fun i -> Sarek_ir_interp.VInt32 (Int32.of_int i))
      in
      let dst = Array.make n (Sarek_ir_interp.VInt32 0l) in

      Sarek_ir_interp.run_kernel
        ir
        ~block:(256, 1, 1)
        ~grid:(n / 256, 1, 1)
        [
          ("src", Sarek_ir_interp.ArgArray src);
          ("dst", Sarek_ir_interp.ArgArray dst);
        ] ;

      let ok = ref true in
      for i = 0 to n - 1 do
        let expected = Int32.of_int (i + 1) in
        let got =
          match dst.(i) with Sarek_ir_interp.VInt32 v -> v | _ -> 0l
        in
        if got <> expected then (
          if !ok then
            Printf.printf
              "Interpreter FAIL: dst[%d] = %ld, expected %ld\n"
              i
              got
              expected ;
          ok := false)
      done ;
      if !ok then print_endline "PASS: Basic module function (Interpreter)" ;
      !ok

let () =
  print_endline "=== Module Functions & Polymorphism Tests (V2) ===" ;
  print_endline "" ;

  print_endline "=== V2 Tests ===" ;
  let v1 = test_basic_v2 () in
  let v2 = test_times_two_v2 () in
  let v3 = test_identity_v2 () in
  print_endline "" ;

  print_endline "=== Interpreter Tests ===" ;
  let i1 = test_basic_interpreter () in
  print_endline "" ;

  print_endline "=== Summary ===" ;
  Printf.printf
    "Basic module function (V2): %s\n"
    (if v1 then "PASS" else "FAIL") ;
  Printf.printf "Times two (V2): %s\n" (if v2 then "PASS" else "FAIL") ;
  Printf.printf
    "Polymorphic identity (V2): %s\n"
    (if v3 then "PASS" else "FAIL") ;
  Printf.printf
    "Basic module function (Interpreter): %s\n"
    (if i1 then "PASS" else "FAIL") ;

  if v1 && v2 && v3 && i1 then (
    print_endline "\nAll tests passed!" ;
    exit 0)
  else (
    print_endline "\nSome tests failed!" ;
    exit 1)
