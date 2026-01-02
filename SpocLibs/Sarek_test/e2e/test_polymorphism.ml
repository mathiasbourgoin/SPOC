(******************************************************************************
 * Sarek PPX - End-to-End Test for Module Functions and Polymorphism
 *
 * Tests:
 * 1. Basic module functions with concrete types
 * 2. Polymorphic module functions used at multiple types
 * Compares SPOC and V2 runtime paths.
 ******************************************************************************)

open Spoc
open Sarek

(* V2 module aliases *)
module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer

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

let basic_kernel =
  [%kernel
    let add_one (x : int32) : int32 = x + 1l in
    fun (src : int32 vector) (dst : int32 vector) ->
      let open Std in
      let idx = global_idx_x in
      dst.(idx) <- add_one src.(idx)]

(** Test 1: Basic module function with concrete type *)
let test_basic_module_fun () =
  let _, kirc = basic_kernel in
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
    Kirc.run basic_kernel (src, dst) (block, grid) 0 dev ;
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
    if !ok then print_endline "PASS: Basic module function (SPOC)" ;
    !ok

let times_two_kernel =
  [%kernel
    let times_two (x : int32) : int32 = x + x in
    fun (src : int32 vector) (dst : int32 vector) ->
      let open Std in
      let idx = global_idx_x in
      dst.(idx) <- times_two src.(idx)]

(** Test 2: Multiple module functions - times_two *)
let test_multiple_module_funs () =
  let _, kirc = times_two_kernel in
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
    Kirc.run times_two_kernel (src, dst) (block, grid) 0 dev ;
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
    if !ok then print_endline "PASS: Multiple module functions (SPOC)" ;
    !ok

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

(** Test 3: Polymorphic identity function used at different types *)
let test_polymorphic_identity () =
  let _, kirc = identity_kernel in
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
    Kirc.run identity_kernel (src_i, src_f, dst_i, dst_f) (block, grid) 0 dev ;
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
    if !ok then print_endline "PASS: Polymorphic identity (SPOC)" ;
    !ok

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
    match kirc.Kirc.body_v2 with
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
    match kirc.Kirc.body_v2 with
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
    match kirc.Kirc.body_v2 with
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
  match kirc.Kirc.body_v2 with
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
  print_endline "=== Module Functions & Polymorphism Tests ===" ;
  print_endline "" ;

  let t1 = test_basic_module_fun () in
  print_endline "" ;

  let t2 = test_multiple_module_funs () in
  print_endline "" ;

  let t3 = test_polymorphic_identity () in
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
    "Basic module function (SPOC): %s\n"
    (if t1 then "PASS" else "FAIL") ;
  Printf.printf "Times two (SPOC): %s\n" (if t2 then "PASS" else "FAIL") ;
  Printf.printf
    "Polymorphic identity (SPOC): %s\n"
    (if t3 then "PASS" else "FAIL") ;
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

  if t1 && t2 && t3 && v1 && v2 && v3 && i1 then (
    print_endline "\nAll tests passed!" ;
    exit 0)
  else (
    print_endline "\nSome tests failed!" ;
    exit 1)
