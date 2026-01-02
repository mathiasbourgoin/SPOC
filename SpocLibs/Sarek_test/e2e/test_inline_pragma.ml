(******************************************************************************
 * E2E test for Sarek PPX - pragma ["sarek.inline N"] for non-tail recursion
 *
 * Tests that non-tail-recursive functions can be inlined using pragma.
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

(* Minimal test: Power of 2 using non-tail recursion *)
let pow2_kernel =
  [%kernel
    let open Std in
    (* Non-tail recursive power of 2: pow2(n) = 2 * pow2(n-1) *)
    let rec pow2 (n : int32) : int32 =
      pragma ["sarek.inline 6"] (if n <= 0l then 1l else 2l * pow2 (n - 1l))
    in
    fun (output : int32 vector) (n : int32) ->
      let idx = global_idx_x in
      if idx = 0l then output.(idx) <- pow2 n]

(* Fibonacci kernel with sufficient inline depth *)
let fib_kernel =
  [%kernel
    let open Std in
    (* Classic Fibonacci: fib(0)=0, fib(1)=1, fib(n) = fib(n-1) + fib(n-2) *)
    let rec fib (n : int32) : int32 =
      pragma
        ["sarek.inline 6"]
        (if n <= 0l then 0l
         else if n = 1l then 1l
         else fib (n - 1l) + fib (n - 2l))
    in
    fun (output : int32 vector) (n : int32) ->
      let idx = global_idx_x in
      if idx = 0l then output.(idx) <- fib n]

(* Fibonacci with shallow inline depth *)
let fib_shallow_kernel =
  [%kernel
    let open Std in
    let rec fib (n : int32) : int32 =
      pragma
        ["sarek.inline 3"]
        (if n <= 0l then 0l
         else if n = 1l then 1l
         else fib (n - 1l) + fib (n - 2l))
    in
    fun (output : int32 vector) (n : int32) ->
      let idx = global_idx_x in
      if idx = 0l then output.(idx) <- fib n]

(* === SPOC Path === *)

let test_pow2_spoc () =
  let devs = Spoc_Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No devices - skipping SPOC runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf
      "SPOC: Testing pow2 on: %s\n%!"
      dev.Spoc_Devices.general_info.Spoc_Devices.name ;

    let output = Spoc_Vector.create Spoc_Vector.int32 1 in
    Spoc_Mem.set output 0 0l ;
    let n = 5 in

    let block = {Spoc.Kernel.blockX = 1; blockY = 1; blockZ = 1} in
    let grid = {Spoc.Kernel.gridX = 1; gridY = 1; gridZ = 1} in
    Kirc.run pow2_kernel (output, n) (block, grid) 0 dev ;
    Spoc_Mem.to_cpu output () ;
    Spoc_Devices.flush dev () ;

    let got = Spoc_Mem.get output 0 in
    let expected = 32l in
    if got = expected then (
      Printf.printf "  PASS: pow2(%d) = %ld (expected %ld)\n" n got expected ;
      true)
    else (
      Printf.printf "  FAIL: pow2(%d) = %ld, expected %ld\n" n got expected ;
      false)

let test_fib_spoc () =
  let devs = Spoc_Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No devices - skipping SPOC runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf
      "SPOC: Testing fib on: %s\n%!"
      dev.Spoc_Devices.general_info.Spoc_Devices.name ;

    let test_cases = [(0, 0); (1, 1); (5, 5); (7, 13)] in
    let all_pass = ref true in
    List.iter
      (fun (n, expected) ->
        let output = Spoc_Vector.create Spoc_Vector.int32 1 in
        Spoc_Mem.set output 0 0l ;
        let block = {Spoc.Kernel.blockX = 1; blockY = 1; blockZ = 1} in
        let grid = {Spoc.Kernel.gridX = 1; gridY = 1; gridZ = 1} in
        Kirc.run fib_kernel (output, n) (block, grid) 0 dev ;
        Spoc_Mem.to_cpu output () ;
        Spoc_Devices.flush dev () ;
        let got = Spoc_Mem.get output 0 in
        if got = Int32.of_int expected then
          Printf.printf "  PASS: fib(%d) = %ld\n" n got
        else (
          Printf.printf "  FAIL: fib(%d) = %ld, expected %d\n" n got expected ;
          all_pass := false))
      test_cases ;
    !all_pass

(* === V2 Path === *)

let test_pow2_v2 () =
  let devs = V2_Device.all () in
  if Array.length devs = 0 then (
    print_endline "No V2 devices - skipping V2 runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "V2: Testing pow2 on: %s\n%!" dev.V2_Device.name ;

    let _, kirc = pow2_kernel in
    let ir =
      match kirc.Kirc.body_v2 with
      | Some ir -> ir
      | None -> failwith "Kernel has no V2 IR"
    in

    let output = V2_Vector.create V2_Vector.int32 1 in
    V2_Vector.set output 0 0l ;
    let n = 5 in

    let block = Execute.dims1d 1 in
    let grid = Execute.dims1d 1 in
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:[Execute.Vec output; Execute.Int32 (Int32.of_int n)]
      ~block
      ~grid
      () ;
    V2_Transfer.flush dev ;

    let got = V2_Vector.get output 0 in
    let expected = 32l in
    if got = expected then (
      Printf.printf "  PASS: pow2(%d) = %ld (expected %ld)\n" n got expected ;
      true)
    else (
      Printf.printf "  FAIL: pow2(%d) = %ld, expected %ld\n" n got expected ;
      false)

let test_fib_v2 () =
  let devs = V2_Device.all () in
  if Array.length devs = 0 then (
    print_endline "No V2 devices - skipping V2 runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "V2: Testing fib on: %s\n%!" dev.V2_Device.name ;

    let _, kirc = fib_kernel in
    let ir =
      match kirc.Kirc.body_v2 with
      | Some ir -> ir
      | None -> failwith "Kernel has no V2 IR"
    in

    let test_cases = [(0, 0); (1, 1); (5, 5); (7, 13)] in
    let all_pass = ref true in
    List.iter
      (fun (n, expected) ->
        let output = V2_Vector.create V2_Vector.int32 1 in
        V2_Vector.set output 0 0l ;
        let block = Execute.dims1d 1 in
        let grid = Execute.dims1d 1 in
        Execute.run_vectors
          ~device:dev
          ~ir
          ~args:[Execute.Vec output; Execute.Int32 (Int32.of_int n)]
          ~block
          ~grid
          () ;
        V2_Transfer.flush dev ;
        let got = V2_Vector.get output 0 in
        if got = Int32.of_int expected then
          Printf.printf "  PASS: fib(%d) = %ld\n" n got
        else (
          Printf.printf "  FAIL: fib(%d) = %ld, expected %d\n" n got expected ;
          all_pass := false))
      test_cases ;
    !all_pass

let () =
  print_endline "=== Pragma Inline Tests ===" ;
  print_endline "" ;

  print_endline "--- SPOC Path ---" ;
  let t1_spoc = test_pow2_spoc () in
  Printf.printf "pow2 (SPOC): %s\n" (if t1_spoc then "PASS" else "FAIL") ;
  let t2_spoc = test_fib_spoc () in
  Printf.printf "fib (SPOC): %s\n" (if t2_spoc then "PASS" else "FAIL") ;
  print_endline "" ;

  print_endline "--- V2 Path ---" ;
  let t1_v2 = test_pow2_v2 () in
  Printf.printf "pow2 (V2): %s\n" (if t1_v2 then "PASS" else "FAIL") ;
  let t2_v2 = test_fib_v2 () in
  Printf.printf "fib (V2): %s\n" (if t2_v2 then "PASS" else "FAIL") ;
  print_endline "" ;

  if t1_spoc && t2_spoc && t1_v2 && t2_v2 then (
    print_endline "=== All pragma inline tests PASSED ===" ;
    exit 0)
  else (
    print_endline "=== Some tests FAILED ===" ;
    exit 1)
