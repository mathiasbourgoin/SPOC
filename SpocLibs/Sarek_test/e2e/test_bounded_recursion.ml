(******************************************************************************
 * E2E test for Sarek PPX - Tail recursion transformation
 *
 * Tests that tail-recursive functions are correctly transformed to loops.
 * Non-tail recursion is not currently supported (requires manual conversion
 * to tail-recursive form using accumulators).
 ******************************************************************************)

open Spoc
open Sarek
module V2_Vector = Sarek_core.Vector
module V2_Device = Sarek_core.Device
module V2_Transfer = Sarek_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

(* Extract kirc for V2 - factorial *)
let factorial_kirc =
  snd
    [%kernel
      let open Std in
      let rec fact_aux (acc : int32) (n : int32) : int32 =
        if n <= 1l then acc else fact_aux (acc * n) (n - 1l)
      in
      fun (output : int32 vector) (n : int32) ->
        let idx = global_idx_x in
        if idx = 0l then output.(idx) <- fact_aux 1l n]

(* Test 1: Tail-recursive factorial with accumulator *)
let test_factorial () =
  let factorial_kernel =
    [%kernel
      let open Std in
      (* Tail-recursive factorial using accumulator pattern *)
      let rec fact_aux (acc : int32) (n : int32) : int32 =
        if n <= 1l then acc else fact_aux (acc * n) (n - 1l)
      in
      fun (output : int32 vector) (n : int32) ->
        let idx = global_idx_x in
        if idx = 0l then output.(idx) <- fact_aux 1l n]
  in
  let _, kirc = factorial_kernel in
  print_endline "=== Tail-recursive factorial ===" ;
  Kirc.print_ast kirc.Kirc.body ;
  print_endline "================================" ;

  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No devices - skipping runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "Testing on: %s\n%!" dev.Devices.general_info.Devices.name ;

    let output = Vector.create Vector.int32 1 in
    Mem.set output 0 0l ;
    let n = 10 in

    let block = {Kernel.blockX = 1; blockY = 1; blockZ = 1} in
    let grid = {Kernel.gridX = 1; gridY = 1; gridZ = 1} in
    Kirc.run factorial_kernel (output, n) (block, grid) 0 dev ;
    Mem.to_cpu output () ;
    Spoc.Devices.flush dev () ;

    let got = Mem.get output 0 in
    (* 10! = 3628800 *)
    let expected = 3628800l in
    if got = expected then (
      Printf.printf "PASS: fact(%d) = %ld (expected %ld)\n" n got expected ;
      true)
    else (
      Printf.printf "FAIL: fact(%d) = %ld, expected %ld\n" n got expected ;
      false)

(* Test 2: Tail-recursive power *)
let test_power () =
  let power_kernel =
    [%kernel
      let open Std in
      (* Tail-recursive power using accumulator - x^n *)
      let rec power_aux (acc : int32) (base : int32) (exp : int32) : int32 =
        if exp <= 0l then acc else power_aux (acc * base) base (exp - 1l)
      in
      fun (output : int32 vector) (base : int32) (exp : int32) ->
        let idx = global_idx_x in
        if idx = 0l then output.(idx) <- power_aux 1l base exp]
  in
  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No devices - skipping runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "Testing on: %s\n%!" dev.Devices.general_info.Devices.name ;

    let output = Vector.create Vector.int32 1 in
    Mem.set output 0 0l ;
    let base = 2 in
    let exp = 10 in

    let block = {Kernel.blockX = 1; blockY = 1; blockZ = 1} in
    let grid = {Kernel.gridX = 1; gridY = 1; gridZ = 1} in
    Kirc.run power_kernel (output, base, exp) (block, grid) 0 dev ;
    Mem.to_cpu output () ;
    Spoc.Devices.flush dev () ;

    let got = Mem.get output 0 in
    (* 2^10 = 1024 *)
    let expected = 1024l in
    if got = expected then (
      Printf.printf
        "PASS: power(%d, %d) = %ld (expected %ld)\n"
        base
        exp
        got
        expected ;
      true)
    else (
      Printf.printf
        "FAIL: power(%d, %d) = %ld, expected %ld\n"
        base
        exp
        got
        expected ;
      false)

(* Test 3: Tail-recursive GCD *)
let test_gcd () =
  let gcd_kernel =
    [%kernel
      let open Std in
      (* Euclidean algorithm is naturally tail-recursive *)
      let rec gcd (a : int32) (b : int32) : int32 =
        if b = 0l then a else gcd b (a mod b)
      in
      fun (output : int32 vector) (a : int32) (b : int32) ->
        let idx = global_idx_x in
        if idx = 0l then output.(idx) <- gcd a b]
  in
  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No devices - skipping runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "Testing on: %s\n%!" dev.Devices.general_info.Devices.name ;

    let output = Vector.create Vector.int32 1 in
    Mem.set output 0 0l ;
    let a = 48 in
    let b = 18 in

    let block = {Kernel.blockX = 1; blockY = 1; blockZ = 1} in
    let grid = {Kernel.gridX = 1; gridY = 1; gridZ = 1} in
    Kirc.run gcd_kernel (output, a, b) (block, grid) 0 dev ;
    Mem.to_cpu output () ;
    Spoc.Devices.flush dev () ;

    let got = Mem.get output 0 in
    (* gcd(48, 18) = 6 *)
    let expected = 6l in
    if got = expected then (
      Printf.printf "PASS: gcd(%d, %d) = %ld (expected %ld)\n" a b got expected ;
      true)
    else (
      Printf.printf "FAIL: gcd(%d, %d) = %ld, expected %ld\n" a b got expected ;
      false)

(* V2 test for tail-recursive factorial *)
let test_factorial_v2 () =
  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length v2_devs = 0 then (
    print_endline "No V2 devices - skipping" ;
    true)
  else
    let v2_dev = v2_devs.(0) in
    Printf.printf "V2 Testing on: %s\n%!" v2_dev.V2_Device.name ;

    let output = V2_Vector.create V2_Vector.int32 1 in
    V2_Vector.set output 0 0l ;
    let n = 10l in

    let ir =
      match factorial_kirc.Kirc.body_v2 with
      | Some ir -> ir
      | None -> failwith "Kernel has no V2 IR"
    in

    let t0 = Unix.gettimeofday () in
    try
      Sarek.Execute.run_vectors
        ~device:v2_dev
        ~block:(Sarek.Execute.dims1d 1)
        ~grid:(Sarek.Execute.dims1d 1)
        ~ir
        ~args:[Sarek.Execute.Vec output; Sarek.Execute.Int32 n]
        () ;
      V2_Transfer.flush v2_dev ;
      let t1 = Unix.gettimeofday () in
      Printf.printf "  V2 exec: %.2f ms\n%!" ((t1 -. t0) *. 1000.0) ;

      let got = V2_Vector.get output 0 in
      (* 10! = 3628800 *)
      let expected = 3628800l in
      if got = expected then (
        Printf.printf "V2 PASS: fact(10) = %ld\n%!" got ;
        true)
      else (
        Printf.printf "V2 FAIL: fact(10) = %ld, expected %ld\n%!" got expected ;
        false)
    with e ->
      Printf.printf "V2 FAIL: %s\n%!" (Printexc.to_string e) ;
      false

let () =
  print_endline "=== Tail Recursion Transformation Tests ===" ;
  print_endline "" ;

  print_endline "--- SPOC Path ---" ;
  let t1 =
    try test_factorial ()
    with e ->
      Printf.printf "SPOC Factorial FAIL: %s\n%!" (Printexc.to_string e) ;
      false
  in
  print_endline "" ;

  let t2 =
    try test_power ()
    with e ->
      Printf.printf "SPOC Power FAIL: %s\n%!" (Printexc.to_string e) ;
      false
  in
  print_endline "" ;

  let t3 =
    try test_gcd ()
    with e ->
      Printf.printf "SPOC GCD FAIL: %s\n%!" (Printexc.to_string e) ;
      false
  in
  print_endline "" ;

  print_endline "--- V2 Path ---" ;
  let t4 = test_factorial_v2 () in
  print_endline "" ;

  print_endline "=== Summary ===" ;
  Printf.printf "SPOC Factorial: %s\n" (if t1 then "PASS" else "FAIL") ;
  Printf.printf "SPOC Power: %s\n" (if t2 then "PASS" else "FAIL") ;
  Printf.printf "SPOC GCD: %s\n" (if t3 then "PASS" else "FAIL") ;
  Printf.printf "V2 Factorial: %s\n" (if t4 then "PASS" else "FAIL") ;

  if t1 && t2 && t3 && t4 then (
    print_endline "\nAll tests passed!" ;
    exit 0)
  else (
    print_endline "\nSome tests failed!" ;
    exit 1)
