(******************************************************************************
 * E2E test for Sarek PPX - Tail recursion transformation
 *
 * Tests that tail-recursive functions are correctly transformed to loops.
 * Non-tail recursion is not currently supported (requires manual conversion
 * to tail-recursive form using accumulators).
 * V2 runtime only.
 ******************************************************************************)

module Std = Sarek_stdlib.Std
open Sarek
module V2_Vector = Sarek_core.Vector
module V2_Device = Sarek_core.Device
module V2_Transfer = Sarek_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
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

(* Extract kirc for V2 - power *)
let power_kirc =
  snd
    [%kernel
      let open Std in
      let rec power_aux (acc : int32) (base : int32) (exp : int32) : int32 =
        if exp <= 0l then acc else power_aux (acc * base) base (exp - 1l)
      in
      fun (output : int32 vector) (base : int32) (exp : int32) ->
        let idx = global_idx_x in
        if idx = 0l then output.(idx) <- power_aux 1l base exp]

(* Extract kirc for V2 - GCD *)
let gcd_kirc =
  snd
    [%kernel
      let open Std in
      let rec gcd (a : int32) (b : int32) : int32 =
        if b = 0l then a else gcd b (a mod b)
      in
      fun (output : int32 vector) (a : int32) (b : int32) ->
        let idx = global_idx_x in
        if idx = 0l then output.(idx) <- gcd a b]

(* V2 test for tail-recursive factorial *)
let test_factorial_v2 dev =
  print_endline "=== Tail-recursive factorial ===" ;
  Printf.printf "V2 Testing on: %s\n%!" dev.V2_Device.name ;

  let output = V2_Vector.create V2_Vector.int32 1 in
  V2_Vector.set output 0 0l ;
  let n = 10l in

  let ir =
    match factorial_kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  let t0 = Unix.gettimeofday () in
  try
    Sarek.Execute.run_vectors
      ~device:dev
      ~block:(Sarek.Execute.dims1d 1)
      ~grid:(Sarek.Execute.dims1d 1)
      ~ir
      ~args:[Sarek.Execute.Vec output; Sarek.Execute.Int32 n]
      () ;
    V2_Transfer.flush dev ;
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

(* V2 test for tail-recursive power *)
let test_power_v2 dev =
  print_endline "=== Tail-recursive power ===" ;
  Printf.printf "V2 Testing on: %s\n%!" dev.V2_Device.name ;

  let output = V2_Vector.create V2_Vector.int32 1 in
  V2_Vector.set output 0 0l ;
  let base = 2l in
  let exp = 10l in

  let ir =
    match power_kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  let t0 = Unix.gettimeofday () in
  try
    Sarek.Execute.run_vectors
      ~device:dev
      ~block:(Sarek.Execute.dims1d 1)
      ~grid:(Sarek.Execute.dims1d 1)
      ~ir
      ~args:
        [
          Sarek.Execute.Vec output;
          Sarek.Execute.Int32 base;
          Sarek.Execute.Int32 exp;
        ]
      () ;
    V2_Transfer.flush dev ;
    let t1 = Unix.gettimeofday () in
    Printf.printf "  V2 exec: %.2f ms\n%!" ((t1 -. t0) *. 1000.0) ;

    let got = V2_Vector.get output 0 in
    (* 2^10 = 1024 *)
    let expected = 1024l in
    if got = expected then (
      Printf.printf "V2 PASS: power(2, 10) = %ld\n%!" got ;
      true)
    else (
      Printf.printf "V2 FAIL: power(2, 10) = %ld, expected %ld\n%!" got expected ;
      false)
  with e ->
    Printf.printf "V2 FAIL: %s\n%!" (Printexc.to_string e) ;
    false

(* V2 test for tail-recursive GCD *)
let test_gcd_v2 dev =
  print_endline "=== Tail-recursive GCD ===" ;
  Printf.printf "V2 Testing on: %s\n%!" dev.V2_Device.name ;

  let output = V2_Vector.create V2_Vector.int32 1 in
  V2_Vector.set output 0 0l ;
  let a = 48l in
  let b = 18l in

  let ir =
    match gcd_kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in

  let t0 = Unix.gettimeofday () in
  try
    Sarek.Execute.run_vectors
      ~device:dev
      ~block:(Sarek.Execute.dims1d 1)
      ~grid:(Sarek.Execute.dims1d 1)
      ~ir
      ~args:
        [Sarek.Execute.Vec output; Sarek.Execute.Int32 a; Sarek.Execute.Int32 b]
      () ;
    V2_Transfer.flush dev ;
    let t1 = Unix.gettimeofday () in
    Printf.printf "  V2 exec: %.2f ms\n%!" ((t1 -. t0) *. 1000.0) ;

    let got = V2_Vector.get output 0 in
    (* gcd(48, 18) = 6 *)
    let expected = 6l in
    if got = expected then (
      Printf.printf "V2 PASS: gcd(48, 18) = %ld\n%!" got ;
      true)
    else (
      Printf.printf "V2 FAIL: gcd(48, 18) = %ld, expected %ld\n%!" got expected ;
      false)
  with e ->
    Printf.printf "V2 FAIL: %s\n%!" (Printexc.to_string e) ;
    false

let () =
  print_endline "=== Tail Recursion Transformation Tests (V2) ===" ;
  print_endline "" ;

  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length v2_devs = 0 then (
    print_endline "No V2 devices found - skipping" ;
    exit 0) ;

  let dev = v2_devs.(0) in

  let t1 = test_factorial_v2 dev in
  print_endline "" ;

  let t2 = test_power_v2 dev in
  print_endline "" ;

  let t3 = test_gcd_v2 dev in
  print_endline "" ;

  print_endline "=== Summary ===" ;
  Printf.printf "V2 Factorial: %s\n" (if t1 then "PASS" else "FAIL") ;
  Printf.printf "V2 Power: %s\n" (if t2 then "PASS" else "FAIL") ;
  Printf.printf "V2 GCD: %s\n" (if t3 then "PASS" else "FAIL") ;

  if t1 && t2 && t3 then (
    print_endline "\nAll tests passed!" ;
    exit 0)
  else (
    print_endline "\nSome tests failed!" ;
    exit 1)
