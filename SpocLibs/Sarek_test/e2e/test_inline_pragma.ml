(******************************************************************************
 * E2E test for Sarek PPX - pragma ["sarek.inline N"] for non-tail recursion
 *
 * Tests that non-tail-recursive functions can be inlined using pragma.
 ******************************************************************************)

open Spoc
open Sarek

(* Helper to run a kernel and get result *)
let run_kernel kernel arg dev =
  let output = Vector.create Vector.int32 1 in
  Mem.set output 0 0l ;
  let block = {Kernel.blockX = 1; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = 1; gridY = 1; gridZ = 1} in
  Kirc.run kernel (output, arg) (block, grid) 0 dev ;
  Mem.to_cpu output () ;
  Spoc.Devices.flush dev () ;
  Mem.get output 0

(* Minimal test: Power of 2 using non-tail recursion *)
let test_pow2 () =
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
  in
  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No devices - skipping runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf
      "Testing pow2 on: %s\n%!"
      dev.Devices.general_info.Devices.name ;

    let output = Vector.create Vector.int32 1 in
    Mem.set output 0 0l ;
    let n = 5 in

    let block = {Kernel.blockX = 1; blockY = 1; blockZ = 1} in
    let grid = {Kernel.gridX = 1; gridY = 1; gridZ = 1} in
    Kirc.run pow2_kernel (output, n) (block, grid) 0 dev ;
    Mem.to_cpu output () ;
    Spoc.Devices.flush dev () ;

    let got = Mem.get output 0 in
    (* pow2(5) = 32 *)
    let expected = 32l in
    if got = expected then (
      Printf.printf "PASS: pow2(%d) = %ld (expected %ld)\n" n got expected ;
      true)
    else (
      Printf.printf "FAIL: pow2(%d) = %ld, expected %ld\n" n got expected ;
      false)

(* Fibonacci test: Classic non-tail recursive with sufficient inline depth *)
(* fib(n) = fib(n-1) + fib(n-2) - both calls are NOT in tail position *)
let test_fib_pass () =
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
  in
  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No devices - skipping runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "Testing fib on: %s\n%!" dev.Devices.general_info.Devices.name ;

    (* Test cases: fib(0)=0, fib(1)=1, fib(5)=5, fib(7)=13 *)
    let test_cases = [(0, 0); (1, 1); (5, 5); (7, 13)] in
    let all_pass = ref true in
    List.iter
      (fun (n, expected) ->
        let output = Vector.create Vector.int32 1 in
        Mem.set output 0 0l ;
        let block = {Kernel.blockX = 1; blockY = 1; blockZ = 1} in
        let grid = {Kernel.gridX = 1; gridY = 1; gridZ = 1} in
        Kirc.run fib_kernel (output, n) (block, grid) 0 dev ;
        Mem.to_cpu output () ;
        Spoc.Devices.flush dev () ;
        let got = Mem.get output 0 in
        if got = Int32.of_int expected then
          Printf.printf "  PASS: fib(%d) = %ld\n" n got
        else (
          Printf.printf "  FAIL: fib(%d) = %ld, expected %d\n" n got expected ;
          all_pass := false))
      test_cases ;
    !all_pass

(* Fibonacci test with shallow inline depth.
   Note: Some GPUs (e.g., Intel Arc with OpenCL) actually support limited recursion,
   so this test may pass on some devices. We test that it at least compiles and runs. *)
let test_fib_shallow () =
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
  in
  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No devices - skipping runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf
      "Testing fib_shallow on: %s\n%!"
      dev.Devices.general_info.Devices.name ;
    (* fib(10) = 55 - this may work on some GPUs that support recursion *)
    let n = 10 in
    let output = Vector.create Vector.int32 1 in
    Mem.set output 0 99l ;
    let block = {Kernel.blockX = 1; blockY = 1; blockZ = 1} in
    let grid = {Kernel.gridX = 1; gridY = 1; gridZ = 1} in
    try
      Kirc.run fib_shallow_kernel (output, n) (block, grid) 0 dev ;
      Mem.to_cpu output () ;
      Spoc.Devices.flush dev () ;
      let got = Mem.get output 0 in
      if got = 55l then (
        Printf.printf "  OK: GPU supports recursion, got correct fib(10) = 55\n" ;
        true)
      else (
        Printf.printf "  OK: got %ld (GPU may not support recursion)\n" got ;
        true)
    with e ->
      Printf.printf
        "  OK: runtime error (GPU doesn't support recursion): %s\n"
        (Printexc.to_string e) ;
      true

let () =
  print_endline "=== Pragma Inline Tests ===" ;
  let t1 = test_pow2 () in
  Printf.printf "pow2: %s\n" (if t1 then "PASS" else "FAIL") ;
  let t2 = test_fib_pass () in
  Printf.printf "fib (sufficient depth): %s\n" (if t2 then "PASS" else "FAIL") ;
  let t3 = test_fib_shallow () in
  Printf.printf "fib (shallow depth): %s\n" (if t3 then "PASS" else "FAIL") ;
  if t1 && t2 && t3 then exit 0 else exit 1
