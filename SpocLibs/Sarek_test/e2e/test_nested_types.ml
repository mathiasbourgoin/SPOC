(******************************************************************************
 * E2E test for composable nested custom types with [@@sarek.type]
 *
 * Tests that nested types work correctly:
 *   type point = { x: float32; y: float32 } [@@sarek.type]
 *   type colored_point = { color: int32; point: point } [@@sarek.type]
 *
 * The PPX generates composable _custom_v2 accessors that can read/write
 * nested records by delegating to the inner type's accessor.
 ******************************************************************************)

module V2_Vector = Sarek_core.Vector
module V2_Device = Sarek_core.Device
module V2_Transfer = Sarek_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

type float32 = float

(* Base type: 2D point *)
type point = {x : float32; y : float32} [@@sarek.type]

(* Enum variant for color *)
type color = Red | Green | Blue [@@sarek.type]

(* Nested type: point with a color enum *)
type colored_point = {color : color; pt : point} [@@sarek.type]

(* Test 1: Basic point type (sanity check) *)
let test_point () =
  print_endline "=== Test: Basic point type ===" ;
  let n = 4 in
  let v = V2_Vector.create_custom point_custom_v2 n in
  for i = 0 to n - 1 do
    V2_Vector.set v i {x = float_of_int i; y = float_of_int (i * 2)}
  done ;
  let ok = ref true in
  for i = 0 to n - 1 do
    let p = V2_Vector.get v i in
    let expected_x = float_of_int i in
    let expected_y = float_of_int (i * 2) in
    if
      abs_float (p.x -. expected_x) > 1e-6
      || abs_float (p.y -. expected_y) > 1e-6
    then (
      Printf.printf
        "  FAIL at %d: got {%.1f, %.1f} expected {%.1f, %.1f}\n"
        i
        p.x
        p.y
        expected_x
        expected_y ;
      ok := false)
  done ;
  if !ok then print_endline "PASSED" else print_endline "FAILED" ;
  !ok

(* Test 2: Nested colored_point type with enum color *)
let test_colored_point () =
  print_endline "=== Test: Nested colored_point type ===" ;
  let colors = [|Red; Green; Blue; Red|] in
  let n = 4 in
  let v = V2_Vector.create_custom colored_point_custom_v2 n in
  for i = 0 to n - 1 do
    V2_Vector.set
      v
      i
      {color = colors.(i); pt = {x = float_of_int i; y = float_of_int (i * 2)}}
  done ;
  let ok = ref true in
  for i = 0 to n - 1 do
    let cp = V2_Vector.get v i in
    let expected_color = colors.(i) in
    let expected_x = float_of_int i in
    let expected_y = float_of_int (i * 2) in
    if
      cp.color <> expected_color
      || abs_float (cp.pt.x -. expected_x) > 1e-6
      || abs_float (cp.pt.y -. expected_y) > 1e-6
    then (
      let color_str c =
        match c with Red -> "Red" | Green -> "Green" | Blue -> "Blue"
      in
      Printf.printf
        "  FAIL at %d: got {color=%s, pt={%.1f, %.1f}} expected {color=%s, \
         pt={%.1f, %.1f}}\n"
        i
        (color_str cp.color)
        cp.pt.x
        cp.pt.y
        (color_str expected_color)
        expected_x
        expected_y ;
      ok := false)
  done ;
  if !ok then print_endline "PASSED" else print_endline "FAILED" ;
  !ok

(* Test 3: Mixed int32 and float32 fields *)
type mixed_record = {count : int32; value : float32; flags : int32}
[@@sarek.type]

let test_mixed_record () =
  print_endline "=== Test: Mixed int32/float32 record ===" ;
  let n = 4 in
  let v = V2_Vector.create_custom mixed_record_custom_v2 n in
  for i = 0 to n - 1 do
    V2_Vector.set
      v
      i
      {
        count = Int32.of_int i;
        value = float_of_int i *. 1.5;
        flags = Int32.of_int (i * 10);
      }
  done ;
  let ok = ref true in
  for i = 0 to n - 1 do
    let r = V2_Vector.get v i in
    let expected_count = Int32.of_int i in
    let expected_value = float_of_int i *. 1.5 in
    let expected_flags = Int32.of_int (i * 10) in
    if
      r.count <> expected_count
      || abs_float (r.value -. expected_value) > 1e-6
      || r.flags <> expected_flags
    then (
      Printf.printf
        "  FAIL at %d: got {count=%ld, value=%.2f, flags=%ld}\n"
        i
        r.count
        r.value
        r.flags ;
      ok := false)
  done ;
  if !ok then print_endline "PASSED" else print_endline "FAILED" ;
  !ok

let () =
  print_endline "=== Nested/Composable Custom Types Test ===" ;
  print_endline "" ;
  let t1 = test_point () in
  print_endline "" ;
  let t2 = test_colored_point () in
  print_endline "" ;
  let t3 = test_mixed_record () in
  print_endline "" ;
  print_endline "=== Summary ===" ;
  Printf.printf "Basic point: %s\n" (if t1 then "PASS" else "FAIL") ;
  Printf.printf "Nested colored_point: %s\n" (if t2 then "PASS" else "FAIL") ;
  Printf.printf "Mixed int32/float32: %s\n" (if t3 then "PASS" else "FAIL") ;
  if t1 && t2 && t3 then (
    print_endline "\nAll tests passed!" ;
    exit 0)
  else (
    print_endline "\nSome tests failed!" ;
    exit 1)
