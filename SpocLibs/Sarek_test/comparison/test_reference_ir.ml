(******************************************************************************
 * Sarek Reference IR Validation
 *
 * This test validates that the reference IR file can be parsed and contains
 * the expected kernels. This is a precursor to full comparison tests.
 ******************************************************************************)

(* Parse the reference IR file and extract individual kernel sections *)
let parse_reference_ir filename =
  let ic = open_in filename in
  let content = really_input_string ic (in_channel_length ic) in
  close_in ic ;

  (* Split by kernel markers *)
  let kernel_re = Str.regexp "=== \\([a-zA-Z0-9_]+\\) ===" in
  let rec find_kernels pos acc =
    try
      let _ = Str.search_forward kernel_re content pos in
      let name = Str.matched_group 1 content in
      let match_end = Str.match_end () in
      find_kernels match_end (name :: acc)
    with Not_found -> List.rev acc
  in
  find_kernels 0 []

(* Expected kernels from extract_ir.ml *)
let expected_kernels =
  [
    "k_int_literal";
    "k_float_literal";
    "k_arithmetic_int";
    "k_arithmetic_float";
    "k_comparisons";
    "k_comparisons_float";
    "k_thread_indices";
    "k_thread_indices_2d";
    "k_global_thread_id";
    "k_if_then_else";
    "k_if_no_else";
    "k_while_loop";
    "k_for_loop";
    "k_nested_loops";
    "k_vector_read";
    "k_vector_write";
    "k_vector_read_write";
    "k_let_simple";
    "k_let_nested";
    "k_let_mutable";
    "k_math_float";
    "k_math_more";
    "k_math_int";
    "k_barrier";
    "k_early_return";
    "k_saxpy";
    "k_bitonic";
    "k_to_gray";
    "k_nested_functions";
    "k_deeply_nested_functions";
    "k_boolean_ops";
    "k_negation";
    "k_int_negation";
    "k_conversions";
  ]

let find_reference_file () =
  let candidates =
    [
      "SpocLibs/Sarek_test/regression/reference_ir.txt";
      "../regression/reference_ir.txt";
      "../../Sarek_test/regression/reference_ir.txt";
    ]
  in
  try List.find Sys.file_exists candidates
  with Not_found -> failwith "Could not find reference_ir.txt"

let () =
  Printf.printf "=== Reference IR Validation ===\n\n" ;

  let filename = find_reference_file () in
  Printf.printf "Using reference file: %s\n\n" filename ;

  let found_kernels = parse_reference_ir filename in
  Printf.printf
    "Found %d kernels in reference file:\n"
    (List.length found_kernels) ;
  List.iter (Printf.printf "  - %s\n") found_kernels ;

  Printf.printf "\nExpected %d kernels\n" (List.length expected_kernels) ;

  (* Check for missing kernels *)
  let missing =
    List.filter (fun k -> not (List.mem k found_kernels)) expected_kernels
  in
  let extra =
    List.filter (fun k -> not (List.mem k expected_kernels)) found_kernels
  in

  if missing <> [] then begin
    Printf.printf "\nMISSING kernels:\n" ;
    List.iter (Printf.printf "  - %s\n") missing
  end ;

  if extra <> [] then begin
    Printf.printf "\nEXTRA kernels (not in expected list):\n" ;
    List.iter (Printf.printf "  - %s\n") extra
  end ;

  let success = missing = [] && List.length found_kernels >= 32 in

  if success then begin
    Printf.printf "\n[PASS] Reference IR validation successful\n" ;
    Printf.printf
      "       %d kernels available for comparison testing\n"
      (List.length found_kernels) ;
    exit 0
  end
  else begin
    Printf.printf "\n[FAIL] Reference IR validation failed\n" ;
    exit 1
  end
