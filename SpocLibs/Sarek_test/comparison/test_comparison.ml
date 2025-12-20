(******************************************************************************
 * Sarek PPX Comparison Tests
 *
 * This module tests that the PPX implementation produces IR that matches
 * the reference IR extracted from the camlp4 implementation.
 *
 * The reference IR is in SpocLibs/Sarek_test/regression/reference_ir.txt
 ******************************************************************************)

open Sarek_ppx_lib

(* Parse the reference IR file and extract individual kernel sections *)
let parse_reference_ir filename =
  let ic = open_in filename in
  let content = really_input_string ic (in_channel_length ic) in
  close_in ic;

  (* Split by kernel markers *)
  let kernel_pattern = Str.regexp "=== \\([a-zA-Z0-9_]+\\) ===" in
  let sections = Str.split_delim kernel_pattern content in

  (* Build a map of kernel name -> (body, ret_val) *)
  let rec parse_sections acc = function
    | [] -> acc
    | name :: rest when String.length name > 0 && name.[0] <> '=' ->
      (match rest with
       | section :: remaining ->
         (* Parse body and ret_val from section *)
         let body_marker = "--- body ---\n" in
         let ret_marker = "--- ret_val ---\n" in
         (try
            let body_start = String.length body_marker in
            let ret_start =
              try Str.search_forward (Str.regexp_string ret_marker) section 0
              with Not_found -> String.length section
            in
            let body = String.sub section body_start (ret_start - body_start) in
            let body = String.trim body in
            let ret_val =
              if ret_start < String.length section then
                let start = ret_start + String.length ret_marker in
                String.trim (String.sub section start (String.length section - start))
              else "Unit"
            in
            parse_sections ((String.trim name, (body, ret_val)) :: acc) remaining
          with _ -> parse_sections acc remaining)
       | [] -> acc)
    | _ :: rest -> parse_sections acc rest
  in
  parse_sections [] sections

(* Get reference for a specific kernel *)
let reference_ir = lazy (
  let filename = "SpocLibs/Sarek_test/regression/reference_ir.txt" in
  if Sys.file_exists filename then
    parse_reference_ir filename
  else
    (* Try relative path from test execution directory *)
    let alt_filename = "../regression/reference_ir.txt" in
    if Sys.file_exists alt_filename then
      parse_reference_ir alt_filename
    else
      failwith ("Reference IR file not found: " ^ filename)
)

let get_reference name =
  List.assoc_opt name (Lazy.force reference_ir)

(* Normalize IR string for comparison:
   - Remove trailing whitespace from each line
   - Normalize variable IDs (they may differ between implementations)
*)
let normalize_ir s =
  let lines = String.split_on_char '\n' s in
  let lines = List.map String.trim lines in
  let lines = List.filter (fun s -> String.length s > 0) lines in
  String.concat "\n" lines

(* Compare IR strings, ignoring variable ID differences *)
let ir_matches expected got =
  (* For now, just compare normalized strings *)
  (* TODO: implement proper structural comparison that ignores variable IDs *)
  let expected_norm = normalize_ir expected in
  let got_norm = normalize_ir got in
  expected_norm = got_norm

(* Helper to extract kernel IR *)
let extract_kernel_ir (kern : Kirc_Ast.kfun) =
  let Kirc_Ast.KernFun (body, _ret_val) = kern in
  Kirc_Ast.string_of_ast body

(* ============================================================================
 * Test Kernels - must match extract_ir.ml from camlp4 version
 * ============================================================================ *)

(* Simple integer assignment *)
let%kernel k_int_literal (v : int32 vector) =
  let open Std in
  v.[<0>] <- 42

(* Float literal *)
let%kernel k_float_literal (v : float32 vector) =
  let open Std in
  v.[<0>] <- 3.14

(* Integer arithmetic *)
let%kernel k_arithmetic_int (v : int32 vector) =
  let open Std in
  let a = 10 in
  let b = 3 in
  v.[<0>] <- a + b;
  v.[<1>] <- a - b;
  v.[<2>] <- a * b;
  v.[<3>] <- a / b

(* Float arithmetic *)
let%kernel k_arithmetic_float (v : float32 vector) =
  let open Std in
  let a = 10.0 in
  let b = 3.0 in
  v.[<0>] <- a +. b;
  v.[<1>] <- a -. b;
  v.[<2>] <- a *. b;
  v.[<3>] <- a /. b

(* ============================================================================
 * Test Suite
 * ============================================================================ *)

let test_kernel_ir name kern () =
  match get_reference name with
  | None ->
    Alcotest.fail (Printf.sprintf "No reference IR found for kernel %s" name)
  | Some (expected_body, _expected_ret) ->
    let got_body = extract_kernel_ir kern in
    if not (ir_matches expected_body got_body) then begin
      Printf.printf "\n=== Expected ===\n%s\n" expected_body;
      Printf.printf "\n=== Got ===\n%s\n" got_body;
      Alcotest.fail "IR mismatch"
    end

let () =
  (* For now, just verify the test infrastructure works *)
  Printf.printf "Comparison tests - checking reference IR file...\n";

  let refs = Lazy.force reference_ir in
  Printf.printf "Found %d kernels in reference file\n" (List.length refs);

  List.iter (fun (name, (body, ret)) ->
    Printf.printf "  - %s (body: %d chars, ret: %s)\n"
      name (String.length body) ret
  ) refs;

  (* TODO: Run actual comparison tests once PPX is complete *)
  Printf.printf "\nNote: Full comparison tests pending PPX implementation\n"
