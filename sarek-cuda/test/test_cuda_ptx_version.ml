(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The header rewrite [load_from_ptx] applies before [cuModuleLoadData].

    [.target] is rewritten to the device SM, so [.version] must be at least the
    PTX ISA that admits that SM: PTX ISA 8.0 does not know sm_120 (issue #506).
    Pure string tests, no device and no libcuda. *)

open Sarek_cuda
open Alcotest

let generated =
  ".version 8.0\n\
   .target sm_86\n\
   .address_size 64\n\
   .visible .entry k() { ret; }\n"

let header ptx =
  let starts prefix l =
    String.length l >= String.length prefix
    && String.sub l 0 (String.length prefix) = prefix
  in
  String.split_on_char '\n' ptx
  |> List.filter (fun l -> starts ".version" l || starts ".target" l)

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

let rewrite ~major ~minor ptx = Cuda_api.Kernel.with_sm_target ~major ~minor ptx

let test_blackwell () =
  check
    (list string)
    "sm_120 needs ISA 8.7"
    [".version 8.7"; ".target sm_120"]
    (header (rewrite ~major:12 ~minor:0 generated))

let test_pascal () =
  check
    (list string)
    "sm_61 keeps the generated 8.0, never lowered"
    [".version 8.0"; ".target sm_61"]
    (header (rewrite ~major:6 ~minor:1 generated))

let test_never_lowers () =
  check
    (list string)
    "a newer .version survives"
    [".version 9.0"; ".target sm_80"]
    (header (rewrite ~major:8 ~minor:0 ".version 9.0\n.target sm_86\n"))

let test_table () =
  let v ~major ~minor =
    Cuda_ptx_isa.(to_string (required_version ~major ~minor))
  in
  check string "sm_86" "7.1" (v ~major:8 ~minor:6) ;
  check string "sm_90" "7.8" (v ~major:9 ~minor:0) ;
  check string "sm_100" "8.6" (v ~major:10 ~minor:0) ;
  check string "sm_120" "8.7" (v ~major:12 ~minor:0) ;
  check string "sm_121" "8.8" (v ~major:12 ~minor:1) ;
  check string "sm_61" "7.0" (v ~major:6 ~minor:1)

let test_driver_gate () =
  let required = Cuda_ptx_isa.required_version ~major:12 ~minor:0 in
  (match Cuda_ptx_isa.check_driver ~driver:(12, 4) ~required with
  | Ok () -> fail "CUDA 12.4 driver cannot accept ISA 8.7"
  | Error msg ->
      check bool "names the needed ISA" true (contains msg "8.7") ;
      check bool "names the driver ISA" true (contains msg "8.4")) ;
  check
    bool
    "CUDA 12.8 driver accepts 8.7"
    true
    (Cuda_ptx_isa.check_driver ~driver:(12, 8) ~required = Ok ()) ;
  check
    bool
    "a driver newer than the table is not gated"
    true
    (Cuda_ptx_isa.check_driver ~driver:(13, 4) ~required = Ok ())

let () =
  run
    "cuda_ptx_version"
    [
      ( "with_sm_target",
        [
          test_case "sm_120 raises .version" `Quick test_blackwell;
          test_case "sm_61 keeps .version" `Quick test_pascal;
          test_case "never lowers .version" `Quick test_never_lowers;
        ] );
      ( "isa",
        [
          test_case "required version per SM" `Quick test_table;
          test_case "driver gate names both versions" `Quick test_driver_gate;
        ] );
    ]
