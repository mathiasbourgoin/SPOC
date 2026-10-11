(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The Metal cooperative-matrix capability rule (specs/coopmat-metal-simdgroup.md),
   driven with any device's facts: no device needed. The device-side half (the
   probe kernel compiled on an Apple GPU, the SIMD width read from its
   pipeline) is exercised by sarek/tests/e2e/test_coopmat_metal_e2e.ml. *)

open Sarek_metal
module C = Sarek_coopmat

let shape8 = {C.m = 8; n = 8; k = 8}

let cfg a b c r =
  {
    C.cfg_shape = shape8;
    cfg_a = a;
    cfg_b = b;
    cfg_c = c;
    cfg_result = r;
    cfg_saturating = false;
    cfg_scope = C.Subgroup;
  }

let f16_cfg = cfg C.Float16 C.Float16 C.Float32 C.Float32

let contains ~needle s =
  let nl = String.length needle and hl = String.length s in
  let rec go i = i + nl <= hl && (String.sub s i nl = needle || go (i + 1)) in
  go 0

let test_supported () =
  let s = Metal_coopmat.support ~simd_width:32 ~probe:(Ok ()) in
  Alcotest.(check int) "one configuration" 1 (List.length s.C.ds_configs) ;
  Alcotest.(check bool)
    "it is f16*f16+f32->f32 8x8x8"
    true
    (List.exists (fun c -> c = f16_cfg) s.C.ds_configs) ;
  Alcotest.(check int) "subgroup size" 32 s.C.ds_subgroup_size ;
  Alcotest.(check int) "advertised = listed" 1 s.C.ds_advertised_count ;
  Alcotest.(check bool)
    "evidence names simdgroup_matrix"
    true
    (contains ~needle:"simdgroup_matrix" s.C.ds_evidence) ;
  Alcotest.(check bool)
    "verdict available"
    true
    (Sarek_capability.permits (C.verdict ~support:(Some s) f16_cfg))

let test_width_is_read () =
  let s = Metal_coopmat.support ~simd_width:16 ~probe:(Ok ()) in
  Alcotest.(check int)
    "subgroup size follows the device"
    16
    s.C.ds_subgroup_size

let test_no_integer_no_f16_accumulate () =
  let s = Metal_coopmat.support ~simd_width:32 ~probe:(Ok ()) in
  List.iter
    (fun (name, c) ->
      Alcotest.(check bool)
        (name ^ " not offered")
        false
        (Sarek_capability.permits (C.verdict ~support:(Some s) c)))
    [
      ("u8*u8+s32", cfg C.Uint8 C.Uint8 C.Sint32 C.Sint32);
      ("f16*f16+f16", cfg C.Float16 C.Float16 C.Float16 C.Float16);
      ("f32*f32+f32", cfg C.Float32 C.Float32 C.Float32 C.Float32);
      ("16x16x16 f16", {f16_cfg with C.cfg_shape = {C.m = 16; n = 16; k = 16}});
    ]

let test_probe_failure () =
  let s =
    Metal_coopmat.support ~simd_width:32 ~probe:(Error "no such type simdgroup")
  in
  Alcotest.(check int) "no configuration" 0 (List.length s.C.ds_configs) ;
  Alcotest.(check bool)
    "evidence carries the compiler's reason"
    true
    (contains ~needle:"no such type simdgroup" s.C.ds_evidence) ;
  Alcotest.(check bool)
    "verdict unavailable (not unknown)"
    true
    (match C.verdict ~support:(Some s) f16_cfg with
    | Sarek_capability.Unavailable _ -> true
    | _ -> false)

let test_probe_source () =
  Alcotest.(check bool)
    "probe kernel multiplies half8x8 into float8x8"
    true
    (contains ~needle:"simdgroup_multiply_accumulate" Metal_coopmat.probe_source
    && contains ~needle:"simdgroup_half8x8" Metal_coopmat.probe_source)

let () =
  Alcotest.run
    "metal_coopmat_probe"
    [
      ( "rule",
        [
          Alcotest.test_case "supported device" `Quick test_supported;
          Alcotest.test_case "SIMD width is read" `Quick test_width_is_read;
          Alcotest.test_case
            "integer, f16 accumulate, f32, 16x16 not offered"
            `Quick
            test_no_integer_no_f16_accumulate;
          Alcotest.test_case "probe failure" `Quick test_probe_failure;
          Alcotest.test_case "probe source" `Quick test_probe_source;
        ] );
    ]
