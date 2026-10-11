(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Shared checks of the matrix-pattern corpus (specs/matrix-pattern-advice.md). *)

module P = Sarek_matrix_pattern
module Adv = Sarek_matrix_advice

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> Alcotest.fail "kernel has no IR"

let only_gemm name kernel =
  match P.analyze (ir_of kernel) with
  | [P.Gemm g] -> g
  | findings ->
      Alcotest.failf
        "%s: expected exactly one GEMM finding, got %d findings"
        name
        (List.length findings)

let layout = Alcotest.testable (Fmt.of_to_string P.layout_name) ( = )

let check_roles name kernel ~a ~b ~c ~layouts:(la, lb, lc) ~batch =
  let g = only_gemm name kernel in
  Alcotest.(check string) (name ^ ": A") a g.a.array ;
  Alcotest.(check string) (name ^ ": B") b g.b.array ;
  Alcotest.(check string) (name ^ ": C") c g.c.array ;
  Alcotest.check layout (name ^ ": A layout") la g.a.layout ;
  Alcotest.check layout (name ^ ": B layout") lb g.b.layout ;
  Alcotest.check layout (name ^ ": C layout") lc g.c.layout ;
  Alcotest.(check int) (name ^ ": batch atoms") batch (List.length g.batch) ;
  g

let case_name = function
  | Adv.F16_inputs -> "f16"
  | Adv.F32_inputs -> "f32"
  | Adv.Int32_inputs -> "int32"
  | Adv.No_case -> "none"

let check_case name g expected =
  Alcotest.(check string)
    (name ^ ": element case")
    expected
    (case_name (Adv.advise g).case)

let reasons kernel =
  List.filter_map
    (function P.Near_miss nm -> Some nm.nm_reason | P.Gemm _ -> None)
    (P.analyze (ir_of kernel))

let check_rejected name kernel expected =
  let findings = P.analyze (ir_of kernel) in
  Alcotest.(check bool)
    (name ^ ": no GEMM finding")
    false
    (List.exists
       (function P.Gemm _ -> true | P.Near_miss _ -> false)
       findings) ;
  match expected with
  | None ->
      Alcotest.(check int)
        (name ^ ": no reduction loop")
        0
        (List.length findings)
  | Some pred ->
      Alcotest.(check bool)
        (name ^ ": reason "
        ^ String.concat "; " (List.map P.reason_text (reasons kernel)))
        true
        (List.exists pred (reasons kernel))
