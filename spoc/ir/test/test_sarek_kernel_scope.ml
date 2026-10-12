(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A [Kernel]-scoped grant is read by the decision point of that kernel and of
   no other (specs/optimisation-policy.md, M2). This file covers the coopmat
   verdict ([Sarek_coopmat.policy_verdict_for]): the accumulation order and the
   bf16 and bf16x3 operand rounding. The other atoms are covered where their
   decision point lives (narrowing: codegen_golden; epilogue: execute/test;
   f16 operand rounding and the automatic rewrite: execute/test).

   No process grant is made here: the process store is one-way. *)

open Sarek_coopmat

let cfg a b c r =
  {
    cfg_shape = {m = 16; n = 16; k = 16};
    cfg_a = a;
    cfg_b = b;
    cfg_c = c;
    cfg_result = r;
    cfg_saturating = false;
    cfg_scope = Subgroup;
  }

let f16_f32 = cfg Float16 Float16 Float32 Float32

let bf16_f32 = cfg BFloat16 BFloat16 Float32 Float32

let available v = v = Sarek_capability.Available

let grant_for name g =
  match Sarek_policy.grant (Sarek_policy.Kernel name) g ~reason:"test" with
  | Ok () -> ()
  | Error _ -> Alcotest.fail "grant refused"

let verdict_for name cfg =
  policy_verdict_for (Sarek_policy.budget ~kernels:[name]) cfg

let accumulation = Sarek_policy.Deviation Sarek_policy.Accumulation_order

let rounding p = Sarek_policy.Deviation (Sarek_policy.Operand_rounding p)

let test_accumulation () =
  grant_for "ks_acc_a" accumulation ;
  Alcotest.(check bool)
    "kernel A"
    true
    (available (verdict_for "ks_acc_a" f16_f32)) ;
  Alcotest.(check bool)
    "kernel B"
    false
    (available (verdict_for "ks_acc_b" f16_f32)) ;
  Alcotest.(check bool)
    "no kernel (process scope)"
    false
    (available (policy_verdict f16_f32))

let test_bf16_rounding () =
  List.iter
    (fun (p, tag) ->
      let a = "ks_" ^ tag ^ "_a" and b = "ks_" ^ tag ^ "_b" in
      grant_for a accumulation ;
      grant_for b accumulation ;
      Alcotest.(check bool)
        (tag ^ ": rounding missing")
        false
        (available (verdict_for a bf16_f32)) ;
      grant_for a (rounding p) ;
      Alcotest.(check bool)
        (tag ^ ": kernel A")
        true
        (available (verdict_for a bf16_f32)) ;
      Alcotest.(check bool)
        (tag ^ ": kernel B")
        false
        (available (verdict_for b bf16_f32)))
    [(Operands_bf16, "bf16"); (Operands_bf16x3, "bf16x3")]

(* The f16 operand rounding does not admit a bf16 product. *)
let test_f16_rounding_does_not_admit_bf16 () =
  grant_for "ks_f16_a" accumulation ;
  grant_for "ks_f16_a" (rounding Operands_f16) ;
  Alcotest.(check bool)
    "bf16 refused"
    false
    (available (verdict_for "ks_f16_a" bf16_f32)) ;
  Alcotest.(check bool)
    "f16 product available"
    true
    (available (verdict_for "ks_f16_a" f16_f32))

let () =
  let case name f = Alcotest.test_case name `Quick f in
  Alcotest.run
    "sarek_kernel_scope"
    [
      ( "coopmat verdict",
        [
          case "accumulation order" test_accumulation;
          case "bf16 operand rounding" test_bf16_rounding;
          case "f16 rounding" test_f16_rounding_does_not_admit_bf16;
        ] );
    ]
