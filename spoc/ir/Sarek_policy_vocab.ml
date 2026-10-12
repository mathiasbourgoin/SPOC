(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* INTERNAL: use Sarek_policy, which re-exports all of this. The vocabulary of
   Sarek_policy, in its own file so that the pure text half of the environment
   (Sarek_policy_env) can use it without depending on the store. See
   Sarek_policy.mli for what each type and name means. *)

type operand_precision = Operands_f16 | Operands_bf16 | Operands_bf16x3

type deviation =
  | Accumulation_order
  | Operand_rounding of operand_precision
  | Single_rounding
  | Cross_kernel_contraction

type consent = Automatic_float_rewrite

type grant = Deviation of deviation | Consent of consent

type optimisation =
  | Launch_window
  | Epilogue_fusion
  | Matrix_rewrite
  | Narrowing_proof
  | Auto_unroll

type scope = Process | Kernel of string

type refusal = Blank_reason | Default_kernel_name

type evidence =
  | Proof of string
  | Granted of {grants : grant list; reason : string; lowering : string option}
  | Exact_by_construction

type decision = Applied of evidence | Refused of string

type record = {
  kernel : string;
  target : string;
  optimisation : optimisation;
  decision : decision;
}

type ignored = Generated_name | Revoked

type event =
  | Decision of record
  | Gate of {kernel : string; target : string; decision : decision}
  | Ignored_grant of {kernel : string; grant : grant; cause : ignored}

type restriction = Disable of optimisation | Revoke of grant

type explain = Explain_off | Explain_stderr | Explain_unknown of string

(* The reasons joined by "; ", each once, in order of first appearance. *)
let join_reasons reasons =
  String.concat
    "; "
    (List.rev
       (List.fold_left
          (fun acc r -> if List.mem r acc then acc else r :: acc)
          []
          reasons))

let grant_name = function
  | Deviation Accumulation_order -> "accumulation-order"
  | Deviation (Operand_rounding Operands_f16) -> "operand-rounding-f16"
  | Deviation (Operand_rounding Operands_bf16) -> "operand-rounding-bf16"
  | Deviation (Operand_rounding Operands_bf16x3) -> "operand-rounding-bf16x3"
  | Deviation Single_rounding -> "single-rounding"
  | Deviation Cross_kernel_contraction -> "cross-kernel-contraction"
  | Consent Automatic_float_rewrite -> "automatic-float-rewrite"

let optimisation_name = function
  | Launch_window -> "window"
  | Epilogue_fusion -> "epilogue"
  | Matrix_rewrite -> "matrix"
  | Narrowing_proof -> "narrowing-proof"
  | Auto_unroll -> "unroll"

let all_grants =
  [
    Deviation Accumulation_order;
    Deviation (Operand_rounding Operands_f16);
    Deviation (Operand_rounding Operands_bf16);
    Deviation (Operand_rounding Operands_bf16x3);
    Deviation Single_rounding;
    Deviation Cross_kernel_contraction;
    Consent Automatic_float_rewrite;
  ]

let all_optimisations =
  [Launch_window; Epilogue_fusion; Matrix_rewrite; Narrowing_proof; Auto_unroll]
