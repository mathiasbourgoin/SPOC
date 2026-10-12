(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

type scope = All_kernels | Kernel of string

type refusal = Sarek_policy.refusal = Blank_reason | Default_kernel_name

let refusal_message = function
  | Blank_reason ->
      "Sarek_f16_policy.accept_single_rounding: the reason must not be empty \
       (it is the record that the deviation from the interpreter was \
       understood)"
  | Default_kernel_name ->
      "Sarek_f16_policy.accept_single_rounding: Kernel \"sarek_kern\" is the \
       name of every unnamed [%kernel], library kernels included; name the \
       kernel, or accept All_kernels"

(* The grant itself lives in {!Sarek_policy}: one-way and process-wide, so a
   source generated with barriers was generated under a policy that still
   holds. *)
let policy_scope = function
  | All_kernels -> Sarek_policy.Process
  | Kernel name -> Sarek_policy.Kernel name

let single_rounding = Sarek_policy.Deviation Sarek_policy.Single_rounding

let accept_single_rounding scope ~reason =
  Sarek_policy.grant (policy_scope scope) single_rounding ~reason

let single_rounding_reason ~kernel = Sarek_policy.reason ~kernel single_rounding
