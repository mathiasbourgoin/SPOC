(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

type scope = All_kernels | Kernel of string

(* One-way and process-wide: entries are only ever added, so a source
   generated with barriers was generated under a policy that still holds. *)
let accepted : (scope * string) list Atomic.t = Atomic.make []

type refusal = Blank_reason | Default_kernel_name

(* The name Sarek_lower_ir gives every [%kernel] written without one. *)
let default_kernel_name = "sarek_kern"

let refusal_message = function
  | Blank_reason ->
      "Sarek_f16_policy.accept_single_rounding: the reason must not be empty \
       (it is the record that the deviation from the interpreter was \
       understood)"
  | Default_kernel_name ->
      "Sarek_f16_policy.accept_single_rounding: Kernel \"sarek_kern\" is the \
       name of every unnamed [%kernel], library kernels included; name the \
       kernel, or accept All_kernels"

let rec add scope reason =
  let current = Atomic.get accepted in
  if List.mem_assoc scope current then ()
  else if
    not (Atomic.compare_and_set accepted current ((scope, reason) :: current))
  then add scope reason

let accept_single_rounding scope ~reason =
  if String.trim reason = "" then Error Blank_reason
  else if scope = Kernel default_kernel_name then Error Default_kernel_name
  else Ok (add scope reason)

let single_rounding_reason ~kernel =
  let current = Atomic.get accepted in
  match List.assoc_opt (Kernel kernel) current with
  | Some r -> Some r
  | None -> List.assoc_opt All_kernels current
