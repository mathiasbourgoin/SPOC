(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

type scope = All_kernels | Kernel of string

(* One-way and process-wide: entries are only ever added, so a source
   generated with barriers was generated under a policy that still holds. *)
let accepted : (scope * string) list Atomic.t = Atomic.make []

let accept_single_rounding scope ~reason =
  if String.trim reason = "" then
    invalid_arg
      "Sarek_f16_policy.accept_single_rounding: the reason must not be empty \
       (it is the record that the deviation from the interpreter was \
       understood)" ;
  let rec add () =
    let current = Atomic.get accepted in
    if List.mem_assoc scope current then ()
    else if Atomic.compare_and_set accepted current ((scope, reason) :: current)
    then ()
    else add ()
  in
  add ()

let single_rounding_reason ~kernel =
  let current = Atomic.get accepted in
  match List.assoc_opt (Kernel kernel) current with
  | Some r -> Some r
  | None -> List.assoc_opt All_kernels current
