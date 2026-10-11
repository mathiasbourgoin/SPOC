(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The jsoo build has no worker domains: the interpreter's parallel blocks run
   inline on the caller, in submission order. As with the native pool
   (../interp/Cpu_domain_pool.ml), every task runs and the first error is
   re-raised afterwards. *)

type task = unit -> unit

let run (tasks : (task -> unit) -> unit) =
  let first_error = ref None in
  let submit task =
    try task ()
    with exn ->
      if Option.is_none !first_error then
        first_error := Some (exn, Printexc.get_raw_backtrace ())
  in
  tasks submit ;
  Option.iter
    (fun (exn, bt) -> Printexc.raise_with_backtrace exn bt)
    !first_error
