(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Backends whose generated kernels cannot be assumed to round every operation
    as written (docs/fp-contraction-policy.md, section 14).

    A backend's plugin marks itself here when a guarantee its code generator
    relies on is lost at run time (Metal: the conformant compile options could
    not be built, so the device compiler runs in its fast-math mode). Features
    that are exact only under that guarantee, such as epilogue fusion on by
    default, consult {!is_exact} and stay off for a marked backend. The mark is
    never cleared: a process that lost the guarantee once has built kernels
    without it. *)

let inexact : (string * string) list Atomic.t = Atomic.make []

(** Record that [backend] lost its exactness guarantee, with the reason. The
    first reason for a backend is kept. *)
let mark_inexact ~(backend : string) ~(reason : string) : unit =
  let rec update () =
    let old = Atomic.get inexact in
    if List.mem_assoc backend old then ()
    else if not (Atomic.compare_and_set inexact old ((backend, reason) :: old))
    then update ()
  in
  update ()

(** The reason [backend] was marked, if it was. *)
let inexact_reason (backend : string) : string option =
  List.assoc_opt backend (Atomic.get inexact)

(** [true] unless [backend] was marked with {!mark_inexact}. *)
let is_exact (backend : string) : bool = inexact_reason backend = None
