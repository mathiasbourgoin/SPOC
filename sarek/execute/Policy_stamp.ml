(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The policy a memoised, policy-dependent result was computed under
    (specs/optimisation-policy.md, cache rule).

    A stamp holds the {!Sarek_policy.version} read before the policy was read
    and the digest of the kernel's effective policy. {!current} is one atomic
    read while nothing changed; when the policy changed somewhere, it compares
    digests, so a grant for another kernel does not invalidate this one. *)

type t = {version : int Atomic.t; digest : string}

let digest_of kernel =
  Sarek_policy.effective_digest (Sarek_policy.effective ~kernel)

(** The stamp to take BEFORE computing the result: the version is read first, so
    a change racing with the computation makes the stamp stale, never wrongly
    current. *)
let make ~kernel =
  let version = Sarek_policy.version () in
  {version = Atomic.make version; digest = digest_of kernel}

(** [true] when the policy of [kernel] is the one [t] was made under. *)
let current t ~kernel =
  let now = Sarek_policy.version () in
  Atomic.get t.version = now
  ||
  if digest_of kernel = t.digest then (
    Atomic.set t.version now ;
    true)
  else false
