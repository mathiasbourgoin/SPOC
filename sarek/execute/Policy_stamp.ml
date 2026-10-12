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

(* How many digests were computed: tests check that a launch under an unchanged
   policy computes none. *)
let digests = Atomic.make 0

let digests_computed () = Atomic.get digests

let digest_of kernel =
  Atomic.incr digests ;
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

(** The value stored under a key accepted by [matches], if its stamp is current
    for [kernel]; entries are [(key, (value, stamp))]. *)
let find_current ~kernel ~matches entries =
  List.find_map
    (fun (k, (v, stamp)) ->
      if matches k && current stamp ~kernel then Some v else None)
    entries

(** [entries] with the entry for the key replaced (a stale one is dropped, not
    shadowed). *)
let install ~matches entries key value stamp =
  (key, (value, stamp)) :: List.filter (fun (k, _) -> not (matches k)) entries
