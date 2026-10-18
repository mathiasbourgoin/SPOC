(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Whether a device shares system memory with the host (an APU or integrated
    GPU), as the backend that owns the device knows it. See the implementation
    for why this is a hook and not a field. The framework name must be the
    backend's [name] ("HIP", "CUDA/PTX", "CUDA/C"): a mismatch is read as "not
    known". *)

(** [set ~framework f] installs [f], which maps a backend-local device index to
    "this device shares host memory". A second call for the same framework
    replaces the first. [f] must be safe to call from any domain. *)
val set : framework:string -> (int -> bool) -> unit

(** [Some true] shared, [Some false] discrete, [None] when no query is installed
    or it raised. [Out_of_memory], [Stack_overflow] and [Sys.Break] are
    re-raised. *)
val query : framework:string -> backend_id:int -> bool option
