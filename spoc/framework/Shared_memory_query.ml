(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Whether a device shares system memory with the host (an APU or integrated
    GPU), as the backend that owns the device knows it.

    {!Spoc_framework.Device_type.t} carries no such flag, and adding a field to
    its capabilities record would break every out-of-tree backend that builds
    one. Backends that can answer therefore install a query here, once, when
    they register; a consumer asks by framework name and backend-local device
    index. A framework that installed nothing answers [None]: "not known", never
    "discrete". *)

let queries : (string, int -> bool) Hashtbl.t = Hashtbl.create 4

let lock = Mutex.create ()

(** [set ~framework f] installs [f], which maps a backend-local device index to
    "this device shares host memory". A second call for the same framework
    replaces the first. [f] must be safe to call from any domain; an exception
    it raises is a "not known" answer, see {!query}. *)
let set ~(framework : string) (f : int -> bool) : unit =
  Mutex.protect lock (fun () -> Hashtbl.replace queries framework f)

(** [query ~framework ~backend_id] is [Some true] for a device that shares host
    memory, [Some false] for a discrete one, [None] when the framework installed
    no query or its query raised. *)
let query ~(framework : string) ~(backend_id : int) : bool option =
  match Mutex.protect lock (fun () -> Hashtbl.find_opt queries framework) with
  | None -> None
  | Some f -> (
      try Some (f backend_id) with
      | (Out_of_memory | Stack_overflow | Sys.Break) as e -> raise e
      | _ -> None)
