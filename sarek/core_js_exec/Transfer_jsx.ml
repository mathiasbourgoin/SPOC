(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Transfer_jsx — jsoo transfer stubs for the Execute path.
 *
 * Execute calls Transfer.to_device and Transfer.to_cpu. The device branch
 * raises (unreachable in the empty-registry path); to_cpu is a no-op for
 * CPU-resident vectors.
 ******************************************************************************)

let _ns op =
  failwith ("Transfer_jsx: " ^ op ^ " — no jsoo backend device path until PR 3")

(** Transfer vector to device — raises (no jsoo device backend yet) *)
let to_device (type a b) (_vec : (a, b) Vector_jsx.t) (_dev : Device_jsx.t) :
    unit =
  _ns "to_device"

(** Transfer vector from device to CPU — no-op for CPU vectors *)
let to_cpu (type a b) (vec : (a, b) Vector_jsx.t) : unit =
  match vec.Vector_jsx.location with
  | Vector_jsx.CPU | Vector_jsx.Stale_GPU _ -> ()
  | Vector_jsx.GPU _ | Vector_jsx.Stale_CPU _ | Vector_jsx.Both _ ->
      _ns "to_cpu (device path)"

(** Streams: no jsoo device path yet (same status as [to_device]). The type
    exists so [Execute.ml], compiled against this shim too, type-checks its
    [?stream] parameter. *)
type stream = unit

let stream_handle (_ : stream) :
    Device_jsx.t * Spoc_framework.Framework_sig.stream_ref =
  _ns "stream_handle"

(** Mirrors [Transfer.keep]: a value rooted only for its liveness. Execute.ml is
    compiled against both, so the constructor must keep [Transfer.Keep]'s name
    and arity. *)
type keep = Keep : 'a -> keep

let retain_on_stream (_ : stream) ~keep:(_ : keep) ~vec_ids:(_ : int list) :
    unit =
  _ns "retain_on_stream"

(* No streams in the jsoo build: nothing to settle. *)
let settle_vector (_ : (_, _) Vector_jsx.t) : unit = ()
