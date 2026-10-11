(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Synchronisation-point hook for deferred kernel launches
    (specs/launch-window.md).

    [Execute] may keep launches pending in a window instead of handing them to
    the driver at once. Every operation in this library that moves data between
    host and device, or orders or observes device work, calls {!flush_pending}
    before it acts, so it sees every launch made before it, exactly as if the
    launches had been issued immediately.

    This module only holds the hook: it knows nothing about launches. The window
    installs its flusher with {!set_flusher} and keeps {!pending} up to date.
    When nothing is pending a call costs one atomic read.

    Re-entrancy: issuing a launch must never re-enter the flusher. A
    domain-local flag is set while this domain runs the flusher, and a nested
    {!flush_pending} on that domain returns at once (the window being flushed
    has already been taken). Another domain calling {!flush_pending} meanwhile
    blocks on the window's own lock, not here. *)

let pending : int Atomic.t = Atomic.make 0

let flusher : (unit -> unit) ref = ref (fun () -> ())

let flushing : bool Domain.DLS.key = Domain.DLS.new_key (fun () -> false)

let touching : (device_id:int -> buf:int -> bool) ref =
  ref (fun ~device_id:_ ~buf:_ -> true)

(** Install the window's flusher, and its test "does a pending launch on
    [device_id] bind the buffer of vector [buf]". Called once by [Execute] at
    initialisation. *)
let set_flusher ?touches (f : unit -> unit) : unit =
  flusher := f ;
  Option.iter (fun t -> touching := t) touches

(** Run [f] with this domain marked as flushing, so a hook reached from inside
    [f] does not recurse into the flusher. *)
let while_flushing (f : unit -> 'a) : 'a =
  let was = Domain.DLS.get flushing in
  Domain.DLS.set flushing true ;
  Fun.protect ~finally:(fun () -> Domain.DLS.set flushing was) f

(** Issue every pending launch, on every device. Called by synchronisation
    points before they act. *)
let flush_pending () : unit =
  if Atomic.get pending > 0 && not (Domain.DLS.get flushing) then !flusher ()

(** Issue every pending launch if one of them, on [device_id], binds the buffer
    of vector [buf] (a vector id). For an operation that only overwrites that
    buffer (an upload): pending launches that do not bind it are unaffected by
    it, so they can stay pending and still be fused. *)
let flush_touching ~(device_id : int) ~(buf : int) : unit =
  if
    Atomic.get pending > 0
    && (not (Domain.DLS.get flushing))
    && !touching ~device_id ~buf
  then !flusher ()
