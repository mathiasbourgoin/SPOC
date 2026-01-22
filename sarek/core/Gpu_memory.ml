(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - GPU Memory Management
 *
 * Provides automatic GC integration for GPU memory:
 * - Finalizers on vectors to free device buffers when GC'd
 * - Retry-with-GC pattern on allocation failure
 * - GPU memory usage tracking
 * - Configurable auto-GC behavior
 ******************************************************************************)

(** {1 Configuration} *)

(** Whether to automatically retry allocations after GC on failure *)
let auto_gc = ref true

let set_auto_gc enabled = auto_gc := enabled

let get_auto_gc () = !auto_gc

(** {1 Memory Tracking} *)

(** Total GPU memory currently allocated (bytes), across all devices *)
let allocated_bytes = Atomic.make 0

let track_alloc bytes = Atomic.fetch_and_add allocated_bytes bytes |> ignore

let track_free bytes = Atomic.fetch_and_add allocated_bytes (-bytes) |> ignore

(** Query current GPU memory usage in bytes *)
let usage () = Atomic.get allocated_bytes

(** {1 GC Integration} *)

(** Trigger a full major GC to free unreachable GPU buffers *)
let trigger_gc () = Gc.full_major ()

(** Attempt an allocation, retrying after GC if it fails. On first failure: run
    Gc.full_major to trigger finalizers. On second failure: run Gc.compact for
    maximum cleanup. On third failure: propagate the exception. *)
let with_retry f =
  if not !auto_gc then f ()
  else
    try f ()
    with exn1 -> (
      (* First retry: trigger finalizers *)
      Gc.full_major () ;
      try f ()
      with _exn2 -> (
        (* Second retry: compact heap *)
        Gc.compact () ;
        try f () with _exn3 -> raise exn1 (* Raise original exception *)))

(** {1 Finalizer Registration} *)

(** Register a finalizer on a vector that frees all its device buffers when the
    vector is garbage collected. This is idempotent - calling free on an
    already-freed buffer is safe (backends handle double-free).

    The finalizer is registered only once per vector (on first device buffer
    allocation). *)
let register_finalizer (vec : (_, _) Vector_types.t) =
  Gc.finalise
    (fun (v : (_, _) Vector_types.t) ->
      Hashtbl.iter
        (fun _dev_id buf ->
          let (module B : Vector_types.DEVICE_BUFFER) = buf in
          let bytes = B.size * B.elem_size in
          (try B.free () with _ -> ()) ;
          track_free bytes)
        v.device_buffers ;
      Hashtbl.clear v.device_buffers)
    vec
