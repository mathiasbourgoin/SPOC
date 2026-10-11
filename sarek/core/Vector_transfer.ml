(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Vector transfer helpers (split from Vector.ml)
 *
 * This module is intended to hold transfer-related helpers to reduce the size
 * of Vector.ml. Currently it only re-exports selected helpers used by other
 * modules; future work can move more logic here.
 ******************************************************************************)

(** Raw host pointers. Neither roots the host storage: the caller keeps [vec]
    reachable while the pointer or address is in use ([vec.host] is immutable,
    so a reachable vector roots its bigarray). *)
let to_ctypes_ptr : type a b. (a, b) Vector_types.t -> unit Ctypes.ptr =
 fun vec ->
  match vec.host with
  | Vector_types.Bigarray_storage ba -> Memory.borrow_bigarray_ptr ba
  | Vector_types.Custom_storage {ptr; _} -> ptr
  | Vector_types.Device_only_storage ->
      Vector_types.device_only_refusal ~op:"to_ctypes_ptr" ~id:vec.id

let host_ptr : type a b. (a, b) Vector_types.t -> nativeint =
 fun vec ->
  match vec.host with
  | Vector_types.Bigarray_storage ba ->
      Ctypes.raw_address_of_ptr (Memory.borrow_bigarray_ptr ba)
  | Vector_types.Custom_storage {ptr; _} -> Ctypes.raw_address_of_ptr ptr
  | Vector_types.Device_only_storage ->
      Vector_types.device_only_refusal ~op:"host_ptr" ~id:vec.id

(** Auto-sync callback registration *)
type sync_callback = {sync : 'a 'b. ('a, 'b) Vector_types.t -> bool}

let sync_to_cpu_callback : sync_callback option ref = ref None

let register_sync_callback (cb : sync_callback) : unit =
  sync_to_cpu_callback := Some cb

let ensure_cpu_sync (type a b) (vec : (a, b) Vector_types.t) : unit =
  if vec.Vector_types.auto_sync then
    match vec.Vector_types.location with
    | Vector_types.Stale_CPU _ -> (
        match !sync_to_cpu_callback with
        | Some cb -> ignore (cb.sync vec)
        | None -> ())
    | _ -> ()
