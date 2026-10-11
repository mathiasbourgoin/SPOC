(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Per-backend limits on the signature of a launch the window may issue: kernel
    argument slots and push-constant bytes. A backend with no entry is unbounded
    as far as the window is concerned. Depends only on the IR and Stdlib, like
    {!Launch_window}. *)

open Sarek_ir_types

(* Kernel argument slots per backend, as an immutable association list. *)
let slot_limits : (string * int) list Atomic.t = Atomic.make [("Metal", 31)]

(** Kernel argument slots a launch's signature may use on [backend]: Metal binds
    every argument at one of 31 [[buffer(n)]] indices. A backend with no limit
    configured here is unbounded as far as the window is concerned. *)
let arg_slot_limit (backend : string) : int =
  Option.value
    ~default:max_int
    (List.assoc_opt backend (Atomic.get slot_limits))

(* Replace [backend]'s entry of an association-list limit table. *)
let set_limit table ~backend n =
  let rec update () =
    let old = Atomic.get table in
    let next = (backend, n) :: List.remove_assoc backend old in
    if not (Atomic.compare_and_set table old next) then update ()
  in
  update ()

(** Set the argument slot limit of [backend] (for tests and for backends whose
    limit becomes known). *)
let set_arg_slot_limit ~(backend : string) (n : int) : unit =
  set_limit slot_limits ~backend n

(* Push-constant block bytes per backend, as an immutable association list. *)
let push_limits : (string * int) list Atomic.t =
  Atomic.make [("Vulkan", Sarek_push_constants.vulkan_block_limit)]

(** Bytes of push constants a launch's signature may use on [backend]: Vulkan
    declares one 128-byte range in every pipeline layout and refuses a launch
    whose block is larger. A backend with no limit configured here is unbounded
    as far as the window is concerned. *)
let push_constant_limit (backend : string) : int =
  Option.value
    ~default:max_int
    (List.assoc_opt backend (Atomic.get push_limits))

(** Set the push-constant byte limit of [backend] (for tests and for backends
    whose limit becomes known). *)
let set_push_constant_limit ~(backend : string) (n : int) : unit =
  set_limit push_limits ~backend n

(** [true] when a group whose members have these parameter lists keeps its
    kernel's push-constant block within [backend]'s limit: the fused kernel's
    block for two or more members, the launch's own for one. Answers at once
    when the backend has no limit. *)
let push_bytes_ok ~(backend : string) (members : decl list list) : bool =
  let limit = push_constant_limit backend in
  if limit = max_int then true
  else
    match members with
    | [] -> true
    | [params] -> Sarek_push_constants.block_bytes params <= limit
    | _ -> Sarek_push_constants.fused_block_bytes members <= limit
