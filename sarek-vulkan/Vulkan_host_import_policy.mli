(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Policy for importing host memory into a Vulkan device (Sarek #503). *)

type decision =
  | Allowed
  | Not_opted_in  (** Default: the caller did not set {!opt_in_variable}. *)
  | Amdgpu_refused
      (** Opted in, but the device is an AMD driver (kernel oops, #503). *)

(** [SPOC_VULKAN_HOST_IMPORT]: set to [1] to allow the import. *)
val opt_in_variable : string

(** Set to [1] together with {!opt_in_variable} to allow the import on AMD
    drivers anyway. For disposable sessions only. *)
val amdgpu_override_variable : string

(** PCI vendor 0x1002, or Vulkan driver ID AMD proprietary / open source / RADV.
*)
val is_amd : vendor_id:int -> driver_id:int -> bool

(** Pure decision. *)
val decide :
  opted_in:bool ->
  amdgpu_override:bool ->
  vendor_id:int ->
  driver_id:int ->
  decision

(** {!decide} with the two switches read from the environment ([1] enables). *)
val of_environment : vendor_id:int -> driver_id:int -> decision

(** The message of the refusal, [None] when {!Allowed}. *)
val refusal_reason : decision -> string option
