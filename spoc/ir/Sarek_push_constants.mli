(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_push_constants - the push-constant block layout of a kernel signature.

    One function decides where each field of the block sits, and both sides use
    it: the Vulkan backend to write the block at launch time, and the launch
    window to size a fused kernel's block before building it, so the two cannot
    drift. The block follows [Sarek_ir_glsl.gen_push_constants]: the length of
    every vector parameter first (one 4-byte [int] each, in declaration order),
    then every scalar parameter in declaration order, each field aligned to its
    own size (std430). *)

open Sarek_ir_types

(** Size and alignment of one field: 4 bytes ([int], [float]) or 8 bytes
    ([int64_t], [double]). *)
type width = W4 | W8

val bytes_of_width : width -> int

(** The width of a scalar kernel parameter of this type. Only 64-bit types are
    wider than 4 bytes; every other scalar is 4 bytes. *)
val width_of_elttype : elttype -> width

(** The block size Vulkan guarantees ([maxPushConstantsSize] is at least this),
    and the range the Vulkan backend declares in its pipeline layouts. *)
val vulkan_block_limit : int

(** [place fields] is the byte offset of each field, in order, and the size of
    the block (the end of the last field; 0 for no fields). *)
val place : width list -> int list * int

(** The fields of a kernel signature: [W4] per vector parameter, then the width
    of each scalar parameter, in declaration order. *)
val fields_of_params : decl list -> width list

(** Size in bytes of the block of the signature [params]. *)
val block_bytes : decl list -> int

(** Size in bytes of the block of the kernel that horizontal fusion
    ([Fusion_horizontal.fuse]) builds from kernels with these parameter lists:
    every member's parameters in member order, then an (offset, size) pair of
    int32 scalars per member. *)
val fused_block_bytes : decl list list -> int
