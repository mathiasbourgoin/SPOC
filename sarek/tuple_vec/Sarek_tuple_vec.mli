(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** L13 — host-side [custom_type] builders for tuple-typed vector elements.

    Build the [Spoc_core.Vector.custom_type] needed by [Vector.create_custom]
    for a [('a * 'b) vector] (or 3-tuple), so tuples can be stored/read from the
    host with OCaml tuple literals while the kernel manipulates them on device.
    The byte layout is taken from the shared layout authority so it matches the
    device element layout exactly. Only scalar-primitive components are
    supported in this tier. *)

module Vector = Spoc_core.Vector

(** A scalar tuple component. *)
type 'a component

val float32 : float component

val float64 : float component

val int32 : int32 component

val int64 : int64 component

(** [pair a b] builds the custom type for a [(a, b)] tuple vector element. The
    result is memoised per shape: every call with the same components returns
    the physically same descriptor (same [Type_id]), which is what lets
    generated Native kernel code, calling [pair] itself, accept the host's
    vector. *)
val pair : 'a component -> 'b component -> ('a * 'b) Vector.custom_type

(** [triple a b c] builds the custom type for an [(a, b, c)] tuple vector
    element, memoised per shape like [pair]. *)
val triple :
  'a component ->
  'b component ->
  'c component ->
  ('a * 'b * 'c) Vector.custom_type

(** Byte layout of one positional field ([_0], [_1], ...) of a tuple element. *)
type field_layout = {
  fl_name : string;
  fl_elttype : Sarek_ir_types.elttype;
  fl_offset : int;
}

(** The byte layout of a tuple-shape element: total size and per-field layout,
    taken from the shared layout authority so it matches the raw element bytes
    the host composite bridge marshals. *)
type shape = {sh_name : string; sh_size : int; sh_fields : field_layout list}

(** [lookup_shape name] returns the registered byte layout of the tuple shape
    [name] (e.g. ["_tup_float32_int32"]), or [None] if no [pair]/[triple] has
    instantiated it yet. Used by the Interpreter to decode/encode tuple vector
    elements without depending on the interpreter's value model. *)
val lookup_shape : string -> shape option
