(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * L13 — Tuple-typed vectors (host side).
 *
 * OCaml tuples are structural, so there is no declaration site to hang
 * [@@sarek.type] off and no PPX-generated [<name>_custom] to feed to
 * [Vector.create_custom]. This module builds the missing [custom_type] values
 * at runtime by composition: a tuple element is a packed record with
 * positional fields [_0], [_1], ... and its byte layout is taken from the
 * shared layout authority [Sarek_ir_layout], byte-for-byte identical to the
 * layout the device codegen computes for the synthesized record
 * ([Sarek_lower_ir.slot_elttype_of_typ]). Host [get]/[set] therefore marshal
 * to/from exactly the bytes the kernel reads and writes on device.
 *
 * Scope (this tier): tuples of two or three scalar-primitive components,
 * running on CUDA/PTX, OpenCL, Vulkan and Native. Native shares the [Type_id]
 * with generated kernel code through the process-wide registry below. The
 * Interpreter path needs value-model unification and is a follow-up. (Design
 * note L13-tuple-vectors.md is NOT part of this repository - unpublished
 * working directory; do not look for it in a clone.)
 ******************************************************************************)

module Vector = Spoc_core.Vector
module CH = Spoc_core.Vector.Custom_helpers
module Layout = Sarek_ir_layout
module Type_id = Sarek_ir_types.Type_id

(** Identity of a scalar component kind. Distinct constructors are distinct
    element layouts even when they share an OCaml type ([F32] and [F64] are both
    [float]). *)
type _ tag =
  | F32 : float tag
  | F64 : float tag
  | I32 : int32 tag
  | I64 : int64 tag

let tag_name : type a. a tag -> string = function
  | F32 -> "float32"
  | F64 -> "float64"
  | I32 -> "int32"
  | I64 -> "int64"

(* Registry index of a tag. Invariant (checked at module initialisation
   below): a bijection from [all_tags] onto [0 .. tag_count - 1]. *)
let tag_index : type a. a tag -> int = function
  | F32 -> 0
  | F64 -> 1
  | I32 -> 2
  | I64 -> 3

type any_tag = Any_tag : 'a tag -> any_tag

(* Successor in the enumeration of all tags, [None] after the last. Exhaustive,
   so a new constructor cannot be added without placing it here. *)
let next_tag : any_tag -> any_tag option = function
  | Any_tag F32 -> Some (Any_tag F64)
  | Any_tag F64 -> Some (Any_tag I32)
  | Any_tag I32 -> Some (Any_tag I64)
  | Any_tag I64 -> None

let all_tags : any_tag list =
  let rec from tag = tag :: Option.fold ~none:[] ~some:from (next_tag tag) in
  from (Any_tag F32)

let tag_count = List.length all_tags

let () =
  let indices = List.map (fun (Any_tag t) -> tag_index t) all_tags in
  if List.sort compare indices <> List.init tag_count Fun.id then
    failwith
      "Sarek_tuple_vec: tag_index is not a bijection from all_tags onto [0, \
       tag_count)"

(* [Some Equal] exactly when both tags are the same constructor. One row per
   constructor and no wildcard on [x], so a new constructor must be added. *)
let tag_eq : type a b. a tag -> b tag -> (a, b) Type.eq option =
 fun x y ->
  match x with
  | F32 -> ( match y with F32 -> Some Type.Equal | F64 | I32 | I64 -> None)
  | F64 -> ( match y with F64 -> Some Type.Equal | F32 | I32 | I64 -> None)
  | I32 -> ( match y with I32 -> Some Type.Equal | F32 | F64 | I64 -> None)
  | I64 -> ( match y with I64 -> Some Type.Equal | F32 | F64 | I32 -> None)

(** A scalar component of a tuple element: its kind, its IR element type (for
    layout) and the typed raw-pointer read/write primitives used to marshal it
    at a byte offset. *)
type 'a component = {
  c_tag : 'a tag;
  c_elttype : Sarek_ir_types.elttype;
  c_read : unit Ctypes.ptr -> int -> 'a;
  c_write : unit Ctypes.ptr -> int -> 'a -> unit;
}

let float32 : float component =
  {
    c_tag = F32;
    c_elttype = Sarek_ir_types.TFloat32;
    c_read = CH.read_float32;
    c_write = CH.write_float32;
  }

let float64 : float component =
  {
    c_tag = F64;
    c_elttype = Sarek_ir_types.TFloat64;
    c_read = CH.read_float64;
    c_write = CH.write_float64;
  }

let int32 : int32 component =
  {
    c_tag = I32;
    c_elttype = Sarek_ir_types.TInt32;
    c_read = CH.read_int32;
    c_write = CH.write_int32;
  }

let int64 : int64 component =
  {
    c_tag = I64;
    c_elttype = Sarek_ir_types.TInt64;
    c_read = CH.read_int64;
    c_write = CH.write_int64;
  }

(** Mangled record name for a component tag list; kept in sync with
    [Sarek_lower_ir.tuple_record_name] so host and device agree on the element
    identity. *)
let mangled_name (tags : string list) : string =
  "_tup" ^ String.concat "" (List.map (fun t -> "_" ^ t) tags)

let field_name i = Printf.sprintf "_%d" i

let layout_of fields name =
  match Layout.record_layout ~type_name:name fields with
  | Ok rl -> rl
  | Error e ->
      failwith
        ("Sarek_tuple_vec: could not lay out " ^ name ^ ": "
        ^ Layout.layout_error_message e)

let offset_of rl fname =
  match List.assoc_opt fname rl.Layout.rl_fields with
  | Some off -> off
  | None -> failwith ("Sarek_tuple_vec: missing field offset for " ^ fname)

(* A tuple shape, typed by the OCaml element type it describes. *)
type _ key =
  | Pair : 'a tag * 'b tag -> ('a * 'b) key
  | Triple : 'a tag * 'b tag * 'c tag -> ('a * 'b * 'c) key

let key_name : type a. a key -> string = function
  | Pair (a, b) -> mangled_name [tag_name a; tag_name b]
  | Triple (a, b, c) -> mangled_name [tag_name a; tag_name b; tag_name c]

(* Registry slot of a shape: pairs occupy [0 .. tag_count^2 - 1], triples the
   [tag_count^3] slots after them. Distinct shapes have distinct slots. *)
(* [tag_index t], refused unless it lies in [0, tag_count). The start-up check
   covers only the tags reachable through [next_tag]; a constructor left out of
   that chain is refused here instead of addressing a slot outside its range.
   If its index duplicates a reachable tag's, the slot it shares holds an
   entry for another shape and [recover] refuses it. *)
let slot_index : type a. a tag -> int =
 fun t ->
  let i = tag_index t in
  if i >= 0 && i < tag_count then i
  else
    failwith
      (Printf.sprintf
         "Sarek_tuple_vec: tag %s has index %d, outside [0, %d): it is not in \
          the next_tag enumeration"
         (tag_name t)
         i
         tag_count)

let pair_slot a b = (slot_index a * tag_count) + slot_index b

let triple_slot a b c =
  (tag_count * tag_count)
  + (((slot_index a * tag_count) + slot_index b) * tag_count)
  + slot_index c

let key_slot : type a. a key -> int = function
  | Pair (a, b) -> pair_slot a b
  | Triple (a, b, c) -> triple_slot a b c

let slot_count = (tag_count * tag_count) + (tag_count * tag_count * tag_count)

(* [Some Equal] exactly when [k] is the pair shape of [a] and [b]. *)
let is_pair : type a b k. a tag -> b tag -> k key -> (k, a * b) Type.eq option =
 fun a b k ->
  match k with
  | Pair (a', b') -> (
      match (tag_eq a' a, tag_eq b' b) with
      | Some Type.Equal, Some Type.Equal -> Some Type.Equal
      | _ -> None)
  | Triple _ -> None

(* [Some Equal] exactly when [k] is the triple shape of [a], [b] and [c]. *)
let is_triple : type a b c k.
    a tag -> b tag -> c tag -> k key -> (k, a * b * c) Type.eq option =
 fun a b c k ->
  match k with
  | Triple (a', b', c') -> (
      match (tag_eq a' a, tag_eq b' b, tag_eq c' c) with
      | Some Type.Equal, Some Type.Equal, Some Type.Equal -> Some Type.Equal
      | _ -> None)
  | Pair _ -> None

(* [Some Equal] exactly when both keys denote the same shape. *)
let key_eq : type a b. a key -> b key -> (a, b) Type.eq option =
 fun x y ->
  match y with
  | Pair (a, b) -> is_pair a b x
  | Triple (a, b, c) -> is_triple a b c x

(* A registered descriptor together with the shape it was built for. *)
type entry = Entry : 'a key * 'a Vector.custom_type -> entry

(* Process-global registry of canonical descriptors, one slot per shape
   ([key_slot]). A shape is instantiated once and shared: the host builds a
   vector from it and generated Native code calls the same [pair]/[triple] on
   every element access, so both sides carry the identical [Type_id] token that
   [vec_get_custom] compares with [Type_id.equal]. Invariants: slot
   [key_slot k] holds [None] or [Some (Entry (k', _))] with
   [key_eq k' k = Some Equal]; a slot changes at most once, from [None], under
   [registry_mutex]. Reads are lock-free [Atomic.get]s, which also make the
   descriptor's construction visible to the reader. *)
let registry : entry option Atomic.t array =
  Array.init slot_count (fun _ -> Atomic.make None)

(* Serialises build-and-publish: two first-time builders of one shape must not
   each publish a fresh [Type_id], or the loser's vector would fail the
   [Type_id.equal] check against the canonical descriptor. Also guards
   [shape_registry]. *)
let registry_mutex = Mutex.create ()

let recover : type a. a key -> entry -> a Vector.custom_type =
 fun key (Entry (found, custom)) ->
  match key_eq found key with
  | Some Type.Equal -> custom
  | None ->
      failwith
        ("Sarek_tuple_vec: registry slot "
        ^ string_of_int (key_slot key)
        ^ " holds shape " ^ key_name found ^ ", not " ^ key_name key)

let memoize : type a.
    a key -> (unit -> a Vector.custom_type) -> a Vector.custom_type =
 fun key build ->
  let cell = registry.(key_slot key) in
  match Atomic.get cell with
  | Some entry -> recover key entry
  | None ->
      Mutex.protect registry_mutex (fun () ->
          match Atomic.get cell with
          | Some entry -> recover key entry
          | None ->
              let custom = build () in
              Atomic.set cell (Some (Entry (key, custom))) ;
              custom)

(* Value-model-neutral description of a tuple element's byte layout, keyed by
   the mangled shape name. The Interpreter (which represents a tuple element as
   a positional record of boxed scalars, not as a native OCaml tuple) resolves a
   shape here to decode the raw element bytes exposed by the host composite
   bridge into per-field scalar values and re-encode them on writeback. This
   keeps the dependency one-directional: the Interpreter depends on this module,
   never the reverse. *)
type field_layout = {
  fl_name : string;
  fl_elttype : Sarek_ir_types.elttype;
  fl_offset : int;
}

type shape = {sh_name : string; sh_size : int; sh_fields : field_layout list}

let shape_registry : (string, shape) Hashtbl.t = Hashtbl.create 16

(** [lookup_shape name] returns the registered byte layout of the tuple shape
    [name] (e.g. ["_tup_float32_int32"]), or [None] if no [pair]/[triple] has
    built it yet. *)
let lookup_shape (name : string) : shape option =
  (* Locked: register_shape_of writes under [registry_mutex]; an unlocked read
     could race a concurrent first-time registration on the mutable Hashtbl
     (interpreter workers look shapes up while the host may still register). *)
  Mutex.protect registry_mutex (fun () -> Hashtbl.find_opt shape_registry name)

(* Build a [shape] from a laid-out field list and record it. Called only from
   within the [memoize] critical section (which already holds
   [registry_mutex]), so it must not re-acquire the lock. *)
let register_shape_of rl name fields =
  let sh_fields =
    List.map
      (fun (fname, elttype) ->
        {fl_name = fname; fl_elttype = elttype; fl_offset = offset_of rl fname})
      fields
  in
  if not (Hashtbl.mem shape_registry name) then
    Hashtbl.replace
      shape_registry
      name
      {sh_name = name; sh_size = rl.Layout.rl_size; sh_fields}

let build_pair (a : 'a component) (b : 'b component) () :
    ('a * 'b) Vector.custom_type =
  let name = mangled_name [tag_name a.c_tag; tag_name b.c_tag] in
  let fields = [(field_name 0, a.c_elttype); (field_name 1, b.c_elttype)] in
  let rl = layout_of fields name in
  register_shape_of rl name fields ;
  let size = rl.Layout.rl_size in
  let o0 = offset_of rl (field_name 0) in
  let o1 = offset_of rl (field_name 1) in
  {
    Vector.elem_size = size;
    type_id = Type_id.create ();
    vector_type_id = Type_id.create ();
    name;
    get =
      (fun ptr idx ->
        let base = idx * size in
        (a.c_read ptr (base + o0), b.c_read ptr (base + o1)));
    set =
      (fun ptr idx (x, y) ->
        let base = idx * size in
        a.c_write ptr (base + o0) x ;
        b.c_write ptr (base + o1) y);
    (* Same [fields] that produced [rl], hence [size] and [o0]/[o1] above. *)
    ir_fields = Some fields;
  }

let build_triple (a : 'a component) (b : 'b component) (c : 'c component) () :
    ('a * 'b * 'c) Vector.custom_type =
  let name =
    mangled_name [tag_name a.c_tag; tag_name b.c_tag; tag_name c.c_tag]
  in
  let fields =
    [
      (field_name 0, a.c_elttype);
      (field_name 1, b.c_elttype);
      (field_name 2, c.c_elttype);
    ]
  in
  let rl = layout_of fields name in
  register_shape_of rl name fields ;
  let size = rl.Layout.rl_size in
  let o0 = offset_of rl (field_name 0) in
  let o1 = offset_of rl (field_name 1) in
  let o2 = offset_of rl (field_name 2) in
  {
    Vector.elem_size = size;
    type_id = Type_id.create ();
    vector_type_id = Type_id.create ();
    name;
    get =
      (fun ptr idx ->
        let base = idx * size in
        ( a.c_read ptr (base + o0),
          b.c_read ptr (base + o1),
          c.c_read ptr (base + o2) ));
    set =
      (fun ptr idx (x, y, z) ->
        let base = idx * size in
        a.c_write ptr (base + o0) x ;
        b.c_write ptr (base + o1) y ;
        c.c_write ptr (base + o2) z);
    (* Same [fields] that produced [rl], hence [size] and [o0]/[o1]/[o2]. *)
    ir_fields = Some fields;
  }

(* Generated Native code calls [pair]/[triple] on every element access, so the
   hit path is a lock-free read that allocates nothing; [memoize] handles the
   first build and reports a slot holding another shape. *)

(** [pair a b] is the [custom_type] for a [(a, b)] tuple vector element. *)
let pair : type a b. a component -> b component -> (a * b) Vector.custom_type =
 fun a b ->
  match Atomic.get registry.(pair_slot a.c_tag b.c_tag) with
  | Some (Entry (found, custom)) -> (
      match is_pair a.c_tag b.c_tag found with
      | Some Type.Equal -> custom
      | None -> recover (Pair (a.c_tag, b.c_tag)) (Entry (found, custom)))
  | None -> memoize (Pair (a.c_tag, b.c_tag)) (build_pair a b)

(** [triple a b c] is the [custom_type] for a [(a, b, c)] tuple vector element.
*)
let triple : type a b c.
    a component -> b component -> c component -> (a * b * c) Vector.custom_type
    =
 fun a b c ->
  match Atomic.get registry.(triple_slot a.c_tag b.c_tag c.c_tag) with
  | Some (Entry (found, custom)) -> (
      match is_triple a.c_tag b.c_tag c.c_tag found with
      | Some Type.Equal -> custom
      | None ->
          recover (Triple (a.c_tag, b.c_tag, c.c_tag)) (Entry (found, custom)))
  | None -> memoize (Triple (a.c_tag, b.c_tag, c.c_tag)) (build_triple a b c)
