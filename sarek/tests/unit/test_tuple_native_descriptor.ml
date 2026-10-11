(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The descriptor expression the Native generator emits for a tuple-shape
 * vector element must be the typed, memoised constructor of Sarek_tuple_vec
 * ([pair]/[triple] applied to component values), never a lookup by mangled
 * name. A name lookup returns a descriptor at whatever type the call site
 * asks for, so a shape that does not match the element type type-checks and
 * reads the vector bytes at the wrong layout. With the typed constructor the
 * OCaml type checker relates the descriptor to the element type, and
 * memoisation still hands back the canonical descriptor the host vector
 * carries (same [Type_id]), which the second group checks.
 ******************************************************************************)

open Sarek_types
module TV = Sarek_tuple_vec
module Vector = Spoc_core.Vector
module Type_id = Sarek_ir_types.Type_id

let loc = Ppxlib.Location.none

let render ty =
  Ppxlib.Pprintast.string_of_expression
    (Sarek_native_gen_base.custom_type_id_expr ~loc ty)

let contains ~sub s =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = sub || at (i + 1))
  in
  at 0

let check_emitted ty expected =
  let code = render ty in
  Alcotest.(check bool)
    ("no lookup by name in: " ^ code)
    false
    (contains ~sub:"descriptor_by_name" code) ;
  Alcotest.(check bool)
    ("typed constructor " ^ expected ^ " in: " ^ code)
    true
    (contains ~sub:expected code)

let pair_emits_typed_constructor () =
  check_emitted
    (TTuple [TReg Float32; TPrim TInt32])
    "Sarek_tuple_vec.pair Sarek_tuple_vec.float32 Sarek_tuple_vec.int32"

let triple_emits_typed_constructor () =
  check_emitted
    (TTuple [TReg Float64; TReg Int64; TReg Float32])
    "Sarek_tuple_vec.triple Sarek_tuple_vec.float64 Sarek_tuple_vec.int64 \
     Sarek_tuple_vec.float32"

(* A tuple-shape name whose components have no Sarek_tuple_vec builder is a
   generator bug; it must stop the expansion, not emit code that fails (or
   mis-reads) at run time. *)
let unknown_component_refused () =
  match
    Sarek_native_gen_base.custom_descriptor_expr ~loc "_tup_float32_bogus"
  with
  | e ->
      Alcotest.fail
        ("expansion accepted an unknown component: "
        ^ Ppxlib.Pprintast.string_of_expression e)
  | exception Ppxlib.Location.Error _ -> ()

let same_type_id (type a) (x : a Type_id.t) (y : a Type_id.t) =
  match Type_id.equal x y with Some Type_id.Refl -> true | None -> false

(* The constructor the generated code calls returns the descriptor the host
   built, so [vec_get_custom]'s [Type_id.equal] check succeeds. *)
let constructor_returns_host_descriptor () =
  let host = TV.pair TV.int64 TV.float64 in
  let native = TV.pair TV.int64 TV.float64 in
  Alcotest.(check bool) "physically the same descriptor" true (host == native) ;
  Alcotest.(check bool)
    "same element Type_id"
    true
    (same_type_id host.Vector.type_id native.Vector.type_id) ;
  Alcotest.(check bool)
    "same vector Type_id"
    true
    (same_type_id host.Vector.vector_type_id native.Vector.vector_type_id)

(* Two shapes with the same OCaml type but different layouts (float32 and
   float64 are both [float]) are distinct descriptors. *)
let same_ocaml_type_distinct_shapes () =
  let narrow = TV.pair TV.float32 TV.float32 in
  let wide = TV.pair TV.float64 TV.float64 in
  Alcotest.(check bool) "distinct descriptors" false (narrow == wide) ;
  Alcotest.(check bool)
    "distinct element Type_ids"
    false
    (same_type_id narrow.Vector.type_id wide.Vector.type_id) ;
  Alcotest.(check (pair int int))
    "element sizes"
    (8, 16)
    (narrow.Vector.elem_size, wide.Vector.elem_size)

let () =
  Alcotest.run
    "Sarek_tuple_vec Native descriptor"
    [
      ( "generated Native code",
        [
          Alcotest.test_case "pair" `Quick pair_emits_typed_constructor;
          Alcotest.test_case "triple" `Quick triple_emits_typed_constructor;
          Alcotest.test_case
            "unknown component refused"
            `Quick
            unknown_component_refused;
        ] );
      ( "canonical descriptor",
        [
          Alcotest.test_case
            "host and Native share it"
            `Quick
            constructor_returns_host_descriptor;
          Alcotest.test_case
            "float32 and float64 shapes differ"
            `Quick
            same_ocaml_type_distinct_shapes;
        ] );
    ]
