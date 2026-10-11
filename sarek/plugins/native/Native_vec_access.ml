(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Element accessors of a vector argument as seen by PPX-generated native
    kernels: every [v.(i)] in a kernel calls one of the [get_*]/[set_*] closures
    of the [Sarek_ir_types.NV] record built here. *)

open Spoc_framework

let mismatch name what =
  Native_error.(raise_error (unsupported_construct name what))

(** The scalar inside a [typed_value], or a structured error naming [name]. *)
let primitive name (tv : Typed_value.typed_value) =
  match tv with
  | Typed_value.TV_Scalar (Typed_value.SV ((module S), x)) -> S.to_primitive x
  | _ -> mismatch name "not a scalar"

let float_of_primitive name = function
  | Typed_value.PFloat f -> f
  | Typed_value.PInt32 n -> Int32.to_float n
  | Typed_value.PInt64 n -> Int64.to_float n
  | _ -> mismatch name "incompatible type"

let int32_of_primitive name = function
  | Typed_value.PInt32 n -> n
  | Typed_value.PInt64 n -> Int64.to_int32 n
  | Typed_value.PFloat f -> Int32.of_float f
  | _ -> mismatch name "incompatible type"

let int64_of_primitive name = function
  | Typed_value.PInt64 n -> n
  | Typed_value.PInt32 n -> Int64.of_int32 n
  | Typed_value.PFloat f -> Int64.of_float f
  | _ -> mismatch name "incompatible type"

let scalar (type a) (module S : Typed_value.SCALAR_TYPE with type t = a) (x : a)
    =
  Typed_value.TV_Scalar (Typed_value.SV ((module S), x))

(** Accessors through the vector's [EXEC_VECTOR] interface: every read goes
    through a [typed_value], so a kernel may read any scalar vector as any
    scalar type (e.g. an int32 vector with [get_f32]). *)
let native_vec_of_exec_vector (module V : Typed_value.EXEC_VECTOR) :
    Sarek_ir_types.native_arg =
  let get_f32 i =
    float_of_primitive "get_f32" (primitive "get_f32" (V.get i))
  in
  let get_f64 i =
    float_of_primitive "get_f64" (primitive "get_f64" (V.get i))
  in
  let get_i32 i =
    int32_of_primitive "get_i32" (primitive "get_i32" (V.get i))
  in
  let get_i64 i =
    int64_of_primitive "get_i64" (primitive "get_i64" (V.get i))
  in
  Sarek_ir_types.NA_Vec
    (Sarek_ir_types.NV
       {
         length = V.length;
         elem_size = V.elem_size;
         type_name = V.type_name;
         type_id = V.type_id;
         get_f32;
         set_f32 =
           (fun i f -> V.set i (scalar (module Typed_value.Float32_type) f));
         get_f64;
         set_f64 =
           (fun i f -> V.set i (scalar (module Typed_value.Float64_type) f));
         get_i32;
         set_i32 =
           (fun i n -> V.set i (scalar (module Typed_value.Int32_type) n));
         get_i64;
         set_i64 =
           (fun i n -> V.set i (scalar (module Typed_value.Int64_type) n));
         get_typed = V.get_typed;
         set_typed = V.set_typed;
         underlying_type_id = V.underlying_type_id;
         underlying = V.underlying;
       })
