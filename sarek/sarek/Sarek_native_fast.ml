(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Direct element access for the vector parameters of PPX-generated native
    kernels.

    A native kernel reads [v.(i)] through the accessor object of its argument
    ([v#get i]). For a float32 or int32 vector held in a Bigarray, the PPX
    instead binds the record built here once per work-item and generates

    {v
    if i >= 0 && i < f.len then Bigarray.Array1.unsafe_get f.ba i else v#get i
    v}

    The result is the value, the bounds check and the error [Vector.get] give,
    with no method call, closure call or boxing on the way.

    Exactness contract (what makes this a pure speed-up):
    - the condition uses the vector's [length], exactly the test of
      [Vector.get]/[Vector.set]; an index outside it takes the original accessor
      and so raises the original [Invalid_argument] with the original message;
    - a vector that is not eligible (other element type, non-Bigarray storage, a
      Bigarray shorter than [length]) gets [len = 0]: every access takes the
      original accessor, which is the previous behaviour;
    - [Vector.ensure_cpu_sync] runs once, when the record is built (a launch
      owns its vector arguments), instead of at every element;
    - a store marks the device copy stale with {!mark_written}, the transition
      [Vector.set] performs. *)

module Vector = Spoc_core.Vector
module Type_id = Sarek_ir_types.Type_id

type f32 = {
  f32_ba : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t;
  f32_vec : (float, Bigarray.float32_elt) Vector.t option;
  f32_len : int;
}

type i32 = {
  i32_ba : (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t;
  i32_vec : (int32, Bigarray.int32_elt) Vector.t option;
  i32_len : int;
}

let empty_f32 =
  {
    f32_ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 0;
    f32_vec = None;
    f32_len = 0;
  }

let empty_i32 =
  {
    i32_ba = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout 0;
    i32_vec = None;
    i32_len = 0;
  }

(** The device copy is stale after a host write: the transition of [Vector.set],
    applied to a vector already synchronised to the host. *)
let mark_written (type a b) (vec : (a, b) Vector.t) : unit =
  match vec.Vector.location with
  | Vector.Both d | Vector.GPU d -> vec.Vector.location <- Vector.Stale_GPU d
  | Vector.CPU | Vector.Stale_CPU _ | Vector.Stale_GPU _ -> ()

let mark_written_f32 (f : f32) : unit =
  match f.f32_vec with Some vec -> mark_written vec | None -> ()

let mark_written_i32 (f : i32) : unit =
  match f.i32_vec with Some vec -> mark_written vec | None -> ()

(** The fast record of a float32 vector argument, or {!empty_f32}. Synchronises
    an eligible vector to the host (once per launch). *)
let f32_of_native_arg (arg : Sarek_ir_types.native_arg) : f32 =
  match arg with
  | Sarek_ir_types.NA_Vec (Sarek_ir_types.NV nv) -> (
      match
        Type_id.equal nv.underlying_type_id Vector.float32_vector_type_id
      with
      | Some Type_id.Refl -> (
          let vec : (float, Bigarray.float32_elt) Vector.t = nv.underlying in
          match vec.Vector.host with
          | Vector.Bigarray_storage ba
            when Bigarray.Array1.dim ba >= vec.Vector.length ->
              Vector.ensure_cpu_sync vec ;
              {f32_ba = ba; f32_vec = Some vec; f32_len = vec.Vector.length}
          | _ -> empty_f32)
      | None -> empty_f32)
  | _ -> empty_f32

(** As {!f32_of_native_arg}, for an int32 vector. *)
let i32_of_native_arg (arg : Sarek_ir_types.native_arg) : i32 =
  match arg with
  | Sarek_ir_types.NA_Vec (Sarek_ir_types.NV nv) -> (
      match Type_id.equal nv.underlying_type_id Vector.int32_vector_type_id with
      | Some Type_id.Refl -> (
          let vec : (int32, Bigarray.int32_elt) Vector.t = nv.underlying in
          match vec.Vector.host with
          | Vector.Bigarray_storage ba
            when Bigarray.Array1.dim ba >= vec.Vector.length ->
              Vector.ensure_cpu_sync vec ;
              {i32_ba = ba; i32_vec = Some vec; i32_len = vec.Vector.length}
          | _ -> empty_i32)
      | None -> empty_i32)
  | _ -> empty_i32
