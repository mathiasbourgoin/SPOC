(******************************************************************************
 * Vector transfer helpers (split from Vector.ml)
 *
 * This module is intended to hold transfer-related helpers to reduce the size
 * of Vector.ml. Currently it only re-exports selected helpers used by other
 * modules; future work can move more logic here.
 ******************************************************************************)

(** Convert bigarray to raw pointer and byte size *)
let bigarray_to_ptr (type a b)
    (ba : (a, b, Bigarray.c_layout) Bigarray.Array1.t) (elem_size : int) :
    unit Ctypes.ptr * int =
  let len = Bigarray.Array1.dim ba in
  let byte_size = len * elem_size in
  let ptr = Ctypes.bigarray_start Ctypes.array1 ba in
  (Ctypes.to_voidp ptr, byte_size)
