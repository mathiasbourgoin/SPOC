(******************************************************************************
* Mathias Bourgoin, Université Pierre et Marie Curie (2011)
 *
 * Mathias.Bourgoin@gmail.com
 *
 * This software is a computer program whose purpose is to allow
 * GPU programming with the OCaml language.
 *
 * This software is governed by the CeCILL-B license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-B
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-B license and that you accept its terms.
*******************************************************************************)

exception NOT_INITIALIZED

exception ALLOC_FAILED

exception INVALID_VALUE

exception ARCH_MISMATCH

exception MAPPING_ERROR

exception EXECUTION_FAILED

exception INTERNAL_ERROR

exception UNKNOWN_ERROR

external init : unit -> unit = "spoc_cublasInit"

external shutdown : unit -> unit = "spoc_cublasShutdown"

external getError : unit -> unit = "spoc_cublasGetError"

type vfloat32 = (float, Bigarray.float32_elt) Spoc.Vector.vector

type vchar = (char, Bigarray.int8_unsigned_elt) Spoc.Vector.vector

type vfloat64 = (float, Bigarray.float64_elt) Spoc.Vector.vector

type vcomplex32 = (Complex.t, Bigarray.complex32_elt) Spoc.Vector.vector

val check_auto :
  ('a, 'b) Spoc.Vector.vector array -> Spoc.Devices.device -> unit

val cublasIsamax : int -> vfloat32 -> int -> Spoc.Devices.device -> int

val cublasIsamin : int -> vfloat32 -> int -> Spoc.Devices.device -> int

val cublasSasum : int -> vfloat32 -> int -> Spoc.Devices.device -> float

val cublasSaxpy :
  int ->
  float ->
  vfloat32 ->
  int ->
  vfloat32 ->
  int ->
  Spoc.Devices.device ->
  unit

val cublasScopy :
  int -> vfloat32 -> int -> vfloat32 -> int -> Spoc.Devices.device -> unit

val cublasSdot :
  int -> vfloat32 -> int -> vfloat32 -> int -> Spoc.Devices.device -> float

val cublasSnrm2 : int -> vfloat32 -> int -> Spoc.Devices.device -> float

val cublasSrot :
  int ->
  vfloat32 ->
  int ->
  vfloat32 ->
  int ->
  float ->
  float ->
  Spoc.Devices.device ->
  unit

val cublasSrotg : vfloat32 -> vfloat32 -> vfloat32 -> vfloat32 -> 'a -> unit

val cublasSrotm :
  int ->
  vfloat32 ->
  int ->
  vfloat32 ->
  int ->
  vfloat32 ->
  Spoc.Devices.device ->
  unit

val cublasSrotmg :
  vfloat32 -> vfloat32 -> vfloat32 -> vfloat32 -> vfloat32 -> 'a -> unit

val cublasSscal : int -> float -> vfloat32 -> int -> Spoc.Devices.device -> unit

val cublasSswap :
  int -> vfloat32 -> int -> vfloat32 -> int -> Spoc.Devices.device -> unit

val cublasCaxpy :
  int ->
  Complex.t ->
  vcomplex32 ->
  int ->
  vcomplex32 ->
  int ->
  Spoc.Devices.device ->
  unit

val cublasScasum : int -> vcomplex32 -> int -> Spoc.Devices.device -> float

val cublasSgemm :
  char ->
  char ->
  int ->
  int ->
  int ->
  float ->
  vfloat32 ->
  int ->
  vfloat32 ->
  int ->
  float ->
  vfloat32 ->
  int ->
  Spoc.Devices.device ->
  unit

(** cublasDgemm transa transb m n k alpha a lda b ldb beta c ldc dev *)
val cublasDgemm :
  char ->
  char ->
  int ->
  int ->
  int ->
  float ->
  vfloat64 ->
  int ->
  vfloat64 ->
  int ->
  float ->
  vfloat64 ->
  int ->
  Spoc.Devices.device ->
  unit

val run : 'a -> ('a -> 'b) -> 'b

val setMatrix :
  int ->
  int ->
  ('a, 'b) Spoc.Vector.vector ->
  int ->
  ('a, 'b) Spoc.Vector.vector ->
  int ->
  Spoc.Devices.device ->
  unit
