(******************************************************************************
* Mathias Bourgoin, Université Pierre et Marie Curie (2011)
*
* Mathias.Bourgoin@gmail.com
*
* This software is a computer program whose purpose is to allow
* GPU programming with the OCaml language.
*
* This software is governed by the CeCILL - B license under French law and
* abiding by the rules of distribution of free software. You can use,
* modify and / or redistribute the software under the terms of the CeCILL - B
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty and the software's author, the holder of the
* economic rights, and the successive licensors have only limited
* liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading, using, modifying and / or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean that it is complicated to manipulate, and that also
* therefore means that it is reserved for developers and experienced
* professionals having in - depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and / or
* data to be ensured and, more generally, to use and operate it in the
* same conditions as regards security.
*
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL - B license and that you accept its terms.
*******************************************************************************)
(** Manages Spoc vectors *)

type device_vec

type customarray

(** Spoc offers many predefined vectors types.
Custom vectors can contain any kind of data types.*)
type 'a custom = {
  elt : 'a; (** an element of the vector*)
  size : int; (** the size of an element when transfered to a gpgpu device*)
  get : customarray -> int -> 'a; (** a function to access elements from the vector *)
  set : customarray -> int -> 'a -> unit; (** a function to modify an element of the vector *)
}

(** Some predifined types *)

type ('a,'b) couple = 'a*'b

type ('a, 'b) kind =
    Float32 of ('a, 'b) Bigarray.kind
  | Char of ('a, 'b) Bigarray.kind
  | Float64 of ('a, 'b) Bigarray.kind
  | Int32 of ('a, 'b) Bigarray.kind
  | Int64 of ('a, 'b) Bigarray.kind
  | Complex32 of ('a, 'b) Bigarray.kind
  | Custom of 'a custom
  | Unit  of ('a, 'b) couple
  | Dummy  of ('a, 'b) couple

(** shortcuts *)

val int : (int, Bigarray.int_elt) kind
val int32 : (int32, Bigarray.int32_elt) kind
val char : (char, Bigarray.int8_unsigned_elt) kind
val int64 : (int64, Bigarray.int64_elt) kind
val float32 : (float, Bigarray.float32_elt) kind
val float64 : (float, Bigarray.float64_elt) kind
val complex32 : (Complex.t, Bigarray.complex32_elt) kind

(** a spoc_vector is a Bigarray or a custom  vector *)
type ('a, 'b) spoc_vec =
    Bigarray of ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  | CustomArray of (customarray * 'a custom)

(**/**)
external float32_of_float : float -> float = "float32_of_float"
external float_of_float32 : float -> float = "float_of_float32"
(**/**)

type vec_device = No_dev | Dev of Devices.device | Transferring of Devices.device

(** a vector  represents every information needed by Spoc to manage it
 It uses Bigarrays to manage data on the cpu side (see the OCaml Bigarray Module for more informations) *)
type ('a, 'b) vector
  
(** sub vectors are vector parts sharing memory space on cpu memory BUT not on gpu memory, 
allowing easy computation distribution over multiple GPUs.
 sub-vector : sub_vector depth * start * ok range * ko range * parent vector (see samples for more info) *)
and ('a,'b) sub = int * int * int * int * ('a,'b) vector

(**/**)
external init_cuda_device_vec : unit -> device_vec
= "spoc_init_cuda_device_vec"
external init_opencl_device_vec : unit -> device_vec
= "spoc_init_opencl_device_vec"
external create_custom : 'a custom -> int -> customarray
= "spoc_create_custom"
val vec_id : int ref
(**/**)

(** @return a new vector.*)
val create :
           ('a, 'b) kind -> ?dev:Devices.device -> int -> ('a, 'b) vector
(** @return the length of a given vector *)
val length : ('a, 'b) vector -> int

(** @return the device where the given vector is located *)
val dev : ('a, 'b) vector -> vec_device

(** checks if a vector is a subvector *)
val is_sub : ('a,'b) vector -> (('a, 'b) sub) option

(** @return the kind of a vector *)
val kind : ('a,'b) vector -> ('a, 'b) kind

(** @return the device id where the given vector is located *)
val device : ('a, 'b) vector -> int

(**/**)
val get_vec_id : ('a, 'b) vector -> int
(**/**)

(**/**)
val vector : ('a, 'b) vector -> ('a, 'b) spoc_vec
(**/**)

(** checks equality between two vectors *)
val equals : ('a, 'b) vector -> ('a, 'b) vector -> bool


val vseek : ('a, 'b) vector -> int -> unit
val get_seek : ('a, 'b) vector -> int


val unsafe_get : ('a, 'b) vector -> int -> 'a
val unsafe_set : ('a, 'b) vector -> int -> 'a -> unit

(**/**)
val update_device_array : ('a, 'b) vector -> ('a, 'b) vector -> unit

val set_device : ('a, 'b) vector -> int -> vec_device -> unit

val temp_vector : ('a, 'b) vector -> ('a, 'b) vector
(**/**)

val sub_vector :
('a, 'b) vector -> int -> int -> int -> int -> ('a, 'b) vector

val device_vec :
('a, 'b) vector -> [< `Cuda | `OpenCL ] -> int -> device_vec

val copy_sub : ('a, 'b) vector -> ('a, 'b) vector -> unit

val of_bigarray_shr :
           ('a, 'b) kind ->
           ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> ('a, 'b) vector
          
val to_bigarray_shr :
  ('a, 'b) vector  ->   ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
