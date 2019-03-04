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
  size : int; (** the size of an element when transferred to a gpgpu device*)
  get : customarray -> int -> 'a; (** a function to access elements from the vector *)
  set : customarray -> int -> 'a -> unit; (** a function to modify an element of the vector *)
}

(** Some predifined types *)
type float32 = float
type float64 = float
type int32 = Int32.t
type int64 = Int64.t
type complex32 = Complex.t

type _ kind =
    Float32 : float32 kind
  | Char : char kind
  | Float64 : float64 kind
  | Int32 : int32 kind
  | Int64 : int64 kind
  | Complex32 : complex32 kind
  | Custom : 'a custom -> 'a  kind

(** shortcuts *)

val int32 : int32 kind
val char : char kind
val int64 : int64 kind
val float32 : float32 kind
val float64 : float64 kind
val complex32 : complex32 kind

val custom  : 'a custom -> 'a kind
type 'a host_vec

(** a spoc_vector is a Bigarray or a custom  vector *)
type _ spoc_vec =
  | CustomArray : (customarray * 'a custom) -> 'a spoc_vec
  | Host_vec : 'a host_vec -> 'a spoc_vec



(**/**)
external float32_of_float : float -> float = "float32_of_float"
external float_of_float32 : float -> float = "float_of_float32"
(**/**)

type vec_device = No_dev | Dev of Devices.device | Transferring of Devices.device

(** a vector  represents every information needed by Spoc to manage it *)
type 'a vector

(** sub vectors are vector parts sharing memory space on cpu memory BUT not on gpu memory,
allowing easy computation distribution over multiple GPUs.
 sub-vector : sub_vector depth * start * ok range * ko range * parent vector (see samples for more info) *)
and 'a sub = int * int * int * int * 'a vector

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
           'a kind -> ?dev:Devices.device -> int -> 'a vector
(** @return the length of a given vector *)
val length : 'a vector -> int

(** @return the device where the given vector is located *)
val dev : 'a vector -> vec_device

(** checks if a vector is a subvector *)
val is_sub : 'a vector -> 'a sub option

(** @return the kind of a vector *)
val kind : 'a vector -> 'a kind

(** @return the device id where the given vector is located *)
val device : 'a vector -> int

(**/**)
val get_vec_id : 'a vector -> int
(**/**)

(**/**)
val vector : 'a vector -> 'a spoc_vec
(**/**)

(** checks equality between two vectors *)
val equals : 'a vector -> 'a vector -> bool


val vseek : 'a vector -> int -> unit
val get_seek : 'a vector -> int


val unsafe_get : 'a vector -> int -> 'a
val unsafe_set : 'a vector -> int -> 'a -> unit

(**/**)
val update_device_array : 'a vector -> 'a vector -> unit

val set_device : 'a vector -> int -> vec_device -> unit

val temp_vector : 'a vector -> 'a vector
(**/**)

val sub_vector :
'a vector -> int -> int -> int -> int -> 'a vector

val device_vec :
'a vector -> [< `Cuda | `OpenCL ] -> int -> device_vec

val copy_sub : 'a vector -> 'a vector -> unit

val of_bigarray_shr :
           'a kind ->
           ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> 'a vector

val to_bigarray_shr :
  'a vector  ->   ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
