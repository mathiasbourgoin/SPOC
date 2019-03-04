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
(** Manages memory transfers *)
(**/**)
val auto : bool ref
val unsafe : bool ref
(**/**)
 
(** By default Spoc will automatically handle memory transfers
	To make explicit transfers you have to stop auto_transfers 
	
	[auto_transfers true] will make transfers automatic
	
	[auto_transfers false] will stop Spoc to make automatic transfers 
	
	default is : [auto_transfers true] *)
val auto_transfers : bool -> unit
val unsafe_rw : bool -> unit

(** Explicit transfer to a device

	[to_device v d] will transfer the vector v to the device d
	@param queue_id allows to specify which command queue to use for the transfer
	
	transfers are asynchronous, [to_device] will return immediately, 
	
	use [Devices.flush] to wait for the end of the command queue*)
val to_device :
  'a Vector.vector -> ?queue_id:int -> Devices.device -> unit

(** Explicit transfer from a device

	[to_cpu v ] will transfer the vector v from its current location to the cpu
	@param queue_id allows to specify which command queue to use for the transfer
	
	transfers are asynchronous, [to_cpu] will return immediately, 
	
	use [Devices.flush] to wait for the end of the command queue*)
val to_cpu : 'a Vector.vector -> ?queue_id:int -> unit -> unit
(**/**)
val unsafe_set : 'a Vector.vector -> int -> 'a -> unit
val unsafe_get : 'a Vector.vector -> int -> 'a
(**/**)

val set : 'a Vector.vector -> int -> 'a -> unit
val get : 'a Vector.vector -> int -> 'a

(** @return a subvector from a given vector
Subvectors share the same cpu memory space with the vector they are from. 
They do not share the same memory space on gpgpu devices.
[sub_vector vec start len] will return a new vector of length [len] sharing the
same cpu memory space with [a] starting from index [start].*)
val sub_vector :
  'a Vector.vector -> int -> int -> 'a Vector.vector
	
val sub_vector :
           'a Vector.vector ->
           int -> ?ok_rng:int -> ?ko_rng:int -> int -> 'a Vector.vector


(** experimental : @return a copy of a given vector, the copy takes place on the vector location (CPU/GPU) *)
val vector_copy : 
           'a Vector.vector ->
           int -> 'a Vector.vector -> int -> int -> unit

(**/**)
val matrix_copy :
           'a Vector.vector ->
           int ->
           int ->
           int ->
           'a Vector.vector -> int -> int -> int -> int -> int -> unit
(**/**)
