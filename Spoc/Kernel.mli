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
(** Manages Kernels *)
type kernel

external relax_vector : ('a,'b) Vector.vector -> ('c,'d) Vector.vector = "%identity"

(** type of parameters usable with kernels *)
type ('a, 'b) kernelArgs =
  | VChar of ('a, 'b) Vector.vector (** unsigned char vector *)
  | VFloat32 of ('a, 'b) Vector.vector (** 32-bit float vector *)
  | VFloat64 of ('a, 'b) Vector.vector (** 64-bit float vector *)
  | VComplex32 of ('a, 'b) Vector.vector (** 32-bit complex vector *)
  | VInt32 of ('a, 'b) Vector.vector (** 32-bit int vector *)
  | VInt64 of ('a, 'b) Vector.vector (** 64-bit int vector *)
  | Int32 of int (** 32-bit int *)
  | Int64 of int (** 64-bit int *)
  | Float32 of float (** 32-bit float *)
  | Float64 of float (** 64-bit float *)
  | Custom of ('a,'b) Vector.custom
  | Vector of ('a, 'b) Vector.vector  (** generic vector type *)
  | VCustom of ('a, 'b) Vector.vector (** custom data type vector, see examples *)

(** A block is a 3 dimension group of threads *)
type block = {
  mutable blockX : int;
  mutable blockY : int;
  mutable blockZ : int;
}

(** A grid is a 3 dimension group of blocks *)
type grid = {
  mutable gridX : int;
  mutable gridY : int;
  mutable gridZ : int;
}

exception ERROR_BLOCK_SIZE
exception ERROR_GRID_SIZE

module Cuda :
	sig
  type cuda_extra 		
  external cuda_create_extra : int -> cuda_extra = "spoc_cuda_create_extra"
  external cuda_launch_grid :
      int ref ->
      kernel ->
      grid -> block -> cuda_extra -> Devices.generalInfo -> int -> unit
      = "spoc_cuda_launch_grid_b" "spoc_cuda_launch_grid_n"
	
	val cuda_load_arg :
      int ref ->
      cuda_extra -> Devices.device -> 'c -> 'd -> ('a, 'b) kernelArgs -> unit
end


module OpenCL :
    sig			 
		external opencl_launch_grid :
      kernel -> grid -> block -> Devices.generalInfo -> int -> unit
      = "spoc_opencl_launch_grid"
    val opencl_load_arg :
      int ref ->
      Devices.device -> kernel -> int -> ('a, 'b) kernelArgs -> unit

end


(**/**)
val exec :
('a,'b) kernelArgs array ->
block * grid -> int -> Devices.device -> kernel -> unit

val compile_and_run :
Devices.device ->
block * grid ->
?cached: bool ->
?debug: bool ->
?queue_id: int ->
('a * (block * grid -> bool -> bool -> int -> Devices.device -> 'b)) * 'c ->
'b
(**/**)

exception No_source_for_device of Devices.device
exception Not_compiled_for_device of Devices.device

(** a Kernel is represented within Spoc a an OCaml object inheriting the spoc_kernel class *)
class virtual ['a, 'b] spoc_kernel :
string ->
string ->
object
  
  (** hashtable containing the compiled binaries of the kernel (one for each device it has been compiled  for *)
  val binaries : (Devices.device, kernel) Hashtbl.t
  (** the cuda source code of the kernel *)
  val mutable cuda_sources : string list
  (**/**)
  val file_file : string
  val kernel_name : string
  (**/**)
  (** the opencl source code of the kernel *)
  val mutable opencl_sources : string list
  (** clean binary cache *)
	method get_binaries : unit -> (Devices.device, kernel) Hashtbl.t
  method reset_binaries : unit -> unit
  (** compiles a kernel for a device *)
  method compile : ?debug: bool -> Devices.device -> unit
  (** compiles and run a device for a kernel *)
  method compile_and_run :
  'a -> block * grid -> ?debug: bool -> int -> Devices.device -> unit
  method virtual exec :
  'a ->
  block * grid -> int -> Devices.device -> kernel -> unit
  (**/**)
  method get_cuda_sources : unit -> string list
  method get_opencl_sources : unit -> string list
  (**/**)
  (** reloads the sources form the file associated with the kernel *)
  method reload_sources : unit -> unit
  (** runs the kernel on a device *)
  method run :
  'a -> block * grid -> int -> Devices.device -> unit
  (**/**)
  method set_cuda_sources : string -> unit
  method set_opencl_sources : string -> unit
  
  method virtual list_to_args : 'b -> 'a
  method virtual args_to_list : 'a -> 'b
	
  (**/**)
end

(** @deprecated you should use kernel#run *)
val run :
Devices.device ->
block * grid ->
('a, 'b) spoc_kernel -> 'a -> unit

(** @deprecated you should use kernel#compile *)
val compile : Devices.device -> ('a, 'b) spoc_kernel -> unit

val set_arg : ('a, 'b) kernelArgs array -> int -> ('a,'b) Vector.vector -> unit
