(******************************************************************************
 * Mathias Bourgoin (2023)
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

open Devices

module type S = sig
  val name : string
  val version : int

  module Devices : sig
    val init : unit -> unit
    val get_compatible_devices : unit -> int
    val get_device : int -> device
    val flush : generalInfo -> device -> int -> unit
    val flush_all : generalInfo -> device -> unit
  end

  module Mem : sig
    val alloc : ('a, 'b) Vector.vector -> int -> generalInfo -> unit
    val alloc_custom : ('a, 'b) Vector.vector -> int -> generalInfo -> unit
    val free : ('a, 'b) Vector.vector -> int -> generalInfo -> unit

    val part_host_to_device :
      ('a, 'b) Vector.vector ->
      ('a, 'b) Vector.vector ->
      int ->
      generalInfo ->
      gcInfo ->
      int ->
      int ->
      int ->
      int ->
      int ->
      unit

    val host_to_device :
      ('a, 'b) Vector.vector -> int -> generalInfo -> gcInfo -> int -> unit

    val device_to_device : ('a, 'b) Vector.vector -> int -> device -> unit

    val device_to_host :
      ('a, 'b) Vector.vector -> int -> generalInfo -> device -> int -> unit

    val custom_part_host_to_device :
      ('a, 'b) Vector.vector ->
      ('a, 'b) Vector.vector ->
      int ->
      generalInfo ->
      gcInfo ->
      int ->
      int ->
      int ->
      int ->
      int ->
      unit

    val custom_device_to_host :
      ('a, 'b) Vector.vector ->
      int ->
      generalInfo ->
      specificInfo ->
      int ->
      unit

    val part_device_to_host :
      ('a, 'b) Vector.vector ->
      ('a, 'b) Vector.vector ->
      int ->
      generalInfo ->
      gcInfo ->
      int ->
      int ->
      int ->
      int ->
      int ->
      unit

    val custom_part_device_to_host :
      ('a, 'b) Vector.vector ->
      ('a, 'b) Vector.vector ->
      int ->
      generalInfo ->
      gcInfo ->
      int ->
      int ->
      int ->
      int ->
      int ->
      unit

    val vector_copy :
      ('a, 'b) Vector.vector ->
      int ->
      ('a, 'b) Vector.vector ->
      int ->
      int ->
      generalInfo ->
      int ->
      unit

    val custom_vector_copy :
      ('a, 'b) Vector.vector ->
      int ->
      ('a, 'b) Vector.vector ->
      int ->
      int ->
      generalInfo ->
      int ->
      unit

    val matrix_copy :
      ('a, 'b) Vector.vector ->
      int ->
      int ->
      int ->
      ('a, 'b) Vector.vector ->
      int ->
      int ->
      int ->
      int ->
      int ->
      generalInfo ->
      int ->
      unit

    val custom_matrix_copy :
      ('a, 'b) Vector.vector ->
      int ->
      int ->
      int ->
      ('a, 'b) Vector.vector ->
      int ->
      int ->
      int ->
      int ->
      int ->
      generalInfo ->
      int ->
      unit
  end

  module Kernel : sig
    open Kernel

    type extra

    val compile : string -> string -> generalInfo -> kernel
    val debug_compile : string -> string -> generalInfo -> kernel

    val load_arg :
      int ref -> extra -> device -> kernel -> int -> ('a, 'b) kernelArgs -> unit

    val launch_grid :
      int ref -> kernel -> grid -> block -> extra -> generalInfo -> int -> unit
  end

  module Vector : sig
    open Vector

    val init_device : unit -> device_vec
    val alloc : ('a, 'b) vector -> int -> generalInfo -> unit
    val custom_alloc : ('a, 'b) vector -> int -> generalInfo -> unit
  end
end
