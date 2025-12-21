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
exception No_OpenCL_Device

exception OPENCL_ERROR_UNKNOWN

exception INVALID_CONTEXT

exception INVALID_DEVICE

exception INVALID_VALUE

exception INVALID_QUEUE_PROPERTIES

exception OUT_OF_RESOURCES

exception MEM_OBJECT_ALLOCATION_FAILURE

exception OUT_OF_HOST_MEMORY

exception FILE_NOT_FOUND

exception INVALID_PROGRAM

exception INVALID_BINARY

exception INVALID_BUILD_OPTIONS

exception INVALID_OPERATION

exception COMPILER_NOT_AVAILABLE

exception BUILD_PROGRAM_FAILURE

exception INVALID_KERNEL

exception INVALID_ARG_INDEX

exception INVALID_ARG_VALUE

exception INVALID_MEM_OBJECT

exception INVALID_SAMPLER

exception INVALID_ARG_SIZE

exception INVALID_COMMAND_QUEUE

(**/**)

external opencl_alloc_vect :
  ('a, 'b) Vector.vector -> int -> Devices.generalInfo -> unit
  = "spoc_opencl_alloc_vect"

external opencl_custom_alloc_vect :
  ('a, 'b) Vector.vector -> int -> Devices.generalInfo -> unit
  = "spoc_opencl_custom_alloc_vect"

external opencl_free_vect : ('a, 'b) Vector.vector -> int -> unit
  = "spoc_opencl_free_vect"

external opencl_part_device_to_cpu :
  ('a, 'b) Vector.vector ->
  ('a, 'b) Vector.vector ->
  int ->
  Devices.generalInfo ->
  Devices.gcInfo ->
  int ->
  int ->
  int ->
  int ->
  int ->
  unit = "spoc_opencl_part_device_to_cpu_b" "spoc_opencl_part_device_to_cpu_n"

external opencl_custom_part_device_to_cpu :
  ('a, 'b) Vector.vector ->
  ('a, 'b) Vector.vector ->
  int ->
  Devices.generalInfo ->
  Devices.gcInfo ->
  int ->
  int ->
  int ->
  int ->
  int ->
  unit
  = "spoc_opencl_custom_part_device_to_cpu_b"
    "spoc_opencl_custom_part_device_to_cpu_n"

external opencl_part_cpu_to_device :
  ('a, 'b) Vector.vector ->
  ('a, 'b) Vector.vector ->
  int ->
  Devices.generalInfo ->
  Devices.gcInfo ->
  int ->
  int ->
  int ->
  int ->
  int ->
  unit = "spoc_opencl_part_cpu_to_device_b" "spoc_opencl_part_cpu_to_device_n"

external opencl_custom_part_cpu_to_device :
  ('a, 'b) Vector.vector ->
  ('a, 'b) Vector.vector ->
  int ->
  Devices.generalInfo ->
  Devices.gcInfo ->
  int ->
  int ->
  int ->
  int ->
  int ->
  unit
  = "spoc_opencl_custom_part_cpu_to_device_b"
    "spoc_opencl_custom_part_cpu_to_device_n"

external opencl_cpu_to_device :
  ('a, 'b) Vector.vector -> int -> Devices.generalInfo -> int -> unit
  = "spoc_opencl_cpu_to_device"

external opencl_device_to_device :
  ('a, 'b) Vector.vector -> int -> Devices.device -> unit
  = "spoc_opencl_device_to_device"

external opencl_device_to_cpu :
  ('a, 'b) Vector.vector ->
  int ->
  Devices.generalInfo ->
  Devices.specificInfo ->
  int ->
  unit = "spoc_opencl_device_to_cpu"

external opencl_custom_cpu_to_device :
  ('a, 'b) Vector.vector -> int -> Devices.generalInfo -> int -> unit
  = "spoc_opencl_custom_cpu_to_device"

external opencl_custom_device_to_cpu :
  ('a, 'b) Vector.vector ->
  int ->
  Devices.generalInfo ->
  Devices.specificInfo ->
  int ->
  unit = "spoc_opencl_custom_device_to_cpu"

external opencl_vector_copy :
  ('a, 'b) Vector.vector ->
  int ->
  ('a, 'b) Vector.vector ->
  int ->
  int ->
  Devices.generalInfo ->
  int ->
  unit = "spoc_opencl_vector_copy_b" "spoc_opencl_vector_copy_n"

external opencl_custom_vector_copy :
  ('a, 'b) Vector.vector ->
  int ->
  ('a, 'b) Vector.vector ->
  int ->
  int ->
  Devices.generalInfo ->
  int ->
  unit = "spoc_opencl_custom_vector_copy_b" "spoc_opencl_custom_vector_copy_n"

external opencl_matrix_copy :
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
  Devices.generalInfo ->
  int ->
  unit = "spoc_opencl_matrix_copy_b" "spoc_opencl_matrix_copy_n"

external opencl_custom_matrix_copy :
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
  Devices.generalInfo ->
  int ->
  unit = "spoc_opencl_custom_matrix_copy_b" "spoc_opencl_custom_matrix_copy_n"

(**/**)
