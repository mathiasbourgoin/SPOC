val kernel_max :
  ('a, 'b) Spoc.Vector.vector * ('a, 'b) Spoc.Vector.vector * int ->
  Spoc.Kernel.block * Spoc.Kernel.grid ->
  int -> Spoc.Devices.device -> Spoc.Kernel.kernel -> unit
class ['a, 'b] class_kernel_max :
  object
    constraint 'a = float
    constraint 'b = Bigarray.float64_elt
    val binaries : (Spoc.Devices.device, Spoc.Kernel.kernel) Hashtbl.t
    val mutable cuda_sources : string list
    val file_file : string
    val kernel_name : string
    val mutable opencl_sources : string list
    method args_to_list :
      ('a, 'b) Spoc.Vector.vector * ('a, 'b) Spoc.Vector.vector * int ->
      ('a, 'b) Spoc.Kernel.kernelArgs array
    method compile : ?debug:bool -> Spoc.Devices.device -> unit
    method compile_and_run :
      ('a, 'b) Spoc.Vector.vector * ('a, 'b) Spoc.Vector.vector * int ->
      Spoc.Kernel.block * Spoc.Kernel.grid ->
      ?debug:bool -> int -> Spoc.Devices.device -> unit
    method exec :
      ('a, 'b) Spoc.Vector.vector * ('a, 'b) Spoc.Vector.vector * int ->
      Spoc.Kernel.block * Spoc.Kernel.grid ->
      int -> Spoc.Devices.device -> Spoc.Kernel.kernel -> unit
    method get_binaries :
      unit -> (Spoc.Devices.device, Spoc.Kernel.kernel) Hashtbl.t
    method get_cuda_sources : unit -> string list
    method get_opencl_sources : unit -> string list
    method list_to_args :
      ('a, 'b) Spoc.Kernel.kernelArgs array ->
      ('a, 'b) Spoc.Vector.vector * ('a, 'b) Spoc.Vector.vector * int
    method reload_sources : unit -> unit
    method reset_binaries : unit -> unit
    method run :
      ('a, 'b) Spoc.Vector.vector * ('a, 'b) Spoc.Vector.vector * int ->
      Spoc.Kernel.block * Spoc.Kernel.grid ->
      int -> Spoc.Devices.device -> unit
    method set_cuda_sources : string -> unit
    method set_opencl_sources : string -> unit
  end
val spoc_max : (float, Bigarray.float64_elt) class_kernel_max
class ['a, 'b] class_dummy_kernel :
  object
    val binaries : (Spoc.Devices.device, Spoc.Kernel.kernel) Hashtbl.t
    val mutable cuda_sources : string list
    val file_file : string
    val kernel_name : string
    val mutable opencl_sources : string list
    method args_to_list : 'a -> 'b
    method compile : ?debug:bool -> Spoc.Devices.device -> unit
    method compile_and_run :
      'a ->
      Spoc.Kernel.block * Spoc.Kernel.grid ->
      ?debug:bool -> int -> Spoc.Devices.device -> unit
    method exec :
      'a ->
      Spoc.Kernel.block * Spoc.Kernel.grid ->
      int -> Spoc.Devices.device -> Spoc.Kernel.kernel -> unit
    method get_binaries :
      unit -> (Spoc.Devices.device, Spoc.Kernel.kernel) Hashtbl.t
    method get_cuda_sources : unit -> string list
    method get_opencl_sources : unit -> string list
    method list_to_args : 'b -> 'a
    method reload_sources : unit -> unit
    method reset_binaries : unit -> unit
    method run :
      'a ->
      Spoc.Kernel.block * Spoc.Kernel.grid ->
      int -> Spoc.Devices.device -> unit
    method set_cuda_sources : string -> unit
    method set_opencl_sources : string -> unit
  end

type reduction = Max
type application = Map | Pipe | Reduce | Par | Filter
class type ['a, 'b, 'c] skeleton =
  object
    constraint 'c = ('d, 'e) Spoc.Vector.vector
    val env : 'a
    val ker : ('a, 'b) Spoc.Kernel.spoc_kernel
    val kind : application
    method env : unit -> 'a
    method ker : unit -> ('a, 'b) Spoc.Kernel.spoc_kernel
    method kind : unit -> application
    method par_run :
      Spoc.Devices.device list -> ('f, 'g) Spoc.Vector.vector -> 'c
    method run :
      ?queue_id:int ->
      Spoc.Devices.device -> ('f, 'g) Spoc.Vector.vector -> 'c
  end
class ['a, 'b, 'c] map :
  ('a, 'b) #Spoc.Kernel.spoc_kernel ->
  'c ->
  'a ->
  object
    constraint 'b = ('d, 'e) Spoc.Kernel.kernelArgs array
    constraint 'c = ('d, 'e) Spoc.Vector.vector
    val env : 'a
    val ker : ('a, 'b) Spoc.Kernel.spoc_kernel
    val kind : application
    method env : unit -> 'a
    method ker : unit -> ('a, 'b) Spoc.Kernel.spoc_kernel
    method kind : unit -> application
    method par_run :
      Spoc.Devices.device list -> ('f, 'g) Spoc.Vector.vector -> 'c
    method run :
      ?queue_id:int ->
      Spoc.Devices.device -> ('f, 'g) Spoc.Vector.vector -> 'c
  end
val get_vector :
  ('a, 'b) Spoc.Kernel.kernelArgs -> ('a, 'b) Spoc.Vector.vector
val transfer_if_vector :
  ('a, 'b) Spoc.Kernel.kernelArgs -> Spoc.Devices.device -> int -> unit
class ['a, 'b, 'c] pipe :
  ('a, 'b, ('d, 'e) Spoc.Vector.vector) #skeleton ->
  ('i, ('h, 'j) Spoc.Kernel.kernelArgs array, 'c) #skeleton ->
  object
    constraint 'b = ('m, 'n) Spoc.Kernel.kernelArgs array
    constraint 'c = ('k, 'l) Spoc.Vector.vector
    val env : 'a
    val ker : ('a, 'b) Spoc.Kernel.spoc_kernel
    val kind : application
    method env : unit -> 'a
    method ker : unit -> ('a, 'b) Spoc.Kernel.spoc_kernel
    method kind : unit -> application
    method par_run :
      Spoc.Devices.device list -> ('f, 'g) Spoc.Vector.vector -> 'c
    method run :
      ?queue_id:int ->
      Spoc.Devices.device -> ('f, 'g) Spoc.Vector.vector -> 'c
  end
class ['a, 'b, 'c] reduce :
  ('a, 'b) #Spoc.Kernel.spoc_kernel ->
  'c ->
  'a ->
  object
    constraint 'b = ('h, 'i) Spoc.Kernel.kernelArgs array
    constraint 'c = ('d, 'e) Spoc.Vector.vector
    val env : 'a
    val ker : ('a, 'b) Spoc.Kernel.spoc_kernel
    val kind : application
    method env : unit -> 'a
    method ker : unit -> ('a, 'b) Spoc.Kernel.spoc_kernel
    method kind : unit -> application
    method par_run :
      Spoc.Devices.device list -> ('f, 'g) Spoc.Vector.vector -> 'c
    method run :
      ?queue_id:int ->
      Spoc.Devices.device -> ('f, 'g) Spoc.Vector.vector -> 'c
  end
val run :
  ('a, 'b, ('c, 'd) Spoc.Vector.vector) #skeleton ->
  ?queue_id:int ->
  Spoc.Devices.device ->
  ('e, 'f) Spoc.Vector.vector -> ('c, 'd) Spoc.Vector.vector
val par_run :
  ('a, 'b, ('c, 'd) Spoc.Vector.vector) #skeleton ->
  Spoc.Devices.device list ->
  ('e, 'f) Spoc.Vector.vector -> ('c, 'd) Spoc.Vector.vector
val pipe :
  ('a, ('b, 'c) Spoc.Kernel.kernelArgs array, ('e, 'f) Spoc.Vector.vector)
  skeleton ->
  ('d, ('g, 'h) Spoc.Kernel.kernelArgs array, ('i, 'j) Spoc.Vector.vector)
  skeleton ->
  ('a, ('b, 'c) Spoc.Kernel.kernelArgs array, ('i, 'j) Spoc.Vector.vector)
  pipe
val map :
  ('a, ('b, 'c) Spoc.Kernel.kernelArgs array) #Spoc.Kernel.spoc_kernel ->
  ('b, 'c) Spoc.Vector.vector ->
  'a ->
  ('a, ('b, 'c) Spoc.Kernel.kernelArgs array, ('b, 'c) Spoc.Vector.vector)
  map
val reduce :
  ('a, ('b, 'c) Spoc.Kernel.kernelArgs array) #Spoc.Kernel.spoc_kernel ->
  ('d, 'e) Spoc.Vector.vector ->
  'a ->
  ('a, ('b, 'c) Spoc.Kernel.kernelArgs array, ('d, 'e) Spoc.Vector.vector)
  reduce
