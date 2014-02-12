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

 type device_vec

 type customarray

 type 'a custom =
   {
     size : int ;
     get: customarray -> int -> 'a;
     set: customarray -> int -> 'a -> unit
   }

 type ('a,'b) ctypes_custom =
   {
     c_elt : 'b Ctypes.typ;
     c_size : int ;
     c_get: 'b Ctypes.ptr -> int -> 'a;
     c_set: 'b Ctypes.ptr -> int -> 'a -> unit
  }


type ('a,'b) couple = 'a*'b

type ('a, 'b) kind =
    Float32 of ('a, 'b) Bigarray.kind
  | Char of ('a, 'b) Bigarray.kind
  | Float64 of ('a, 'b) Bigarray.kind
  | Int32 of ('a, 'b) Bigarray.kind
  | Int64 of ('a, 'b) Bigarray.kind
  | Complex32 of ('a, 'b) Bigarray.kind
  | Custom of 'a custom
  | Ccustom of ('a,'b) ctypes_custom
  | Unit of ('a, 'b) couple
  | Dummy of ('a, 'b) couple

(* shortcuts *)
let int = Int32 (Bigarray.int)
let char = Char (Bigarray.char)
let int32 = Int32 (Bigarray.int32)
let int64 = Int64 (Bigarray.int64)
let float32 = Float32 (Bigarray.float32)
let float64 = Float64 (Bigarray.float64)
let complex32 = Complex32 (Bigarray.complex32)

type ('a,'b) spoc_vec =
  | Bigarray of ('a, 'b, Bigarray.c_layout)Bigarray.Array1.t
  | CustomArray of (customarray * 'a custom)
  | CcustomArray of ('b Ctypes.ptr)

external float32_of_float : float -> float = "float32_of_float"
external float_of_float32 : float -> float = "float_of_float32"

type vec_device =
  | No_dev
  | Dev of Devices.device
  | Transferring of Devices.device

type ('a,'b) sub = int * int * int * int * ('a,'b) vector
  (* sub_vector depth * start * ok range * ko range * parent vector *)

and ('a,'b) vector = {
  mutable device : int; (* -1 -> CPU, 0+ -> device *)
  vector : ('a,'b) spoc_vec;
  cuda_device_vec : device_vec array;
  opencl_device_vec : device_vec array;
  length : int;
  mutable dev: vec_device;
  kind: ('a, 'b)kind;
  mutable is_sub: (('a,'b) sub) option;(*('a, 'b) sub;*)
  sub : ('a, 'b) vector list;
  vec_id: int;
	mutable seek:int
}

external init_cuda_device_vec: unit -> device_vec = "spoc_init_cuda_device_vec"
external init_opencl_device_vec : unit -> device_vec = "spoc_init_opencl_device_vec"

external create_custom : 'a custom -> int -> customarray = "spoc_create_custom"


external cuda_custom_alloc_vect :
  ('a, 'b) vector -> int -> Devices.generalInfo -> unit =
  "spoc_cuda_custom_alloc_vect"
  
external cuda_alloc_vect :
  ('a, 'b) vector -> int -> Devices.generalInfo -> unit =
  "spoc_cuda_alloc_vect"
	
external opencl_alloc_vect :
  ('a, 'b) vector -> int -> Devices.generalInfo -> unit =
  "spoc_opencl_alloc_vect"
external opencl_custom_alloc_vect :
  ('a, 'b) vector -> int -> Devices.generalInfo -> unit =
  "spoc_opencl_alloc_vect"
	
let vec_id = ref 0

let create (kind: ('a,'b) kind) ?dev size =
  incr vec_id;
  let vec = 
    match kind with
		| Unit x | Dummy x-> assert false
    |  Float32 x | Char x | Float64 x
    | Int32 x | Int64 x | Complex32 x -> {
      device = -1;
      vector =
        (let res = (Bigarray.Array1.create x Bigarray.c_layout size)
         in
         Bigarray res);
      cuda_device_vec = Array.create (Devices.cuda_devices() +1) (init_cuda_device_vec ());
      opencl_device_vec = Array.create (Devices.opencl_devices() +1) (init_opencl_device_vec ());
      length = size;
      dev = No_dev;
      kind = kind;
      is_sub = None;
      sub = [];
      vec_id = !vec_id;
      seek = 0; }
      
  | Custom c -> {
        device = -1;
        vector = CustomArray ((create_custom c size), c);
        cuda_device_vec = Array.create (Devices.cuda_devices() +1) (init_cuda_device_vec ());
        opencl_device_vec = Array.create (Devices.opencl_devices() +1) (init_opencl_device_vec ());
        length = size;
        dev = No_dev;
        kind = kind;
        is_sub = None;
        sub = [];
        vec_id = !vec_id; seek = 0; }
  | Ccustom c -> {
        device = -1;
        vector = 
          CcustomArray 
            (Ctypes.allocate_n (c.c_elt) ~count:size);
        cuda_device_vec = Array.create (Devices.cuda_devices() +1) (init_cuda_device_vec ());
        opencl_device_vec = Array.create (Devices.opencl_devices() +1) (init_opencl_device_vec ());
        length = size;
        dev = No_dev;
        kind = kind;
        is_sub = None;
        sub = [];
        vec_id = !vec_id; seek = 0; }

	in
	(match dev with 
	| None  ->  ()
	| Some dev  -> 
		let alloc_on_dev () = 
		  (match dev.Devices.specific_info with 
		  | Devices.CudaInfo  ci -> 
		    (match kind with 
		    | Custom c  -> 
		      cuda_custom_alloc_vect vec dev.Devices.general_info.Devices.id dev.Devices.general_info
		    | _  -> cuda_alloc_vect vec dev.Devices.general_info.Devices.id dev.Devices.general_info)
		  | Devices.OpenCLInfo cli -> 	
		    (match kind with 
		    | Custom c  -> 
		      opencl_custom_alloc_vect vec  (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ())) dev.Devices.general_info
		    | _  ->  opencl_alloc_vect vec  (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ())) dev.Devices.general_info)
		  )
		in
		(try
		   alloc_on_dev ()
		 with
		 | _  ->  Gc.full_major (); alloc_on_dev ());
		vec.dev <- Dev dev;
 );
	vec   

let length v =
  v.length

let dev v =
  v.dev

let is_sub v =
  v.is_sub

let kind v = 
  v.kind

let device v =
  v.device

let get_vec_id v =
  v.vec_id

let vector v =
  v.vector

let set_device v device dev =
  v.device <- device;
  v.dev <- dev

let equals v1 v2 =
  v1.vec_id = v2.vec_id
	
	
let vseek v s =
	v.seek <- s

let get_seek v =
	v.seek

let device_vec v framework id =
  match framework with
  | `Cuda -> v.cuda_device_vec.(id)
  | `OpenCL -> v.opencl_device_vec.(id)

let update_device_array vect new_vect =
  (for i = 0 to (Array.length new_vect.cuda_device_vec) - 2 do
      new_vect.cuda_device_vec.(i) <- vect.cuda_device_vec.(i)
    done;
    for i = 0 to (Array.length new_vect.opencl_device_vec) - 2 do
      new_vect.opencl_device_vec.(i) <-
      vect.opencl_device_vec.(i)
    done)

let unsafe_set vect idx value =
  match vect.vector with
  | Bigarray v -> Bigarray.Array1.set v idx value
  | CustomArray c -> (snd c).set (fst c) idx value

let unsafe_get vect idx =
  match vect.vector with
  | Bigarray v -> Bigarray.Array1.get v idx
  | CustomArray c -> (snd c).get (fst c) idx

let temp_vector vect =
  match vect.is_sub with
  | Some (a, _, _, _, v) ->
  (* sub sub vector, contiguity cannot be assured *)
      let new_v = create (vect.kind) (vect.length)
      in
      (new_v.device <- vect.device;
        new_v.dev <- vect.dev;
        update_device_array vect new_v;
        new_v)
  | None -> vect

let copy_sub vect1 vect2 =
  vect2.is_sub <- vect1.is_sub

external sub_custom_array : customarray -> 'a custom -> int -> customarray =
  "spoc_sub_custom_array"

let sub_vector (vect : ('a, 'b) vector) _start _len =
  incr vec_id;
  (match vect.vector with
    | Bigarray b ->
        {
          device = (-1);
          vector = Bigarray (Bigarray.Array1.sub b _start _len);
          cuda_device_vec =
            Array.create ((Devices.cuda_devices ()) + 1) (init_cuda_device_vec ());
          opencl_device_vec =
            Array.create ((Devices.opencl_devices ()) + 1)
              (init_opencl_device_vec ());
          length = _len;
          dev = No_dev;
          kind = vect.kind;
          is_sub = None;
          sub = [];
          vec_id = !vec_id;
					seek = _start;
        }
    | CustomArray (cA, c) ->
        {
          device = (-1);
          vector = CustomArray ((sub_custom_array cA c _start), c);
          cuda_device_vec =
            Array.create ((Devices.cuda_devices ()) + 1) (init_cuda_device_vec ());
          opencl_device_vec =
            Array.create ((Devices.opencl_devices ()) + 1)
              (init_opencl_device_vec ());
          length = _len;
          dev = No_dev;
          kind = vect.kind;
          is_sub = None;
          sub = [];
          vec_id = !vec_id;
					seek = 0;
        })

let dep = function | None -> 0 | Some (a, _, _, _, _) -> a

let sub_vector (vect : ('a, 'b) vector) _start _ok_r
    _ko_r _len =
  incr vec_id;
  (match vect.vector with
    | Bigarray b ->
        {
          device = (-1);
          vector = vect.vector;
          cuda_device_vec =
            Array.create ((Devices.cuda_devices ()) + 1) (init_cuda_device_vec ());
          opencl_device_vec =
            Array.create ((Devices.opencl_devices ()) + 1)
              (init_opencl_device_vec ());
          length = _len;
          dev = No_dev;
          kind = vect.kind;
          is_sub =
            Some (((dep vect.is_sub) + 1), _start, _ok_r, _ko_r, vect);
          sub = [];
          vec_id = !vec_id;
					seek = 0;
        }
    | CustomArray (cA, c) ->
        {
          device = (-1);
          vector = vect.vector;
          cuda_device_vec =
            Array.create ((Devices.cuda_devices ()) + 1) (init_cuda_device_vec ());
          opencl_device_vec =
            Array.create ((Devices.opencl_devices ()) + 1)
              (init_opencl_device_vec ());
          length = _len;
          dev = No_dev;
          kind = vect.kind;
          is_sub =
            Some (((dep vect.is_sub) + 1), _start, _ok_r, _ko_r, vect);
          sub = [];
          vec_id = !vec_id;
					seek = 0;
        })



let of_bigarray_shr kind b = 
    incr vec_id;
    let open Devices in 
     {
      device = (-1);
      vector = Bigarray b;
      cuda_device_vec = Array.create (cuda_devices() +1) (init_cuda_device_vec ());
      opencl_device_vec = Array.create (opencl_devices() +1) (init_opencl_device_vec ());
      length =  Bigarray.Array1.dim b;
      dev = No_dev;
      kind = kind;
      is_sub = None;
      sub = [];
      vec_id = !vec_id;
			seek = 0 }
      
let to_bigarray_shr v =
  match v.vector with
    | Bigarray b -> b
    | _  ->  raise (Invalid_argument "v")
