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

type float32 = float
type float64 = float
type complex32 = Complex.t

type int32 = Int32.t
type int64 = Int64.t
type _ kind =
    Float32 : float32 kind
  | Char : char kind
  | Float64 : float64 kind
  | Int32 : int32 kind
  | Int64 : int64 kind
  | Complex32 : complex32 kind
  | Custom : 'a custom -> 'a  kind

(* shortcuts *)

let char = Char
let int32 = Int32
let int64 = Int64
let float32 = Float32
let float64 = Float64
let complex32 = Complex32
let custom a = Custom a

type 'a ptr
type 'a get_f = 'a ptr -> int -> 'a
type 'a set_f = 'a ptr -> int -> 'a -> unit

type 'a host_vec =
  {
    ptr : 'a ptr;
    get : 'a get_f;
    set : 'a set_f;
  }


type _ spoc_vec =
  | CustomArray : (customarray * 'a custom) -> 'a spoc_vec
  | Host_vec : 'a host_vec -> 'a spoc_vec

external float32_of_float : float -> float = "float32_of_float"
external float_of_float32 : float -> float = "float_of_float32"

type vec_device =
  | No_dev
  | Dev of Devices.device
  | Transferring of Devices.device

type 'a sub = int * int * int * int * 'a vector
(* sub_vector depth * start * ok range * ko range * parent vector *)

and 'a vector = {
  mutable device : int; (* -1 -> CPU, 0+ -> device *)
  vector : 'a spoc_vec;
  cuda_device_vec : device_vec array;
  opencl_device_vec : device_vec array;
  length : int;
  mutable dev: vec_device;
  kind: 'a kind;
  mutable is_sub: 'a sub option;(*('a, 'b) sub;*)
  sub : 'a vector list;
  vec_id: int;
  mutable seek:int
}

external init_cuda_device_vec: unit -> device_vec = "spoc_init_cuda_device_vec"
external init_opencl_device_vec : unit -> device_vec = "spoc_init_opencl_device_vec"

external create_custom : 'a custom -> int ->  customarray = "spoc_create_custom"



external host_alloc :
  int -> int -> 'a ptr = "host_alloc"
external get_float32 :
  'a ptr -> int -> 'a = "get_float32"
external set_float32 :
  'a ptr -> int -> 'a -> unit = "set_float32"

external get_float64 :
  'a ptr -> int -> 'a = "get_float64"
external set_float64 :
  'a ptr -> int -> 'a -> unit = "set_float64"

external get_int32 :
  'a ptr -> int -> 'a = "get_int32"
external set_int32 :
  'a ptr -> int -> 'a -> unit = "set_int32"

external get_int64 :
  'a ptr -> int -> 'a = "get_int64"
external set_int64 :
  'a ptr -> int -> 'a -> unit = "set_int64"

external get_char :
  'a ptr -> int -> 'a = "get_char"
external set_char :
  'a ptr -> int -> 'a -> unit = "set_char"

external get_complex32 :
  'a ptr -> int -> 'a = "get_complex32"
external set_complex32 :
  'a ptr -> int -> 'a -> unit = "set_complex32"



external cuda_custom_alloc_vect :
  'a vector -> int -> Devices.generalInfo -> unit =
  "spoc_cuda_custom_alloc_vect"

external cuda_alloc_vect :
  'a vector -> int -> Devices.generalInfo -> unit =
  "spoc_cuda_alloc_vect"

external opencl_alloc_vect :
  'a vector -> int -> Devices.generalInfo -> unit =
  "spoc_opencl_alloc_vect"
external opencl_custom_alloc_vect :
  'a vector -> int -> Devices.generalInfo -> unit =
  "spoc_opencl_alloc_vect"

let vec_id = ref 0

(******************************************************************************************************)

let emitVect _ _ = ()

#ifdef SPOC_PROFILE
external printVector : int -> int -> int -> int -> string -> bool -> int -> int -> int -> int -> int -> unit = "print_vector_bytecode" "print_vector_native"


let emitVect (vect : 'a vector) length =
    let isSub = match vect.is_sub with
    | None -> "false"
    | Some x -> "true" in
    let dev = vect.device in
    let id = vect.vec_id in
    let kindS = match vect.kind with
    | Char -> "char"
    | Float32 -> "float32"
    | Int32 -> "int32"
    | Float64 -> "float64"
    | Int64 -> "int64"
    | Complex32 -> "complex32"
    | _ -> "unknown" in
    let size = match vect.kind with
    | Char -> 1
    | Float32 | Int32 -> 4
    | Float64 | Int64 | Complex32 -> 8
    | _ -> 0 in
    if isSub = "true" then begin
      let (depth, start, ok_range, ko_range, parent) = match vect.is_sub with
        | None -> failwith "Subvector not found"
        | Some e -> e in
      let parentId = parent.vec_id in
      printVector id dev length size kindS true depth start ok_range ko_range parentId;
    end
    else
      printVector id dev length size kindS false 1 1 1 1 1;
    
#endif
(*******************************************************************************************************)


external sizeofFloat32  : unit -> int = "sizeofFloat32"
external sizeofFloat64  : unit -> int = "sizeofFloat64"
external sizeofChar  : unit -> int = "sizeofChar"
external sizeofInt32  : unit -> int = "sizeofInt32"
external sizeofInt64  : unit -> int = "sizeofInt64"
external sizeofComplex32  : unit -> int = "sizeofComplex32"

let printEvent _ = ()

#ifdef SPOC_PROFILE
external printEvent : string -> unit = "print_event"
#endif


                                         

let create (type a) (kind: a kind) ?dev size : (a vector)=
  printEvent "VectorCreation";
  incr vec_id;
  let vec = 
    {
      device = -1;
      vector =
        (match kind with
         | Float32 ->
           Host_vec
             {
               ptr =  (host_alloc (sizeofFloat32 ()) size);
               get = get_float32;
               set = set_float32;
             }
         | Char ->
           Host_vec {
             ptr = (host_alloc (sizeofChar ()) size);
             get = get_char;
             set = set_char;
           }
         | Float64 ->
           Host_vec {
             ptr = (host_alloc (sizeofFloat64 ()) size);
             get = get_float64;
             set = set_float64;
           }
         | Int32 ->
           Host_vec
             {
               ptr = (host_alloc (sizeofInt32 ()) size);
               get = get_int32;
               set = set_int32;
           }
         | Int64 -> Host_vec
                                     {
               ptr = (host_alloc (sizeofInt64 ()) size);
               get = get_int64;
               set = set_int64;
           }
         | Complex32 -> Host_vec
                                         {
               ptr = (host_alloc (sizeofComplex32 ()) size);
               get = get_complex32;
               set = set_complex32;
           }
             (*(let res = (Bigarray.Array1.create x Bigarray.c_layout size)
               in
               Bigarray res)*)        
         | Custom c ->
           CustomArray ((create_custom c size), c)
        );
      cuda_device_vec = Array.make (Devices.cuda_devices() +1) (init_cuda_device_vec ());
      opencl_device_vec = Array.make (Devices.opencl_devices() +1) (init_opencl_device_vec ());
      length = size;
      dev = No_dev;
      kind = kind;
      is_sub = None;
      sub = [];
      vec_id = !vec_id;
      seek = 0; }
    (* { *)
    (*    device = -1; *)
    (*    vector =  *)
    (*    cuda_device_vec = Array.create (Devices.cuda_devices() +1) (init_cuda_device_vec ()); *)
    (*    opencl_device_vec = Array.create (Devices.opencl_devices() +1) (init_opencl_device_vec ()); *)
    (*    length = size; *)
    (*    dev = No_dev; *)
    (*    kind = kind; *)
    (*    is_sub = None; *)
    (*    sub = []; *)
    (*    vec_id = !vec_id; seek = 0; } *)
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
    emitVect vec size;
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
  | CustomArray c -> (snd c).set (fst c) idx value
  | Host_vec v -> v.set v.ptr idx value

let unsafe_get vect idx =
  match vect.vector with
  | CustomArray c -> (snd c).get (fst c) idx
  | Host_vec v -> v.get v.ptr idx 

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

(* external sub_host_vec : ('a,'b) host_vec -> int -> int -> ('a,'b) host_vec = "spoc_sub_host_vec" *)
  
(* let sub_vector (vect : ('a, 'b) vector) _start _len = *)
(*   incr vec_id; *)
(*   { *)
(*     device = (-1); *)
(*     vector = *)
(*       (match vect.vector with *)
(*        | Host_vec v -> Host_vec (sub_host_vec  v _start _len) *)
(*        | Bigarray b -> Bigarray (Bigarray.Array1.sub b _start _len) *)
(*        | CustomArray (cA, c) -> *)
(*          CustomArray ((sub_custom_array cA c _start), c);  ); *)
(*     cuda_device_vec = *)
(*       Array.create ((Devices.cuda_devices ()) + 1) (init_cuda_device_vec ()); *)
(*     opencl_device_vec = *)
(*       Array.create ((Devices.opencl_devices ()) + 1) *)
(*         (init_opencl_device_vec ()); *)
(*     length = _len; *)
(*     dev = No_dev; *)
(*     kind = vect.kind; *)
(*     is_sub = None; *)
(*     sub = []; *)
(*     vec_id = !vec_id; *)
(*     seek = _start; *)
(*      } *)
   
(*      (\* { *\) *)
(*      (\*   device = (-1); *\) *)
(*      (\*   vector =  *\) *)
(*      (\*   cuda_device_vec = *\) *)
(*      (\*     Array.create ((Devices.cuda_devices ()) + 1) (init_cuda_device_vec ()); *\) *)
(*      (\*   opencl_device_vec = *\) *)
(*      (\*     Array.create ((Devices.opencl_devices ()) + 1) *\) *)
(*      (\*       (init_opencl_device_vec ()); *\) *)
(*      (\*   length = _len; *\) *)
(*      (\*   dev = No_dev; *\) *)
(*      (\*   kind = vect.kind; *\) *)
(*      (\*   is_sub = None; *\) *)
(*      (\*   sub = []; *\) *)
(*      (\*   vec_id = !vec_id; *\) *)
(*      (\*   seek = 0; *\) *)
(*      (\* }) *\) *)

let dep = function | None -> 0 | Some (a, _, _, _, _) -> a

let sub_vector (vect : 'a vector) _start _ok_r
    _ko_r _len =
  incr vec_id;
(*  (match vect.vector with
    | Bigarray b ->*)
     {
       device = (-1);
       vector = vect.vector;
       cuda_device_vec =
         Array.make ((Devices.cuda_devices ()) + 1) (init_cuda_device_vec ());
       opencl_device_vec =
         Array.make ((Devices.opencl_devices ()) + 1)
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
(*   | CustomArray (cA, c) ->
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
     })*)



external bigarray_adress : 'b -> int -> int -> 'a ptr = "spoc_bigarray_adress"

let of_bigarray_shr (type a) (kind : a kind) b = 
  incr vec_id;
  let open Devices in
  {
    device = (-1);
    vector = (*Bigarray b;*)
      Host_vec
        (
         match kind with
         | Float32 ->
           {
             ptr =  (bigarray_adress b (sizeofFloat32 ()) (Bigarray.Array1.dim b));
             get = get_float32;
             set = set_float32;
           }
         | Char ->
           {
             ptr = (bigarray_adress b (sizeofChar ()) (Bigarray.Array1.dim b));
             get = get_char;
             set = set_char;
           }
         | Float64 ->
           {
             ptr = (bigarray_adress b (sizeofFloat64 ()) (Bigarray.Array1.dim b));
             get = get_float64;
             set = set_float64;
           }
         | Int32 ->
           {
             ptr = (bigarray_adress b (sizeofInt32 ()) (Bigarray.Array1.dim b));
             get = get_int32;
             set = set_int32;
           }
         | Int64 ->
           {
             ptr = (bigarray_adress b (sizeofInt64 ()) (Bigarray.Array1.dim b));
             get = get_int64;
             set = set_int64;
           }
         | Complex32 ->
           {
             ptr = (bigarray_adress b (sizeofComplex32 ()) (Bigarray.Array1.dim b));
             get = get_complex32;
             set = set_complex32;
           }
         | Custom _ -> assert false
        ); 
    cuda_device_vec = Array.make (cuda_devices() +1) (init_cuda_device_vec ());
    opencl_device_vec = Array.make (opencl_devices() +1) (init_opencl_device_vec ());
    length =  Bigarray.Array1.dim b;
    dev = No_dev;
    kind = kind;
    is_sub = None;
    sub = [];
    vec_id = !vec_id;
    seek = 0 }

let to_bigarray_shr v =
  match v.vector with
  | _  ->  raise (Invalid_argument "v")
