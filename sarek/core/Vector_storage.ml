(******************************************************************************
 * Vector storage and creation helpers (split from Vector.ml)
 ******************************************************************************)

open Vector_types

(** Global vector ID counter *)
let next_id = ref 0

(** {1 Creation} *)

(** Create a new scalar vector on CPU *)
let create_scalar (sk : ('a, 'b) scalar_kind) ?(dev : Device.t option)
    (length : int) : ('a, 'b) t =
  incr next_id ;
  let ba_kind = to_bigarray_kind sk in
  let ba = Bigarray.Array1.create ba_kind Bigarray.c_layout length in
  let vec =
    {
      host = Bigarray_storage ba;
      device_buffers = Hashtbl.create 4;
      length;
      kind = Scalar sk;
      location = CPU;
      auto_sync = true;
      id = !next_id;
    }
  in
  (match dev with Some d -> vec.location <- Stale_GPU d | None -> ()) ;
  vec

(** Create from kind (allocates storage for custom types too) *)
let create : type a b. (a, b) kind -> ?dev:Device.t -> int -> (a, b) t =
 fun kind ?dev length ->
  match kind with
  | Scalar sk -> create_scalar sk ?dev length
  | Custom c -> (
      incr next_id ;
      let byte_size = length * c.elem_size in
      let ptr = Ctypes.(allocate_n (array 1 char) ~count:byte_size) in
      let ptr =
        Ctypes.coerce Ctypes.(ptr (array 1 char)) Ctypes.(ptr void) ptr
      in
      let vec =
        {
          host = Custom_storage {ptr; custom = c; length};
          device_buffers = Hashtbl.create 4;
          length;
          kind = Custom c;
          location = CPU;
          auto_sync = true;
          id = !next_id;
        }
      in
      match dev with
      | Some d ->
          vec.location <- Stale_GPU d ;
          vec
      | None -> vec)

(** Create a custom vector with explicit type descriptor *)
let create_custom (c : 'a custom_type) ?(dev : Device.t option) (length : int) :
    ('a, unit) t =
  create (Custom c) ?dev length

(** {1 Creation from Existing Data} *)

(** Create from existing Bigarray (shares memory) *)
let of_bigarray (sk : ('a, 'b) scalar_kind)
    (ba : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t) : ('a, 'b) t =
  incr next_id ;
  {
    host = Bigarray_storage ba;
    device_buffers = Hashtbl.create 4;
    length = Bigarray.Array1.dim ba;
    kind = Scalar sk;
    location = CPU;
    auto_sync = true;
    id = !next_id;
  }

(** Create from existing ctypes pointer (shares memory) *)
let of_ctypes_ptr (c : 'a custom_type) (ptr : unit Ctypes.ptr) (length : int) :
    ('a, unit) t =
  incr next_id ;
  {
    host = Custom_storage {ptr; custom = c; length};
    device_buffers = Hashtbl.create 4;
    length;
    kind = Custom c;
    location = CPU;
    auto_sync = true;
    id = !next_id;
  }

(** {1 Accessors} *)

let to_bigarray : type a b.
    (a, b) t -> (a, b, Bigarray.c_layout) Bigarray.Array1.t =
 fun vec ->
  match vec.host with
  | Bigarray_storage ba -> ba
  | Custom_storage _ -> invalid_arg "to_bigarray: vector uses custom storage"

(** Create from OCaml list *)
let of_list : type a b. (a, b) kind -> a list -> (a, b) t =
 fun kind lst ->
  let len = List.length lst in
  let vec = create kind len in
  List.iteri
    (fun i v ->
      match vec.host with
      | Bigarray_storage ba -> Bigarray.Array1.set ba i v
      | Custom_storage {ptr; custom; _} -> custom.set ptr i v)
    lst ;
  vec

(** Create from OCaml array *)
let of_array : type a b. (a, b) kind -> a array -> (a, b) t =
 fun kind arr ->
  let vec = create kind (Array.length arr) in
  Array.iteri
    (fun i v ->
      match vec.host with
      | Bigarray_storage ba -> Bigarray.Array1.set ba i v
      | Custom_storage {ptr; custom; _} -> custom.set ptr i v)
    arr ;
  vec

(** {1 Device buffer bookkeeping} *)

(** Check if vector has buffer on specific device *)
let has_buffer (vec : ('a, 'b) t) (dev : Device.t) : bool =
  Hashtbl.mem vec.device_buffers dev.id

(** Get device buffer if allocated *)
let get_buffer (vec : ('a, 'b) t) (dev : Device.t) : device_buffer option =
  Hashtbl.find_opt vec.device_buffers dev.id

(** {1 Copy & Slicing} *)

(** Copy vector (CPU data only). Caller must ensure sync if needed. *)
let copy_host_only (type a b) (vec : (a, b) t) : (a, b) t =
  incr next_id ;
  let host =
    match vec.host with
    | Bigarray_storage ba ->
        let new_ba =
          Bigarray.Array1.create
            (Bigarray.Array1.kind ba)
            Bigarray.c_layout
            vec.length
        in
        Bigarray.Array1.blit ba new_ba ;
        Bigarray_storage new_ba
    | Custom_storage {ptr; custom; length} ->
        let byte_size = length * custom.elem_size in
        let new_ptr = Ctypes.(allocate_n (array 1 char) ~count:byte_size) in
        let new_ptr =
          Ctypes.coerce Ctypes.(ptr (array 1 char)) Ctypes.(ptr void) new_ptr
        in
        for i = 0 to length - 1 do
          custom.set new_ptr i (custom.get ptr i)
        done ;
        Custom_storage {ptr = new_ptr; custom; length}
  in
  {
    host;
    device_buffers = Hashtbl.create 4;
    length = vec.length;
    kind = vec.kind;
    location = CPU;
    auto_sync = vec.auto_sync;
    id = !next_id;
  }

(** Create subvector that views the same host storage with an offset *)
let sub_vector_host (type a b) (vec : (a, b) t) ~(start : int) ~(len : int) :
    (a, b) t =
  if start < 0 || start + len > vec.length then
    invalid_arg
      (Printf.sprintf
         "sub_vector: range [%d, %d) out of bounds [0, %d)"
         start
         (start + len)
         vec.length) ;
  incr next_id ;
  let host =
    match vec.host with
    | Bigarray_storage ba -> Bigarray_storage (Bigarray.Array1.sub ba start len)
    | Custom_storage {ptr; custom; _} ->
        let byte_offset = start * custom.elem_size in
        let raw_addr = Ctypes.raw_address_of_ptr ptr in
        let offset_addr =
          Nativeint.add raw_addr (Nativeint.of_int byte_offset)
        in
        let offset_ptr = Ctypes.ptr_of_raw_address offset_addr in
        Custom_storage {ptr = offset_ptr; custom; length = len}
  in
  {
    host;
    device_buffers = Hashtbl.create 4;
    length = len;
    kind = vec.kind;
    location = CPU;
    auto_sync = vec.auto_sync;
    id = !next_id;
  }

(** Partition host storage evenly across devices (no device buffers) *)
let partition_host (type a b) (vec : (a, b) t) (devices : Device.t array) :
    (a, b) t array =
  let n = Array.length devices in
  if n = 0 then [||]
  else
    let base = vec.length / n in
    let rem = vec.length mod n in
    Array.init n (fun i ->
        let extra = if i < rem then 1 else 0 in
        let len = base + extra in
        let start = (i * base) + min i rem in
        sub_vector_host vec ~start ~len)
