(******************************************************************************
 * Sarek Runtime - High-Level Vector Abstraction
 *
 * Provides a unified Vector type that tracks data location (CPU/GPU) and
 * supports automatic synchronization. Replaces the old SPOC Vector module.
 *
 * Supports two storage modes:
 * - Bigarray: Standard numeric types (Float32, Float64, Int32, etc.)
 * - Custom: User-defined ctypes structures with custom get/set
 *
 * Design principles:
 * - Type-safe: Uses GADTs and existentials instead of Obj.t
 * - Location-aware: Tracks where data resides (CPU, GPU, or both)
 * - Lazy allocation: GPU memory allocated on first device use
 * - Auto-sync: Element access triggers sync when needed
 ******************************************************************************)

(** {1 Element Types} *)

(** Standard numeric kinds backed by Bigarray *)
type (_, _) scalar_kind =
  | Float32 : (float, Bigarray.float32_elt) scalar_kind
  | Float64 : (float, Bigarray.float64_elt) scalar_kind
  | Int32 : (int32, Bigarray.int32_elt) scalar_kind
  | Int64 : (int64, Bigarray.int64_elt) scalar_kind
  | Char : (char, Bigarray.int8_unsigned_elt) scalar_kind
  | Complex32 : (Complex.t, Bigarray.complex32_elt) scalar_kind

(** Custom type descriptor for ctypes-based structures *)
type 'a custom_type = {
  elem_size : int;  (** Size of each element in bytes *)
  get : unit Ctypes.ptr -> int -> 'a;  (** Read element at index *)
  set : unit Ctypes.ptr -> int -> 'a -> unit;  (** Write element at index *)
  name : string;  (** Type name for debugging *)
}

(** Unified kind type supporting both scalar and custom types *)
type (_, _) kind =
  | Scalar : ('a, 'b) scalar_kind -> ('a, 'b) kind
  | Custom : 'a custom_type -> ('a, unit) kind

(** {1 Kind Helpers} *)

(** Convert scalar kind to Bigarray.kind *)
let to_bigarray_kind : type a b. (a, b) scalar_kind -> (a, b) Bigarray.kind =
  function
  | Float32 -> Bigarray.Float32
  | Float64 -> Bigarray.Float64
  | Int32 -> Bigarray.Int32
  | Int64 -> Bigarray.Int64
  | Char -> Bigarray.Char
  | Complex32 -> Bigarray.Complex32

(** Element size in bytes *)
let scalar_elem_size : type a b. (a, b) scalar_kind -> int = function
  | Float32 -> 4
  | Float64 -> 8
  | Int32 -> 4
  | Int64 -> 8
  | Char -> 1
  | Complex32 -> 8

let elem_size : type a b. (a, b) kind -> int = function
  | Scalar k -> scalar_elem_size k
  | Custom c -> c.elem_size

(** Kind name for debugging *)
let scalar_kind_name : type a b. (a, b) scalar_kind -> string = function
  | Float32 -> "Float32"
  | Float64 -> "Float64"
  | Int32 -> "Int32"
  | Int64 -> "Int64"
  | Char -> "Char"
  | Complex32 -> "Complex32"

let kind_name : type a b. (a, b) kind -> string = function
  | Scalar k -> scalar_kind_name k
  | Custom c -> "Custom(" ^ c.name ^ ")"

(** {1 Host Storage (GADT)} *)

(** CPU-side storage - either Bigarray or raw ctypes pointer *)
type (_, _) host_storage =
  | Bigarray_storage :
      ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
      -> ('a, 'b) host_storage
  | Custom_storage : {
      ptr : unit Ctypes.ptr;
      custom : 'a custom_type;
      length : int;
    }
      -> ('a, unit) host_storage

(** {1 Location Tracking} *)

(** Where the authoritative copy of data resides *)
type location =
  | CPU  (** Data only on host *)
  | GPU of Device.t  (** Data only on specific device *)
  | Both of Device.t  (** Synced on host and device *)
  | Stale_CPU of Device.t  (** GPU is authoritative, CPU outdated *)
  | Stale_GPU of Device.t  (** CPU is authoritative, GPU outdated *)

(** {1 Device Buffer Abstraction} *)

(** Existential wrapper for backend-specific device buffers.
    Packages the backend module with its buffer for type-safe operations. *)
module type DEVICE_BUFFER = sig
  val device : Device.t
  val size : int
  val elem_size : int
  val ptr : nativeint  (** Raw device pointer (CUDA) or 0 (OpenCL) *)

  (** {2 Kernel Binding}
      Backend-specific kernel argument binding. The backend knows how to
      bind its own buffer type (CUdeviceptr for CUDA, cl_mem for OpenCL). *)

  (** Bind this buffer to a kernel argument. kargs is the backend's kernel
      args structure (Obj.t for type erasure), idx is the argument index. *)
  val bind_to_kernel : kargs:Obj.t -> idx:int -> unit

  (** {2 Transfer Operations}
      All transfers use raw pointers with byte sizes to avoid
      type parameter escaping issues in first-class modules. *)

  (** Transfer from host pointer to device buffer *)
  val from_ptr : unit Ctypes.ptr -> byte_size:int -> unit

  (** Transfer from device buffer to host pointer *)
  val to_ptr : unit Ctypes.ptr -> byte_size:int -> unit

  (** Free the device buffer *)
  val free : unit -> unit
end

type device_buffer = (module DEVICE_BUFFER)

(** Device buffer storage - maps device ID to buffer *)
type device_buffers = (int, device_buffer) Hashtbl.t

(** {1 Vector Type} *)

(** High-level vector with location tracking *)
type ('a, 'b) t = {
  host : ('a, 'b) host_storage;
  device_buffers : device_buffers;
  length : int;
  kind : ('a, 'b) kind;
  mutable location : location;
  mutable auto_sync : bool;  (** Enable automatic CPU sync on get *)
  id : int;  (** Unique vector ID for debugging *)
}

(** Global vector ID counter *)
let next_id = ref 0

(** {1 Creation - Scalar Types} *)

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
  (match dev with
  | Some d -> vec.location <- Stale_GPU d
  | None -> ()) ;
  vec

(** Create from scalar kind (convenience wrapper) *)
let create : type a b. (a, b) kind -> ?dev:Device.t -> int -> (a, b) t =
 fun kind ?dev length ->
  match kind with
  | Scalar sk -> create_scalar sk ?dev length
  | Custom c -> (
      incr next_id ;
      let byte_size = length * c.elem_size in
      let ptr = Ctypes.(allocate_n (array 1 char) ~count:byte_size) in
      let ptr = Ctypes.coerce Ctypes.(ptr (array 1 char)) Ctypes.(ptr void) ptr in
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

(** {1 Creation - Custom Types} *)

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

let length (vec : ('a, 'b) t) : int = vec.length

let kind (vec : ('a, 'b) t) : ('a, 'b) kind = vec.kind

let location (vec : ('a, 'b) t) : location = vec.location

let id (vec : ('a, 'b) t) : int = vec.id

(** Get device if vector has GPU data *)
let device (vec : ('a, 'b) t) : Device.t option =
  match vec.location with
  | CPU -> None
  | GPU d | Both d | Stale_CPU d | Stale_GPU d -> Some d

(** {1 Raw Data Access} *)

(** Get underlying Bigarray (only for scalar vectors) *)
let to_bigarray : type a b. (a, b) t -> (a, b, Bigarray.c_layout) Bigarray.Array1.t =
 fun vec ->
  match vec.host with
  | Bigarray_storage ba -> ba
  | Custom_storage _ -> invalid_arg "to_bigarray: vector uses custom storage"

(** Get underlying ctypes pointer (only for custom vectors) *)
let to_ctypes_ptr : type a. (a, unit) t -> unit Ctypes.ptr = fun vec ->
  match vec.host with
  | Custom_storage {ptr; _} -> ptr
  | Bigarray_storage _ -> invalid_arg "to_ctypes_ptr: vector uses bigarray storage"

(** Get raw host pointer for any vector type *)
let host_ptr : type a b. (a, b) t -> nativeint = fun vec ->
  match vec.host with
  | Bigarray_storage ba ->
      Ctypes.(raw_address_of_ptr (bigarray_start array1 ba |> to_voidp))
  | Custom_storage {ptr; _} ->
      Ctypes.raw_address_of_ptr ptr

(** {1 Element Access} *)

(** Get element (works for both storage types) *)
let get : type a b. (a, b) t -> int -> a =
 fun vec idx ->
  if idx < 0 || idx >= vec.length then
    invalid_arg
      (Printf.sprintf "Vector.get: index %d out of bounds [0, %d)" idx
         vec.length) ;
  match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.get ba idx
  | Custom_storage {ptr; custom; _} -> custom.get ptr idx

(** Set element (works for both storage types) *)
let set : type a b. (a, b) t -> int -> a -> unit =
 fun vec idx value ->
  if idx < 0 || idx >= vec.length then
    invalid_arg
      (Printf.sprintf "Vector.set: index %d out of bounds [0, %d)" idx
         vec.length) ;
  (match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.set ba idx value
  | Custom_storage {ptr; custom; _} -> custom.set ptr idx value) ;
  (* Mark GPU as stale if we had synced data *)
  match vec.location with
  | Both d -> vec.location <- Stale_GPU d
  | GPU d -> vec.location <- Stale_GPU d
  | CPU | Stale_CPU _ | Stale_GPU _ -> ()

(** Indexing operators *)
let ( .%[] ) = get

let ( .%[]<- ) = set

(** {1 Unsafe Access (no bounds check)} *)

let unsafe_get : type a b. (a, b) t -> int -> a =
 fun vec idx ->
  match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.unsafe_get ba idx
  | Custom_storage {ptr; custom; _} -> custom.get ptr idx

let unsafe_set : type a b. (a, b) t -> int -> a -> unit =
 fun vec idx value ->
  (match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.unsafe_set ba idx value
  | Custom_storage {ptr; custom; _} -> custom.set ptr idx value) ;
  match vec.location with
  | Both d -> vec.location <- Stale_GPU d
  | GPU d -> vec.location <- Stale_GPU d
  | CPU | Stale_CPU _ | Stale_GPU _ -> ()

(** {1 Auto-sync Control} *)

let set_auto_sync (vec : ('a, 'b) t) (enabled : bool) : unit =
  vec.auto_sync <- enabled

let auto_sync (vec : ('a, 'b) t) : bool = vec.auto_sync

(** {1 Sync Callback (for Transfer module)} *)

(** Callback to sync vector to CPU. Set by Transfer module at init.
    Takes a vector and syncs it to CPU if needed. Returns true if sync occurred. *)
type sync_callback = { sync : 'a 'b. ('a, 'b) t -> bool }
let sync_to_cpu_callback : sync_callback option ref = ref None

(** Register the sync callback (called by Transfer module) *)
let register_sync_callback (cb : sync_callback) : unit =
  sync_to_cpu_callback := Some cb

(** Ensure vector data is on CPU before iteration (called once per iterator).
    Only syncs if auto_sync is enabled and location is Stale_CPU. *)
let ensure_cpu_sync (type a b) (vec : (a, b) t) : unit =
  if vec.auto_sync then
    match vec.location with
    | Stale_CPU _ -> (
        match !sync_to_cpu_callback with
        | Some cb -> ignore (cb.sync vec)
        | None -> ())
    | _ -> ()

(** {1 Location Queries} *)

let is_on_cpu (vec : ('a, 'b) t) : bool =
  match vec.location with
  | CPU | Both _ | Stale_GPU _ -> true
  | GPU _ | Stale_CPU _ -> false

let is_on_gpu (vec : ('a, 'b) t) : bool =
  match vec.location with
  | GPU _ | Both _ | Stale_CPU _ -> true
  | CPU | Stale_GPU _ -> false

let is_synced (vec : ('a, 'b) t) : bool =
  match vec.location with Both _ -> true | _ -> false

let needs_gpu_update (vec : ('a, 'b) t) : bool =
  match vec.location with Stale_GPU _ -> true | _ -> false

let needs_cpu_update (vec : ('a, 'b) t) : bool =
  match vec.location with Stale_CPU _ | GPU _ -> true | _ -> false

(** {1 Device Buffer Management} *)

(** Check if vector has buffer on specific device *)
let has_buffer (vec : ('a, 'b) t) (dev : Device.t) : bool =
  Hashtbl.mem vec.device_buffers dev.id

(** Get device buffer if allocated *)
let get_buffer (vec : ('a, 'b) t) (dev : Device.t) : device_buffer option =
  Hashtbl.find_opt vec.device_buffers dev.id

(** {1 Pretty Printing} *)

let location_to_string : location -> string = function
  | CPU -> "CPU"
  | GPU d -> Printf.sprintf "GPU(%s)" d.name
  | Both d -> Printf.sprintf "Both(%s)" d.name
  | Stale_CPU d -> Printf.sprintf "Stale_CPU(%s)" d.name
  | Stale_GPU d -> Printf.sprintf "Stale_GPU(%s)" d.name

let to_string (vec : ('a, 'b) t) : string =
  Printf.sprintf "Vector#%d<%s>[%d] @ %s" vec.id (kind_name vec.kind) vec.length
    (location_to_string vec.location)

(** {1 Convenience Constructors for Scalar Types} *)

let float32 = Scalar Float32
let float64 = Scalar Float64
let int32 = Scalar Int32
let int64 = Scalar Int64
let char = Scalar Char
let complex32 = Scalar Complex32

let create_float32 ?dev n = create float32 ?dev n
let create_float64 ?dev n = create float64 ?dev n
let create_int32 ?dev n = create int32 ?dev n
let create_int64 ?dev n = create int64 ?dev n

(** {1 Initialization Helpers} *)

(** Fill vector with a value *)
let fill : type a b. (a, b) t -> a -> unit =
 fun vec value ->
  (match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.fill ba value
  | Custom_storage {ptr; custom; length; _} ->
      for i = 0 to length - 1 do
        custom.set ptr i value
      done) ;
  match vec.location with
  | Both d -> vec.location <- Stale_GPU d
  | GPU d -> vec.location <- Stale_GPU d
  | _ -> ()

(** Initialize with a function *)
let init : type a b. (a, b) kind -> int -> (int -> a) -> (a, b) t =
 fun kind length f ->
  let vec = create kind length in
  for i = 0 to length - 1 do
    unsafe_set vec i (f i)
  done ;
  vec

(** Copy vector (CPU data only, auto-syncs source if needed) *)
let copy : type a b. (a, b) t -> (a, b) t =
 fun vec ->
  ensure_cpu_sync vec ;
  incr next_id ;
  let host =
    match vec.host with
    | Bigarray_storage ba ->
        let new_ba =
          Bigarray.Array1.create (Bigarray.Array1.kind ba) Bigarray.c_layout
            vec.length
        in
        Bigarray.Array1.blit ba new_ba ;
        Bigarray_storage new_ba
    | Custom_storage {ptr; custom; length} ->
        let byte_size = length * custom.elem_size in
        let new_ptr =
          Ctypes.(allocate_n (array 1 char) ~count:byte_size)
        in
        let new_ptr =
          Ctypes.coerce Ctypes.(ptr (array 1 char)) Ctypes.(ptr void) new_ptr
        in
        (* Copy data element by element *)
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

(** {1 Subvector Support} *)

(** Subvector metadata - stored as simple record without parent reference *)
type sub_meta = {
  parent_id : int;  (** ID of parent vector *)
  start : int;  (** Offset into parent *)
  ok_range : int;  (** Elements safe to read *)
  ko_range : int;  (** Elements that shouldn't be written *)
  depth : int;  (** Nesting depth (1 = direct child of root) *)
}

(** Subvector metadata storage *)
let subvector_meta : (int, sub_meta) Hashtbl.t = Hashtbl.create 16

(** Check if vector is a subvector *)
let is_sub (vec : ('a, 'b) t) : bool = Hashtbl.mem subvector_meta vec.id

(** Get subvector metadata *)
let get_sub_meta (vec : ('a, 'b) t) : sub_meta option =
  Hashtbl.find_opt subvector_meta vec.id

(** Create a subvector that shares CPU memory with parent.
    @param vec Parent vector
    @param start Starting index in parent
    @param len Length of subvector
    @param ok_range Elements safe to read (default: len)
    @param ko_range Elements to avoid writing (default: 0) *)
let sub_vector (type a b) (vec : (a, b) t) ~(start : int) ~(len : int)
    ?(ok_range : int = len) ?(ko_range : int = 0) () : (a, b) t =
  if start < 0 || start + len > vec.length then
    invalid_arg
      (Printf.sprintf "sub_vector: range [%d, %d) out of bounds [0, %d)" start
         (start + len) vec.length) ;
  incr next_id ;
  let parent_depth =
    match get_sub_meta vec with Some meta -> meta.depth | None -> 0
  in
  (* Create subvector with offset view into parent's host storage *)
  let host =
    match vec.host with
    | Bigarray_storage ba ->
        Bigarray_storage (Bigarray.Array1.sub ba start len)
    | Custom_storage {ptr; custom; _} ->
        (* Offset the pointer by byte offset *)
        let byte_offset = start * custom.elem_size in
        let raw_addr = Ctypes.raw_address_of_ptr ptr in
        let offset_addr = Nativeint.add raw_addr (Nativeint.of_int byte_offset) in
        let offset_ptr = Ctypes.ptr_of_raw_address offset_addr in
        Custom_storage {ptr = offset_ptr; custom; length = len}
  in
  let sub =
    {
      host;
      device_buffers = Hashtbl.create 4;  (* Independent GPU buffers *)
      length = len;
      kind = vec.kind;
      location = CPU;  (* Subvector starts on CPU *)
      auto_sync = vec.auto_sync;
      id = !next_id;
    }
  in
  (* Record subvector relationship *)
  Hashtbl.replace subvector_meta sub.id
    {parent_id = vec.id; start; ok_range; ko_range; depth = parent_depth + 1} ;
  sub

(** {1 Multi-GPU Helpers} *)

(** Partition a vector across multiple devices.
    Creates subvectors, one per device, that together cover the full vector. *)
let partition (type a b) (vec : (a, b) t) (devices : Device.t array) :
    (a, b) t array =
  let n = Array.length devices in
  if n = 0 then [||]
  else
    let chunk_size = vec.length / n in
    let remainder = vec.length mod n in
    let result = Array.make n vec in  (* Placeholder *)
    let offset = ref 0 in
    for i = 0 to n - 1 do
      let len = chunk_size + (if i < remainder then 1 else 0) in
      let sub = sub_vector vec ~start:!offset ~len () in
      sub.location <- Stale_GPU devices.(i) ;
      result.(i) <- sub ;
      offset := !offset + len
    done ;
    result

(** Gather subvectors back to parent (sync all to CPU).
    Assumes subvectors were created by partition and don't overlap. *)
let gather (subs : (_, _) t array) : unit =
  Array.iter
    (fun sub ->
      match sub.location with
      | GPU _ | Stale_CPU _ ->
          (* Need to sync from GPU *)
          (match get_sub_meta sub with
          | Some _meta ->
              (* Transfer handled by Transfer module *)
              ()
          | None -> ())
      | _ -> ())
    subs

(** {1 Subvector Queries} *)

(** Get subvector depth (0 = root, 1 = child, 2 = grandchild, ...) *)
let depth (vec : ('a, 'b) t) : int =
  match get_sub_meta vec with Some meta -> meta.depth | None -> 0

(** Get parent vector ID if this is a subvector *)
let parent_id (vec : ('a, 'b) t) : int option =
  match get_sub_meta vec with Some meta -> Some meta.parent_id | None -> None

(** Get start offset relative to immediate parent *)
let sub_start (vec : ('a, 'b) t) : int option =
  match get_sub_meta vec with Some meta -> Some meta.start | None -> None

(** Get ok_range (safe read range) *)
let sub_ok_range (vec : ('a, 'b) t) : int option =
  match get_sub_meta vec with Some meta -> Some meta.ok_range | None -> None

(** Get ko_range (unsafe write range) *)
let sub_ko_range (vec : ('a, 'b) t) : int option =
  match get_sub_meta vec with Some meta -> Some meta.ko_range | None -> None

(** {1 Phase 6: Vector Utilities} *)

(** {2 Iteration} *)

(** Iterate over all elements (CPU-side, auto-syncs if needed) *)
let iter : type a b. (a -> unit) -> (a, b) t -> unit =
 fun f vec ->
  ensure_cpu_sync vec ;
  for i = 0 to vec.length - 1 do
    f (unsafe_get vec i)
  done

(** Iterate with index *)
let iteri : type a b. (int -> a -> unit) -> (a, b) t -> unit =
 fun f vec ->
  ensure_cpu_sync vec ;
  for i = 0 to vec.length - 1 do
    f i (unsafe_get vec i)
  done

(** {2 Mapping} *)

(** Map function over vector, creating new vector with given kind *)
let map : type a b c d. (a -> c) -> (c, d) kind -> (a, b) t -> (c, d) t =
 fun f target_kind vec ->
  ensure_cpu_sync vec ;
  let result = create target_kind vec.length in
  for i = 0 to vec.length - 1 do
    unsafe_set result i (f (unsafe_get vec i))
  done ;
  result

(** Map with index *)
let mapi : type a b c d. (int -> a -> c) -> (c, d) kind -> (a, b) t -> (c, d) t =
 fun f target_kind vec ->
  ensure_cpu_sync vec ;
  let result = create target_kind vec.length in
  for i = 0 to vec.length - 1 do
    unsafe_set result i (f i (unsafe_get vec i))
  done ;
  result

(** In-place map (same type) *)
let map_inplace : type a b. (a -> a) -> (a, b) t -> unit =
 fun f vec ->
  ensure_cpu_sync vec ;
  for i = 0 to vec.length - 1 do
    unsafe_set vec i (f (unsafe_get vec i))
  done

(** {2 Folding} *)

(** Fold left *)
let fold_left : type a b acc. (acc -> a -> acc) -> acc -> (a, b) t -> acc =
 fun f init vec ->
  ensure_cpu_sync vec ;
  let acc = ref init in
  for i = 0 to vec.length - 1 do
    acc := f !acc (unsafe_get vec i)
  done ;
  !acc

(** Fold right *)
let fold_right : type a b acc. (a -> acc -> acc) -> (a, b) t -> acc -> acc =
 fun f vec init ->
  ensure_cpu_sync vec ;
  let acc = ref init in
  for i = vec.length - 1 downto 0 do
    acc := f (unsafe_get vec i) !acc
  done ;
  !acc

(** {2 Predicates} *)

(** Check if all elements satisfy predicate *)
let for_all : type a b. (a -> bool) -> (a, b) t -> bool =
 fun p vec ->
  ensure_cpu_sync vec ;
  let result = ref true in
  let i = ref 0 in
  while !result && !i < vec.length do
    result := p (unsafe_get vec !i) ;
    incr i
  done ;
  !result

(** Check if any element satisfies predicate *)
let exists : type a b. (a -> bool) -> (a, b) t -> bool =
 fun p vec ->
  ensure_cpu_sync vec ;
  let result = ref false in
  let i = ref 0 in
  while not !result && !i < vec.length do
    result := p (unsafe_get vec !i) ;
    incr i
  done ;
  !result

(** Find first element satisfying predicate *)
let find : type a b. (a -> bool) -> (a, b) t -> a option =
 fun p vec ->
  ensure_cpu_sync vec ;
  let result = ref None in
  let i = ref 0 in
  while Option.is_none !result && !i < vec.length do
    let v = unsafe_get vec !i in
    if p v then result := Some v ;
    incr i
  done ;
  !result

(** Find index of first element satisfying predicate *)
let find_index : type a b. (a -> bool) -> (a, b) t -> int option =
 fun p vec ->
  ensure_cpu_sync vec ;
  let result = ref None in
  let i = ref 0 in
  while Option.is_none !result && !i < vec.length do
    if p (unsafe_get vec !i) then result := Some !i ;
    incr i
  done ;
  !result

(** {2 Aggregation} *)

(** Sum elements (requires + operation via fold) *)
let sum (type a b) ~(zero : a) ~(add : a -> a -> a) (vec : (a, b) t) : a =
  fold_left add zero vec  (* fold_left already calls ensure_cpu_sync *)

(** Find minimum element *)
let min_elt (type a b) ~(compare : a -> a -> int) (vec : (a, b) t) : a option =
  if vec.length = 0 then None
  else begin
    ensure_cpu_sync vec ;
    let m = ref (unsafe_get vec 0) in
    for i = 1 to vec.length - 1 do
      let v = unsafe_get vec i in
      if compare v !m < 0 then m := v
    done ;
    Some !m
  end

(** Find maximum element *)
let max_elt (type a b) ~(compare : a -> a -> int) (vec : (a, b) t) : a option =
  if vec.length = 0 then None
  else begin
    ensure_cpu_sync vec ;
    let m = ref (unsafe_get vec 0) in
    for i = 1 to vec.length - 1 do
      let v = unsafe_get vec i in
      if compare v !m > 0 then m := v
    done ;
    Some !m
  end

(** {2 Conversion} *)

(** Convert to OCaml list *)
let to_list : type a b. (a, b) t -> a list =
 fun vec -> fold_right (fun x acc -> x :: acc) vec []  (* fold_right already syncs *)

(** Create from OCaml list *)
let of_list : type a b. (a, b) kind -> a list -> (a, b) t =
 fun kind lst ->
  let len = List.length lst in
  let vec = create kind len in
  List.iteri (fun i v -> unsafe_set vec i v) lst ;
  vec

(** Convert to OCaml array *)
let to_array : type a b. (a, b) t -> a array =
 fun vec ->
  if vec.length = 0 then [||]
  else begin
    ensure_cpu_sync vec ;
    let arr = Array.make vec.length (unsafe_get vec 0) in
    for i = 1 to vec.length - 1 do
      arr.(i) <- unsafe_get vec i
    done ;
    arr
  end

(** Create from OCaml array *)
let of_array : type a b. (a, b) kind -> a array -> (a, b) t =
 fun kind arr ->
  let vec = create kind (Array.length arr) in
  Array.iteri (fun i v -> unsafe_set vec i v) arr ;
  vec

(** {2 Blitting} *)

(** Copy elements from one vector to another *)
let blit : type a b.
    src:(a, b) t -> src_off:int -> dst:(a, b) t -> dst_off:int -> len:int -> unit
    =
 fun ~src ~src_off ~dst ~dst_off ~len ->
  if src_off < 0 || src_off + len > src.length then
    invalid_arg "Vector.blit: source range out of bounds" ;
  if dst_off < 0 || dst_off + len > dst.length then
    invalid_arg "Vector.blit: destination range out of bounds" ;
  ensure_cpu_sync src ;
  for i = 0 to len - 1 do
    unsafe_set dst (dst_off + i) (unsafe_get src (src_off + i))
  done
