(******************************************************************************
 * Sarek Runtime - High-Level Vector Abstraction
 *
 * Provides a unified Vector type that tracks data location (CPU/GPU) and
 * supports automatic synchronization. Replaces the old SPOC Vector module.
 *
 * Design principles:
 * - Type-safe: Uses GADTs and existentials instead of Obj.t
 * - Location-aware: Tracks where data resides (CPU, GPU, or both)
 * - Lazy allocation: GPU memory allocated on first device use
 * - Auto-sync: Element access triggers sync when needed
 ******************************************************************************)

(** {1 Element Types (GADT)} *)

(** Element kind with type witness - replaces Bigarray.kind wrapper *)
type (_, _) kind =
  | Float32 : (float, Bigarray.float32_elt) kind
  | Float64 : (float, Bigarray.float64_elt) kind
  | Int32 : (int32, Bigarray.int32_elt) kind
  | Int64 : (int64, Bigarray.int64_elt) kind
  | Char : (char, Bigarray.int8_unsigned_elt) kind
  | Complex32 : (Complex.t, Bigarray.complex32_elt) kind

(** Convert our kind to Bigarray.kind *)
let to_bigarray_kind : type a b. (a, b) kind -> (a, b) Bigarray.kind = function
  | Float32 -> Bigarray.Float32
  | Float64 -> Bigarray.Float64
  | Int32 -> Bigarray.Int32
  | Int64 -> Bigarray.Int64
  | Char -> Bigarray.Char
  | Complex32 -> Bigarray.Complex32

(** Element size in bytes *)
let elem_size : type a b. (a, b) kind -> int = function
  | Float32 -> 4
  | Float64 -> 8
  | Int32 -> 4
  | Int64 -> 8
  | Char -> 1
  | Complex32 -> 8

(** Kind name for debugging *)
let kind_name : type a b. (a, b) kind -> string = function
  | Float32 -> "Float32"
  | Float64 -> "Float64"
  | Int32 -> "Int32"
  | Int64 -> "Int64"
  | Char -> "Char"
  | Complex32 -> "Complex32"

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
    Uses first-class modules to avoid Obj.t. *)
module type DEVICE_BUFFER = sig
  type t
  val device : Device.t
  val size : int
  val elem_size : int
  val ptr : nativeint  (** Raw device pointer for kernel args *)
end

type device_buffer = (module DEVICE_BUFFER)

(** Device buffer storage - maps device ID to buffer *)
type device_buffers = (int, device_buffer) Hashtbl.t

(** {1 Vector Type} *)

(** High-level vector with location tracking *)
type ('a, 'b) t = {
  mutable cpu_data : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t;
  device_buffers : device_buffers;
  length : int;
  kind : ('a, 'b) kind;
  mutable location : location;
  mutable auto_sync : bool;  (** Enable automatic CPU sync on get *)
  id : int;  (** Unique vector ID for debugging *)
}

(** Global vector ID counter *)
let next_id = ref 0

(** {1 Creation} *)

(** Create a new vector on CPU *)
let create (kind : ('a, 'b) kind) ?(dev : Device.t option) (length : int) :
    ('a, 'b) t =
  incr next_id ;
  let ba_kind = to_bigarray_kind kind in
  let cpu_data = Bigarray.Array1.create ba_kind Bigarray.c_layout length in
  let vec =
    {
      cpu_data;
      device_buffers = Hashtbl.create 4;
      length;
      kind;
      location = CPU;
      auto_sync = true;
      id = !next_id;
    }
  in
  (* If device specified, mark as target but don't allocate yet (lazy) *)
  (match dev with
  | Some d -> vec.location <- Stale_GPU d  (* CPU authoritative, GPU will need update *)
  | None -> ()) ;
  vec

(** Create from existing Bigarray (shares memory) *)
let of_bigarray (kind : ('a, 'b) kind)
    (ba : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t) : ('a, 'b) t =
  incr next_id ;
  {
    cpu_data = ba;
    device_buffers = Hashtbl.create 4;
    length = Bigarray.Array1.dim ba;
    kind;
    location = CPU;
    auto_sync = true;
    id = !next_id;
  }

(** Get underlying Bigarray (syncs to CPU if needed) *)
let to_bigarray (vec : ('a, 'b) t) :
    ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t =
  (* TODO: sync_to_cpu if location is GPU-only *)
  vec.cpu_data

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

(** {1 Element Access} *)

(** Get element (syncs to CPU if needed and auto_sync enabled) *)
let get (vec : ('a, 'b) t) (idx : int) : 'a =
  if idx < 0 || idx >= vec.length then
    invalid_arg
      (Printf.sprintf "Vector.get: index %d out of bounds [0, %d)" idx
         vec.length) ;
  (* TODO: If auto_sync and data is GPU-only, sync first *)
  Bigarray.Array1.get vec.cpu_data idx

(** Set element (marks GPU as stale if synced) *)
let set (vec : ('a, 'b) t) (idx : int) (value : 'a) : unit =
  if idx < 0 || idx >= vec.length then
    invalid_arg
      (Printf.sprintf "Vector.set: index %d out of bounds [0, %d)" idx
         vec.length) ;
  Bigarray.Array1.set vec.cpu_data idx value ;
  (* Mark GPU as stale if we had synced data *)
  match vec.location with
  | Both d -> vec.location <- Stale_GPU d
  | GPU d ->
      (* Data was GPU-only, now we have CPU copy that's authoritative *)
      vec.location <- Stale_GPU d
  | CPU | Stale_CPU _ | Stale_GPU _ -> ()

(** Indexing operators *)
let ( .%[] ) = get

let ( .%[]<- ) = set

(** {1 Unsafe Access (no bounds check)} *)

let unsafe_get (vec : ('a, 'b) t) (idx : int) : 'a =
  Bigarray.Array1.unsafe_get vec.cpu_data idx

let unsafe_set (vec : ('a, 'b) t) (idx : int) (value : 'a) : unit =
  Bigarray.Array1.unsafe_set vec.cpu_data idx value ;
  match vec.location with
  | Both d -> vec.location <- Stale_GPU d
  | GPU d -> vec.location <- Stale_GPU d
  | CPU | Stale_CPU _ | Stale_GPU _ -> ()

(** {1 Auto-sync Control} *)

let set_auto_sync (vec : ('a, 'b) t) (enabled : bool) : unit =
  vec.auto_sync <- enabled

let auto_sync (vec : ('a, 'b) t) : bool = vec.auto_sync

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
  match vec.location with
  | Both _ -> true
  | CPU | GPU _ | Stale_CPU _ | Stale_GPU _ -> false

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
  Printf.sprintf "Vector#%d<%s>[%d] @ %s" vec.id (kind_name vec.kind)
    vec.length
    (location_to_string vec.location)

(** {1 Convenience Constructors} *)

let create_float32 ?dev n = create Float32 ?dev n

let create_float64 ?dev n = create Float64 ?dev n

let create_int32 ?dev n = create Int32 ?dev n

let create_int64 ?dev n = create Int64 ?dev n

(** {1 Initialization Helpers} *)

(** Fill vector with a value *)
let fill (vec : ('a, 'b) t) (value : 'a) : unit =
  Bigarray.Array1.fill vec.cpu_data value ;
  match vec.location with
  | Both d -> vec.location <- Stale_GPU d
  | GPU d -> vec.location <- Stale_GPU d
  | CPU | Stale_CPU _ | Stale_GPU _ -> ()

(** Initialize with a function *)
let init (kind : ('a, 'b) kind) (length : int) (f : int -> 'a) : ('a, 'b) t =
  let vec = create kind length in
  for i = 0 to length - 1 do
    Bigarray.Array1.unsafe_set vec.cpu_data i (f i)
  done ;
  vec

(** Copy vector *)
let copy (vec : ('a, 'b) t) : ('a, 'b) t =
  incr next_id ;
  let cpu_data =
    Bigarray.Array1.create
      (to_bigarray_kind vec.kind)
      Bigarray.c_layout vec.length
  in
  Bigarray.Array1.blit vec.cpu_data cpu_data ;
  {
    cpu_data;
    device_buffers = Hashtbl.create 4;  (* Don't copy GPU buffers *)
    length = vec.length;
    kind = vec.kind;
    location = CPU;  (* Copy is on CPU only *)
    auto_sync = vec.auto_sync;
    id = !next_id;
  }
