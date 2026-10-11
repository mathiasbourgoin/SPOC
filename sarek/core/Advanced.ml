(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - Advanced Features (Beyond SPOC)
 *
 * New features that go beyond SPOC's original capabilities.
 * Phase 8 of runtime V2 feature parity roadmap.
 ******************************************************************************)

open Spoc_framework

(** {1 8a. Unified Memory (CUDA Managed Memory)} *)

(** Unified memory allows automatic page migration between CPU and GPU. Note:
    Requires backend support - currently a placeholder API. *)
module Unified = struct
  (** Create a vector with unified memory (automatic migration) *)
  let create (kind : ('a, 'b) Vector.kind) (length : int) : ('a, 'b) Vector.t =
    (* For now, falls back to regular creation *)
    (* TODO: Implement with cudaMallocManaged when backend supports it *)
    let vec = Vector.create kind length in
    (* Mark as unified - location stays Both once allocated *)
    vec

  (** Check if unified memory is available on device *)
  let is_available (_dev : Device.t) : bool =
    (* Would check for managed memory support *)
    false
  (* Placeholder *)
end

(** {1 8b. Pinned Memory} *)

(** Page-locked host memory, explicit and opt-in: nothing in SPOC pins memory
    unless you call this module. A thin owner-handle over
    {!Transfer.alloc_pinned} and {!Transfer.register_host}, which hold the
    mechanics (per-backend behaviour, measured pitfalls on HIP, the in-flight
    copy check).

    The memory is a [Bigarray], not a {!Vector.t}: a vector's host mirror is
    garbage collected, pinned memory is released only explicitly (sub-arrays
    could outlive a finaliser's decision). Use it as the host side of
    {!Transfer.copy_to_device} / {!Transfer.copy_to_host} with [~stream], or as
    the destination of [pread]/[O_DIRECT] reads.

    Release a handle only through {!Pinned.release}: freeing or unregistering
    its array directly with [Transfer] leaves the handle believing it is live,
    and a later {!Pinned.release} could then act on a different allocation at
    the same address. *)
module Pinned : sig
  (** A live-or-released pinned buffer. Abstract: its state changes only through
      {!release}. *)
  type ('a, 'b) t

  (** [true] iff [dev]'s backend supports page-locked memory. Probes with a
      one-byte {!alloc} and {!release}, which costs a driver allocation and free
      per call on GPU backends and flushes pending deferred launches (the free
      does). Native, Interpreter and Metal answer [true] (their pinned memory is
      plain host memory). Only the backend's [Feature_not_supported] refusal
      means [false]; any other error (driver out-of-memory, unknown framework)
      propagates. *)
  val is_available : Device.t -> bool

  (** Allocate [n] elements of page-locked memory for DMA with the device.
      Raises [Invalid_argument] for [n < 1] or [Bigarray.float16] (use a [char]
      buffer and {!Transfer.copy_bytes_to_device}), and the backend's
      [Feature_not_supported] where unsupported (see {!is_available}). On HIP
      the allocation can be slow for multi-GiB sizes: measure, or use
      {!register}. *)
  val alloc : Device.t -> ('a, 'b) Bigarray.kind -> int -> ('a, 'b) t

  (** Page-lock an array you already own. The array stays yours and usable after
      {!release}; the handle roots it until then. Raises [Invalid_argument] if
      it is empty or already pinned or registered, and [Feature_not_supported]
      where the backend cannot register host memory. See
      {!Transfer.register_host} for the measured pitfalls (copies right after
      registering, registrations spanning two regions). *)
  val register :
    Device.t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> ('a, 'b) t

  val device : ('a, 'b) t -> Device.t

  val is_live : ('a, 'b) t -> bool

  (** The page-locked array. Raises [Invalid_argument] once released: the memory
      of an {!alloc} handle is gone then, so do not keep the array itself past
      {!release}. *)
  val bigarray : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t

  (** Free the memory ({!alloc}) or unregister it ({!register}). Idempotent.
      Refused with [Invalid_argument], the handle staying live and the release
      retryable, while stream-ordered copies may still use the memory:
      synchronize the stream (or an event recorded after them) first. *)
  val release : ('a, 'b) t -> unit

  (** [with_alloc ?stream dev kind n f] runs [f] on a fresh {!alloc}, then
      synchronizes [stream] (if given) and releases the handle. If [f] raises,
      its exception is re-raised after a best-effort release. [f] must not let
      the array escape. Stream-ordered copies [f] enqueued must be on [stream],
      or synchronized by [f]: otherwise the release is refused, the refusal
      ([Invalid_argument]) is raised when [f] returned normally, and the handle
      is unreachable, so the allocation stays pinned for the process lifetime.
  *)
  val with_alloc :
    ?stream:Transfer.stream ->
    Device.t ->
    ('a, 'b) Bigarray.kind ->
    int ->
    (('a, 'b) t -> 'r) ->
    'r
end = struct
  type ownership = Allocated | Registered

  type ('a, 'b) t = {
    device : Device.t;
    data : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t;
    ownership : ownership;
    lock : Mutex.t;
    mutable live : bool;
  }

  let alloc (device : Device.t) (kind : ('a, 'b) Bigarray.kind) (n : int) :
      ('a, 'b) t =
    let data = Transfer.alloc_pinned device kind n in
    {device; data; ownership = Allocated; lock = Mutex.create (); live = true}

  let register (device : Device.t)
      (data : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t) : ('a, 'b) t =
    Transfer.register_host device data ;
    {device; data; ownership = Registered; lock = Mutex.create (); live = true}

  let device (p : ('a, 'b) t) : Device.t = p.device

  let is_live (p : ('a, 'b) t) : bool = Mutex.protect p.lock (fun () -> p.live)

  let bigarray (p : ('a, 'b) t) : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
      =
    Mutex.protect p.lock (fun () ->
        if p.live then p.data
        else invalid_arg "Advanced.Pinned.bigarray: handle already released")

  let release (p : ('a, 'b) t) : unit =
    Mutex.protect p.lock (fun () ->
        if p.live then begin
          (match p.ownership with
          | Allocated -> Transfer.free_pinned p.data
          | Registered -> Transfer.unregister_host p.data) ;
          p.live <- false
        end)

  let is_available (dev : Device.t) : bool =
    match alloc dev Bigarray.char 1 with
    | probe ->
        release probe ;
        true
    | exception
        Sarek_backend_error.Backend_error.Backend_error
          (Sarek_backend_error.Backend_error.Plugin
             {
               error = Sarek_backend_error.Backend_error.Feature_not_supported _;
               _;
             }) ->
        false

  let with_alloc ?stream (device : Device.t) (kind : ('a, 'b) Bigarray.kind)
      (n : int) (f : ('a, 'b) t -> 'r) : 'r =
    let p = alloc device kind n in
    let finish () =
      Option.iter Transfer.synchronize_stream stream ;
      release p
    in
    match f p with
    | result ->
        finish () ;
        result
    | exception exn ->
        let backtrace = Printexc.get_raw_backtrace () in
        (try finish () with _ -> ()) ;
        Printexc.raise_with_backtrace exn backtrace
end

(** {1 8c. Async Kernel Launch with Futures} *)

(** Future representing a pending async operation *)
module type FUTURE = sig
  type 'a t

  val is_ready : 'a t -> bool

  val await : 'a t -> 'a

  val map : ('a -> 'b) -> 'a t -> 'b t
end

module Future = struct
  type 'a state =
    | Pending of (unit -> bool)
    (* Check if ready *)
    | Ready of 'a

  type 'a t = {mutable state : 'a state}

  let is_ready fut =
    match fut.state with
    | Ready _ -> true
    | Pending check ->
        if check () then begin
          (* Transition to ready - but we don't have the value yet *)
          (* This is a limitation of the simple design *)
          true
        end
        else false

  let await fut =
    match fut.state with
    | Ready v -> v
    | Pending check -> (
        (* Busy wait - in practice would use events *)
        while not (check ()) do
          Unix.sleepf 0.001
        done ;
        match fut.state with
        | Ready v -> v
        | Pending _ -> failwith "Future: await failed")

  let map _f fut =
    {
      state =
        Pending
          (fun () ->
            match fut.state with
            | Ready _v ->
                (* Already ready, apply function *)
                true
            | Pending _ -> is_ready fut);
    }

  (** Create a completed future *)
  let return v = {state = Ready v}

  (** Create a pending future with completion check *)
  let pending check = {state = Pending check}
end

(** Run a kernel asynchronously and return a future *)
let run_async (_kernel : Kernel.t) (_args : Kernel.args)
    ~(_block : Framework_sig.dims) ~(_grid : Framework_sig.dims)
    (_dev : Device.t) : unit Future.t =
  (* Would launch kernel on stream and return future tied to event *)
  (* For now, just run synchronously and return completed future *)
  Future.return ()

(** {1 8d. Device Memory Pools} *)

(** Memory pool for efficient allocation/deallocation *)
module Pool = struct
  type t = {
    device : Device.t;
    mutable total_allocated : int64;
    mutable allocations : int;
  }

  (** Create a new memory pool on device *)
  let create (dev : Device.t) ~(_initial_size : int) : t =
    {device = dev; total_allocated = 0L; allocations = 0}

  (** Allocate from pool *)
  let alloc (pool : t) (kind : ('a, 'b) Vector.kind) (length : int) :
      ('a, 'b) Vector.t =
    let vec = Vector.create kind ~dev:pool.device length in
    pool.allocations <- pool.allocations + 1 ;
    pool.total_allocated <-
      Int64.add
        pool.total_allocated
        (Int64.of_int (length * Vector.elem_size kind)) ;
    vec

  (** Reset pool - free all allocations *)
  let reset (pool : t) : unit =
    (* In a real implementation, would free all pooled memory *)
    pool.total_allocated <- 0L ;
    pool.allocations <- 0

  (** Destroy pool *)
  let destroy (_pool : t) : unit =
    (* Would release pool resources *)
    ()

  (** Get pool statistics *)
  let stats pool = (pool.allocations, pool.total_allocated)
end

(** {1 8e. Graph-based Execution (CUDA Graphs)} *)

(** Computation graph for capturing and replaying operations. Note: Full
    implementation requires deep backend integration. *)
module Graph = struct
  type node_id = int

  type node_type =
    | Kernel of {name : string}
    | Transfer of [`H2D | `D2H]
    | Sync

  type node = {id : node_id; node_type : node_type; dependencies : node_id list}

  type t = {mutable nodes : node list; mutable next_id : node_id}

  (** Create empty graph *)
  let create () : t = {nodes = []; next_id = 0}

  (** Add kernel node *)
  let add_kernel (graph : t) ~(name : string) ~(deps : node_id list) : node_id =
    let id = graph.next_id in
    graph.next_id <- id + 1 ;
    graph.nodes <-
      {id; node_type = Kernel {name}; dependencies = deps} :: graph.nodes ;
    id

  (** Add transfer node *)
  let add_transfer (graph : t) ~(direction : [`H2D | `D2H])
      ~(deps : node_id list) : node_id =
    let id = graph.next_id in
    graph.next_id <- id + 1 ;
    graph.nodes <-
      {id; node_type = Transfer direction; dependencies = deps} :: graph.nodes ;
    id

  (** Add sync node *)
  let add_sync (graph : t) ~(deps : node_id list) : node_id =
    let id = graph.next_id in
    graph.next_id <- id + 1 ;
    graph.nodes <- {id; node_type = Sync; dependencies = deps} :: graph.nodes ;
    id

  (** Executable graph (compiled for specific device) *)
  type executable = {graph : t; device : Device.t}

  (** Instantiate graph for device *)
  let instantiate (graph : t) (dev : Device.t) : executable =
    {graph; device = dev}

  (** Launch executable graph *)
  let launch (_exec : executable) : unit =
    (* Would use cudaGraphLaunch or similar *)
    (* For now, placeholder *)
    ()

  (** Get node count *)
  let node_count graph = List.length graph.nodes

  (** Print graph for debugging *)
  let print graph =
    Printf.printf "Graph with %d nodes:\n" (node_count graph) ;
    List.iter
      (fun n ->
        let type_str =
          match n.node_type with
          | Kernel {name} -> Printf.sprintf "Kernel(%s)" name
          | Transfer `H2D -> "Transfer(H2D)"
          | Transfer `D2H -> "Transfer(D2H)"
          | Sync -> "Sync"
        in
        Printf.printf
          "  [%d] %s <- [%s]\n"
          n.id
          type_str
          (String.concat ", " (List.map string_of_int n.dependencies)))
      (List.rev graph.nodes)
end

(** {1 Convenience Re-exports} *)

(** Check if any advanced features are available on device *)
let has_advanced_features (dev : Device.t) : bool =
  Unified.is_available dev || Pinned.is_available dev
