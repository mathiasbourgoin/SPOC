(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - Transfer Control & Streams
 *
 * Provides explicit and async data transfer API with stream management.
 * Phase 2 of runtime V2 feature parity roadmap.
 *
 * Uses the framework plugin system directly.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** {1 Auto-Transfer Mode} *)

let auto_mode = ref true

let enable_auto () = auto_mode := true

let disable_auto () = auto_mode := false

let is_auto () = !auto_mode

let set_auto enabled = auto_mode := enabled

(** {1 Device Buffer Allocation} *)

(** Allocate a device buffer for a scalar vector, returning a DEVICE_BUFFER
    module. The buffer is packaged with its backend for type-safe operations.

    For CPU devices (OpenCL CPU, Native), uses zero-copy allocation when
    possible to avoid memory transfers entirely. *)
let alloc_scalar_buffer (type a b) (dev : Device.t) (length : int)
    (sk : (a, b) Vector.scalar_kind) : Vector.device_buffer =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get dev.backend_id in
      let ba_kind = Vector.to_bigarray_kind sk in
      let buf = B.Memory.alloc backend_dev length ba_kind in
      let elem_sz = Vector.scalar_elem_size sk in
      let device_ptr = B.Memory.device_ptr buf in
      (module struct
        let device = dev

        let size = length

        let elem_size = elem_sz

        let device_ptr = device_ptr

        let bind_to_kargs kargs idx =
          (* Unwrap kargs to the backend's kernel args type and bind buffer *)
          Log.debugf
            Log.Transfer
            "bind_to_kargs: idx=%d ptr=%Ld"
            idx
            (Int64.of_nativeint device_ptr) ;
          match B.unwrap_kargs kargs with
          | Some args -> B.Kernel.set_arg_buffer args idx buf
          | None -> failwith "bind_to_kargs: backend mismatch"

        let host_ptr_to_device src_ptr ~byte_size =
          Log.debugf
            Log.Transfer
            "host_ptr_to_device: ptr=%Ld size=%d"
            (Int64.of_nativeint device_ptr)
            byte_size ;
          if B.Memory.is_zero_copy buf then () (* Skip for zero-copy *)
          else
            B.Memory.host_ptr_to_device
              ~src_ptr:(Ctypes.raw_address_of_ptr src_ptr)
              ~byte_size
              ~dst:buf

        let device_to_host_ptr dst_ptr ~byte_size =
          if B.Memory.is_zero_copy buf then () (* Skip for zero-copy *)
          else
            B.Memory.device_to_host_ptr
              ~src:buf
              ~dst_ptr:(Ctypes.raw_address_of_ptr dst_ptr)
              ~byte_size

        let range_from_host ~src_ptr ~dst_offset ~byte_size ~stream =
          B.Async.host_to_device
            ~src_ptr
            ~dst:buf
            ~dst_offset
            ~byte_size
            ~stream:
              (Memory.unwrap_stream_for ~backend:B.name B.unwrap_stream stream)

        let range_to_host ~src_offset ~dst_ptr ~byte_size ~stream =
          B.Async.device_to_host
            ~src:buf
            ~src_offset
            ~dst_ptr
            ~byte_size
            ~stream:
              (Memory.unwrap_stream_for ~backend:B.name B.unwrap_stream stream)

        let free () = B.Memory.free buf
      end : Vector.DEVICE_BUFFER)

(** Allocate a device buffer using zero-copy (host memory sharing) if supported.
    Returns None if the backend doesn't support zero-copy for this device. *)
let alloc_scalar_buffer_zero_copy (type a b) (dev : Device.t)
    (ba : (a, b, Bigarray.c_layout) Bigarray.Array1.t)
    (sk : (a, b) Vector.scalar_kind) : Vector.device_buffer option =
  match Framework_registry.find_backend dev.framework with
  | None -> None
  | Some (module B : Framework_sig.BACKEND) -> (
      let backend_dev = B.Device.get dev.backend_id in
      let ba_kind = Vector.to_bigarray_kind sk in
      match B.Memory.alloc_zero_copy backend_dev ba ba_kind with
      | None -> None
      | Some buf ->
          let elem_sz = Vector.scalar_elem_size sk in
          let length = Bigarray.Array1.dim ba in
          let device_ptr = B.Memory.device_ptr buf in
          Some
            (module struct
              let device = dev

              let size = length

              let elem_size = elem_sz

              let device_ptr = device_ptr

              let bind_to_kargs kargs idx =
                Log.debugf
                  Log.Transfer
                  "bind_to_kargs (zero-copy): idx=%d ptr=%Ld"
                  idx
                  (Int64.of_nativeint device_ptr) ;
                match B.unwrap_kargs kargs with
                | Some args -> B.Kernel.set_arg_buffer args idx buf
                | None -> failwith "bind_to_kargs: backend mismatch"

              let host_ptr_to_device _src_ptr ~byte_size:_ =
                () (* No-op for zero-copy *)

              let device_to_host_ptr _dst_ptr ~byte_size:_ =
                () (* No-op for zero-copy *)

              let range_from_host ~src_ptr ~dst_offset ~byte_size ~stream =
                B.Async.host_to_device
                  ~src_ptr
                  ~dst:buf
                  ~dst_offset
                  ~byte_size
                  ~stream:
                    (Memory.unwrap_stream_for
                       ~backend:B.name
                       B.unwrap_stream
                       stream)

              let range_to_host ~src_offset ~dst_ptr ~byte_size ~stream =
                B.Async.device_to_host
                  ~src:buf
                  ~src_offset
                  ~dst_ptr
                  ~byte_size
                  ~stream:
                    (Memory.unwrap_stream_for
                       ~backend:B.name
                       B.unwrap_stream
                       stream)

              let free () = B.Memory.free buf
            end : Vector.DEVICE_BUFFER))

(** Allocate a device buffer for a custom vector *)
let alloc_custom_buffer (dev : Device.t) (length : int) (elem_sz : int) :
    Vector.device_buffer =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get dev.backend_id in
      let buf =
        B.Memory.alloc_custom backend_dev ~size:length ~elem_size:elem_sz
      in
      let device_ptr = B.Memory.device_ptr buf in
      (module struct
        let device = dev

        let size = length

        let elem_size = elem_sz

        let device_ptr = device_ptr

        let bind_to_kargs kargs idx =
          (* Unwrap kargs to the backend's kernel args type and bind buffer *)
          Log.debugf
            Log.Transfer
            "bind_to_kargs: idx=%d ptr=%Ld"
            idx
            (Int64.of_nativeint device_ptr) ;
          match B.unwrap_kargs kargs with
          | Some args -> B.Kernel.set_arg_buffer args idx buf
          | None -> failwith "bind_to_kargs: backend mismatch"

        let host_ptr_to_device src_ptr ~byte_size =
          B.Memory.host_ptr_to_device
            ~src_ptr:(Ctypes.raw_address_of_ptr src_ptr)
            ~byte_size
            ~dst:buf

        let device_to_host_ptr dst_ptr ~byte_size =
          B.Memory.device_to_host_ptr
            ~src:buf
            ~dst_ptr:(Ctypes.raw_address_of_ptr dst_ptr)
            ~byte_size

        let range_from_host ~src_ptr ~dst_offset ~byte_size ~stream =
          B.Async.host_to_device
            ~src_ptr
            ~dst:buf
            ~dst_offset
            ~byte_size
            ~stream:
              (Memory.unwrap_stream_for ~backend:B.name B.unwrap_stream stream)

        let range_to_host ~src_offset ~dst_ptr ~byte_size ~stream =
          B.Async.device_to_host
            ~src:buf
            ~src_offset
            ~dst_ptr
            ~byte_size
            ~stream:
              (Memory.unwrap_stream_for ~backend:B.name B.unwrap_stream stream)

        let free () = B.Memory.free buf
      end : Vector.DEVICE_BUFFER)

(** {1 Buffer Management for Vectors} *)

(** Ensure vector has a device buffer, allocating if needed. For backends that
    support zero-copy (typically CPU backends), automatically uses zero-copy to
    avoid memory transfer overhead. The backend decides via alloc_zero_copy. *)
let ensure_buffer (type a b) (vec : (a, b) Vector.t) (dev : Device.t) :
    Vector.device_buffer =
  match Vector.get_buffer vec dev with
  | Some buf -> buf
  | None ->
      Log.debugf
        Log.Transfer
        "ensure_buffer: allocating for dev=%d len=%d"
        dev.id
        vec.length ;
      let buf =
        Gpu_memory.with_retry (fun () ->
            match (vec.kind, vec.host) with
            | Vector.Scalar sk, Vector.Bigarray_storage ba -> (
                (* Try zero-copy first - backend decides if supported *)
                Log.debug Log.Transfer "  -> trying zero-copy path" ;
                match alloc_scalar_buffer_zero_copy dev ba sk with
                | Some zc_buf ->
                    Log.debugf
                      Log.Transfer
                      "  -> using zero-copy for device %d"
                      dev.id ;
                    zc_buf
                | None ->
                    Log.debug
                      Log.Transfer
                      "  -> zero-copy not supported, using regular alloc" ;
                    let buf = alloc_scalar_buffer dev vec.length sk in
                    buf)
            | Vector.Scalar _, Vector.Custom_storage _ -> .
            | Vector.Scalar sk, Vector.Device_only_storage ->
                (* A device-only vector has no host storage to share, so never
                   the zero-copy path: real device memory, nothing mirrored. *)
                alloc_scalar_buffer dev vec.length sk
            | Vector.Custom c, _ ->
                Log.debug Log.Transfer "  -> custom alloc" ;
                alloc_custom_buffer dev vec.length c.elem_size)
      in
      Log.debugf
        Log.Transfer
        "ensure_buffer: storing buffer for dev=%d (hashtbl key=%d)"
        dev.id
        dev.id ;
      (* Register GC finalizer on first device buffer allocation *)
      if Hashtbl.length vec.device_buffers = 0 then
        Gpu_memory.register_finalizer vec ;
      Hashtbl.replace vec.device_buffers dev.id buf ;
      let (module B : Vector.DEVICE_BUFFER) = buf in
      (* Track GPU memory usage *)
      Gpu_memory.track_alloc (B.size * B.elem_size) ;
      Log.debugf
        Log.Transfer
        "ensure_buffer: stored buffer ptr=%Ld size=%d"
        (Int64.of_nativeint B.device_ptr)
        B.size ;
      buf

(** {1 Transfer Operations} *)

(** Everything that reads device memory back into host storage, and NOTHING
    else. The module exists for its signature: [copy_device_to_host] is defined
    inside and deliberately absent from it, so the only code that can call the
    packed-buffer read directly is {!Read_back.read_back_to_host} below.

    That is the enforcement the previous version of this claim lacked. It said
    the call sites were "exhaustive BY CONSTRUCTION" while nothing constructed
    anything: there is no [Transfer.mli], so every top-level binding in this
    file is exported and a fifth direct caller compiled fine. The claim happened
    to be true when it was written — it was re-audited, and it held — but an
    audited fact and an enforced one differ exactly where it matters, which is
    the next person to add a read-back path. Three of the four existing call
    sites were added or corrected in this branch precisely because they had
    bypassed the decision, so "someone adds a fifth and bypasses it too" is the
    observed failure mode, not a hypothetical one.

    Now a bypass does not compile. *)
module Read_back : sig
  val has_device_data : ('a, 'b) Vector.t -> Device.t -> bool

  val read_back_to_host : ('a, 'b) Vector.t -> Device.t -> unit
end = struct
  (** Copy vector data from a device buffer to CPU storage. *)
  let copy_device_to_host (type a b) (vec : (a, b) Vector.t) (dev : Device.t) :
      unit =
    match Vector.get_buffer vec dev with
    | None -> failwith "to_cpu: no device buffer to transfer from"
    | Some buf -> (
        let (module B : Vector.DEVICE_BUFFER) = buf in
        Log.debugf
          Log.Transfer
          "to_cpu: got buffer ptr=%Ld size=%d"
          (Int64.of_nativeint B.device_ptr)
          B.size ;
        match vec.host with
        | Vector.Bigarray_storage ba ->
            let ptr, byte_size =
              Vector_transfer.bigarray_to_ptr ba B.elem_size
            in
            B.device_to_host_ptr ptr ~byte_size ;
            (* [device_to_host_ptr] takes a bare [nativeint] address (see
             Framework_sig.ml:218-223), which is not a GC root, and this is a
             device->HOST WRITE: if [ba] were collected mid-transfer the backend
             would write into freed memory. Keep it reachable across the call —
             without this the transfer was the function's last expression and
             nothing rooted [ba]. *)
            ignore (Sys.opaque_identity ba)
        | Vector.Custom_storage {ptr; custom; length} ->
            B.device_to_host_ptr ptr ~byte_size:(length * custom.elem_size) ;
            ignore (Sys.opaque_identity ptr)
        | Vector.Device_only_storage ->
            (* Refuse rather than allocate a host mirror (FR-002): the caller
               asked for a whole-vector read-back of a vector that has none. *)
            Vector.device_only_refusal
              ~op:"Transfer.to_cpu (whole-vector read-back)"
              ~id:vec.id)

  (** Is there device-resident data for [vec] that a read-back could actually
      fetch? Two disjoint sources, and under the SoA ABI it is the second one:
      the packed buffer is not merely stale there, it was never allocated.

      [dev] selects the packed buffer, and the SoA arm ignores it: the binding
      holds ONE [soa_leaves_live] flag for the whole vector, not one per device,
      so it cannot answer "on [dev]" and does not pretend to. Every caller in
      this module — all four listed on {!read_back_to_host} below — passes the
      device the vector's own [location] names, so the two questions coincide
      there; making the flag per-device is the change that would let this arm
      honour [dev]. Until then the honest reading of the result is "somewhere on
      a device", not "on [dev]".

      That whole-vector scope is also a KNOWN EXPOSURE for a caller outside this
      module, and one this branch does not close (there is no [Transfer.mli], so
      {!read_back_to_host} is exported). Found by review of PR #375 round 5, and
      pre-existing: two transparent launches, on A then B, leave leaf buffers on
      both; [free_buffer sv B] drains B into host storage and releases B's
      leaves, and the flag stays set because A's are still allocated —
      correctly, since that is the rule which stops a per-device free from
      disowning another device's live leaves. But A's leaves are now STALE (they
      hold the first launch's data, the host holds the second's), and a direct
      [read_back_to_host sv A] then takes the SoA arm and gathers them over the
      host copy. No path inside this module reaches it: [to_cpu] returns before
      the read-back on the [CPU] location a free leaves behind, and the free
      paths guard on {!has_device_data} plus a device-naming location. Closing
      it needs the flag to name a device, which is a change to the binding
      contract and not to this function.

      That the flag is whole-vector while {!free_buffer} releases per-device is
      not a cosmetic asymmetry: [soa_free_leaves] used to CLEAR the flag on a
      per-device free, which made this function answer [false] for a device
      whose leaves were still live, and the drain-before-free in
      {!free_all_buffers} consults exactly this answer. It now NARROWS the flag
      — still set iff it already was and some leaf survives — so a per-device
      release cannot disown another device's, and a release cannot resurrect
      ownership a packed launch gave up either. *)
  let has_device_data (type a b) (vec : (a, b) Vector.t) (dev : Device.t) : bool
      =
    Option.is_some (Vector.get_buffer vec dev)
    ||
    match vec.Vector.soa with
    | Some b -> !(b.Vector.soa_leaves_live)
    | None -> false

  (** The ONE place that decides WHERE a read-back reads from, shared by every
      path in this module that pulls device data into host storage. As of
      2026-07-30 that is four call sites, and the list is exhaustive because
      [copy_device_to_host] is not in the enclosing module's signature — a fifth
      direct caller does not compile:

      - {!to_cpu} (and {!sync} through it) — the explicit read-back;
      - {!to_device}'s cross-device MIGRATION arm — a vector resident on one
        device is drained to the host before being uploaded to another;
      - {!free_buffer} and {!free_all_buffers} — drain before release.

      It exists as a function because those paths did not have this decision:
      they called {!copy_device_to_host} directly, so after a transparent SoA
      launch they downloaded a packed buffer the kernel never wrote (silently
      discarding the output) or, with no packed buffer at all, failed with "no
      device buffer to transfer from" from inside a free or a migration.

      An earlier version of this comment claimed the migration arm already
      shared this decision. It did not — it still called {!copy_device_to_host}
      — and both halves of the failure pair above were reachable through it
      (measured 2026-07-30 on two CUDA/PTX devices: the raise with no packed
      buffer, and silent wrong data with one present). [check_cross_device_soa]
      in [test_soa_emitter_equiv.ml] now pins both.

      The condition reads the flag the UPLOAD set; it does not re-decide "SoA or
      AoS?" from the device or the vector's shape. A second independent answer
      to that question is how a round trip ends up uploading leaves and
      downloading a packed buffer. *)
  let read_back_to_host (type a b) (vec : (a, b) Vector.t) (dev : Device.t) :
      unit =
    match vec.Vector.soa with
    | Some b when !(b.Vector.soa_leaves_live) ->
        (* The launch took the SoA ABI, so the results are in the N leaf buffers
           and the packed device buffer this function would otherwise download
           was never written. Reading it back would hand the caller the
           pre-launch host contents with no error anywhere. *)
        b.Vector.soa_from_device dev
    | Some _ | None -> copy_device_to_host vec dev
end

let has_device_data = Read_back.has_device_data

let read_back_to_host = Read_back.read_back_to_host

(** Transfer vector data to a device *)
let to_device (type a b) (vec : (a, b) Vector.t) (dev : Device.t) : unit =
  let loc_str =
    match vec.location with
    | Vector.CPU -> "CPU"
    | Vector.GPU _ -> "GPU"
    | Vector.Both _ -> "Both"
    | Vector.Stale_CPU _ -> "Stale_CPU"
    | Vector.Stale_GPU _ -> "Stale_GPU"
  in
  Log.debugf Log.Transfer "to_device: location=%s dev=%d" loc_str dev.id ;
  (* MIGRATION: the vector is resident on another device, so drain it to the
     host before uploading to [dev]. Through {!read_back_to_host}, not
     {!copy_device_to_host} — this arm is one of the four paths that pull device
     data into host storage, and it used to be the one that did NOT share the
     decision. After a transparent SoA launch the results are in the leaves, so
     the direct call either raised "no device buffer to transfer from" (no packed
     buffer exists on that path) or, when an earlier packed upload had left one,
     downloaded its pre-launch bytes over the host storage and discarded the
     launch's output silently. *)
  (match vec.location with
  | (Vector.GPU d | Vector.Stale_CPU d) when d.id <> dev.id ->
      read_back_to_host vec d ;
      vec.location <- Vector.Both d
  | _ -> ()) ;
  (* Check if already up-to-date on this device *)
  match vec.location with
  | Vector.GPU d when d.id = dev.id -> Log.debug Log.Transfer "-> skip (GPU)"
  | Vector.Both d when d.id = dev.id -> Log.debug Log.Transfer "-> skip (Both)"
  | Vector.Stale_CPU d when d.id = dev.id ->
      Log.debug Log.Transfer "-> skip (Stale_CPU)"
  | _ -> (
      (* A device-only vector has nothing to upload: refuse BEFORE
         [ensure_buffer] would allocate a buffer on [dev]. *)
      (match vec.host with
      | Vector.Device_only_storage ->
          Vector.device_only_refusal
            ~op:
              (Printf.sprintf
                 "Transfer.to_device (upload to %s; a freed device-only vector \
                  or one resident on another device cannot be uploaded)"
                 dev.name)
            ~id:vec.id
      | Vector.Bigarray_storage _ | Vector.Custom_storage _ -> ()) ;
      (* Ensure buffer exists and transfer *)
      let buf = ensure_buffer vec dev in
      let (module B : Vector.DEVICE_BUFFER) = buf in
      Log.debugf
        Log.Transfer
        "to_device: transferring %d bytes to dev=%d"
        (vec.length * B.elem_size)
        dev.id ;
      Log.debugf
        Log.Transfer
        "-> transferring %d bytes"
        (vec.length * B.elem_size) ;
      (match vec.host with
      | Vector.Bigarray_storage ba ->
          let ptr, byte_size = Vector_transfer.bigarray_to_ptr ba B.elem_size in
          Log.debugf
            Log.Transfer
            "to_device: calling host_ptr_to_device byte_size=%d"
            byte_size ;
          B.host_ptr_to_device ptr ~byte_size ;
          (* Same keep-alive obligation as [copy_device_to_host] above. *)
          ignore (Sys.opaque_identity ba)
      | Vector.Custom_storage {ptr; custom; length} ->
          B.host_ptr_to_device ptr ~byte_size:(length * custom.elem_size) ;
          ignore (Sys.opaque_identity ptr)
      | Vector.Device_only_storage ->
          Vector.device_only_refusal ~op:"Transfer.to_device" ~id:vec.id) ;
      vec.location <- Vector.Both dev ;
      (* The packed buffer on [dev] now holds host storage, so a read-back must
         read IT and not the leaves — and only this arm may say so, because only
         this arm uploaded anything. The three already-resident arms above
         ([GPU], [Both], [Stale_CPU], each on [dev] itself) upload nothing — and
         [Stale_CPU dev] on a transparent vector is exactly the case where the
         leaves ARE the device copy — so clearing there would disown live
         leaves.

         Reaching here with the flag set means the host copy was authoritative or
         in sync first — that is what makes the upload above meaningful, and it is
         what every path into this arm establishes: the migration arm at the top of
         this function has just drained the other device through
         [read_back_to_host], and the remaining locations ([CPU], [Both d],
         [Stale_GPU d]) each assert the host copy is not behind. So this does not
         discard a pending leaf result; it
         stops [read_back_to_host] from answering a question about [dev] by
         reading leaves on another device, which is the one thing the whole-vector
         flag cannot express (see {!Read_back.has_device_data}).

         [Execute.transfer_vectors_to_device] normalises the same flag on the
         packed-launch path. That site is not this one: it runs before a LAUNCH
         and gathers first, this one after an UPLOAD that has already made the
         host copy authoritative. Both exist because the flag names the ABI of the
         most recent operation that made a device copy authoritative, and an
         upload is such an operation. *)
      match vec.Vector.soa with
      | Some b -> b.Vector.soa_leaves_live := false
      | None -> ())

(** Transfer vector data from device to CPU.
    @param force
      If true, always transfer even if location is Both (useful after kernel
      writes) *)
let to_cpu ?(force = false) (type a b) (vec : (a, b) Vector.t) : unit =
  Log.debugf
    Log.Transfer
    "to_cpu: CALLED: force=%b location=%s"
    force
    (match vec.location with
    | Vector.CPU -> "CPU"
    | Vector.GPU _ -> "GPU"
    | Vector.Both _ -> "Both"
    | Vector.Stale_GPU _ -> "Stale_GPU"
    | Vector.Stale_CPU _ -> "Stale_CPU") ;
  let needs_transfer =
    match vec.location with
    | Vector.CPU -> false (* No device buffer *)
    | Vector.Both _ -> force (* Transfer if forced *)
    | Vector.Stale_GPU _ -> false (* CPU already authoritative *)
    | Vector.GPU _ | Vector.Stale_CPU _ -> true
  in
  Log.debugf Log.Transfer "to_cpu: needs_transfer=%b" needs_transfer ;
  if needs_transfer then begin
    let dev =
      match vec.location with
      | Vector.GPU d | Vector.Stale_CPU d | Vector.Both d -> d
      | _ -> failwith "to_cpu: no device"
    in
    Log.debugf
      Log.Transfer
      "to_cpu: transferring from dev=%d (force=%b)"
      dev.id
      force ;
    (* [read_back_to_host] carries the SoA-vs-packed decision; the packed arm's
       own "no device buffer to transfer from" failure comes from
       [copy_device_to_host], so the precondition stays where the read happens
       rather than being re-derived here. *)
    read_back_to_host vec dev ;
    vec.location <- Vector.Both dev
  end
  else
    Log.debugf
      Log.Transfer
      "to_cpu: skip (location=%s, force=%b)"
      (match vec.location with
      | Vector.CPU -> "CPU"
      | Vector.GPU _ -> "GPU"
      | Vector.Both _ -> "Both"
      | Vector.Stale_CPU _ -> "Stale_CPU"
      | Vector.Stale_GPU _ -> "Stale_GPU")
      force

(** Ensure vector is fully synchronized *)
let sync (type a b) (vec : (a, b) Vector.t) : unit =
  match vec.location with
  | Vector.CPU -> ()
  | Vector.Both _ -> ()
  | Vector.GPU dev ->
      to_cpu vec ;
      vec.location <- Vector.Both dev
  | Vector.Stale_CPU dev ->
      to_cpu vec ;
      vec.location <- Vector.Both dev
  | Vector.Stale_GPU dev ->
      to_device vec dev ;
      vec.location <- Vector.Both dev

(** {1 Keep-alive ledger for stream-ordered work}

    A copy or launch enqueued on a stream may still be reading or writing its
    operands after the call returns. The backends see raw addresses, which are
    not GC roots, so SPOC roots the operands here — Bigarray sources and
    destinations, the vectors involved — until it KNOWS the work has completed:

    - the stream was synchronised ({!synchronize_stream}, {!destroy_stream}), or
      its device was ({!flush});
    - an event recorded after the work was synchronised, or queried complete;
    - transitively: when a stream [b] that waited on an event of stream [a]
      completes past that wait, [a]'s work up to the event has completed too.
      This is what releases a copy stream that is never synchronised itself —
      the spoc_llm M3 pattern (copy on [a], record [e], [b] waits on [e], sync
      [b]).

    Entries are ordered by a per-stream sequence number. The ledger is protected
    by a mutex and is never touched from a GC finaliser (which could run inside
    [Mutex.protect] on the same domain and self-deadlock). *)
module Ledger = struct
  type entry = {
    seq : int;
    keep : Obj.t;  (** Rooted for liveness only; never inspected. *)
    vec_ids : int list;
    host_ranges : (nativeint * int) list;  (** (address, bytes) in flight. *)
  }

  type stream_state = {
    sid : int;
    sdev : Device.t;
    mutable next_seq : int;
    mutable pending : entry list;
    mutable waits : (int * stream_state * int) list;
        (** (seq of the wait on this stream, waited stream, its seq) *)
    mutable destroyed : bool;
        (** Set by [destroy_stream]. A destroyed handle is refused: the driver
            may have recycled it, and work enqueued through it would land in a
            ledger nothing consults any more. *)
  }

  let lock = Mutex.create ()

  let next_sid = ref 0

  let live : (int, stream_state) Hashtbl.t = Hashtbl.create 8

  let new_stream (dev : Device.t) : stream_state =
    Mutex.protect lock (fun () ->
        incr next_sid ;
        let st =
          {
            sid = !next_sid;
            sdev = dev;
            next_seq = 0;
            pending = [];
            waits = [];
            destroyed = false;
          }
        in
        Hashtbl.replace live st.sid st ;
        st)

  let forget (st : stream_state) =
    Mutex.protect lock (fun () ->
        st.destroyed <- true ;
        Hashtbl.remove live st.sid)

  let check_live ~(op : string) (st : stream_state) : unit =
    if st.destroyed then invalid_arg (op ^ ": the stream has been destroyed")

  let enqueue (st : stream_state) ~(keep : Obj.t) ~(vec_ids : int list)
      ~(host_ranges : (nativeint * int) list) : unit =
    Mutex.protect lock (fun () ->
        let seq = st.next_seq in
        st.next_seq <- seq + 1 ;
        st.pending <- {seq; keep; vec_ids; host_ranges} :: st.pending)

  (** Sequence number of the last operation enqueued so far: what an event
      recorded now covers. *)
  let mark (st : stream_state) : int =
    Mutex.protect lock (fun () -> st.next_seq - 1)

  (** A wait is itself an operation on [b]: once [b] has completed past it, the
      waited work on the other stream has completed. *)
  let add_wait (b : stream_state) ~(on : stream_state * int) : unit =
    Mutex.protect lock (fun () ->
        let seq = b.next_seq in
        b.next_seq <- seq + 1 ;
        let a, aseq = on in
        b.waits <- (seq, a, aseq) :: b.waits)

  (* Each wait is removed before recursing, so a wait cycle terminates. *)
  let rec release_locked (st : stream_state) (upto : int) : unit =
    st.pending <- List.filter (fun e -> e.seq > upto) st.pending ;
    let reached, rest = List.partition (fun (s, _, _) -> s <= upto) st.waits in
    st.waits <- rest ;
    List.iter (fun (_, a, aseq) -> release_locked a aseq) reached

  let release (st : stream_state) (upto : int) : unit =
    Mutex.protect lock (fun () -> release_locked st upto)

  let release_all (st : stream_state) : unit = release st max_int

  let release_device (dev : Device.t) : unit =
    Mutex.protect lock (fun () ->
        Hashtbl.iter
          (fun _ st -> if st.sdev.id = dev.id then release_locked st max_int)
          live)

  let pending_count (st : stream_state) : int =
    Mutex.protect lock (fun () -> List.length st.pending)

  let holds_vector (id : int) : bool =
    Mutex.protect lock (fun () ->
        Hashtbl.fold
          (fun _ st acc ->
            acc || List.exists (fun e -> List.mem id e.vec_ids) st.pending)
          live
          false)

  let holds_host_range (addr : nativeint) (bytes : int) : bool =
    let hi = Nativeint.add addr (Nativeint.of_int bytes) in
    let overlaps (a, n) =
      let a_hi = Nativeint.add a (Nativeint.of_int n) in
      Nativeint.compare a hi < 0 && Nativeint.compare addr a_hi < 0
    in
    Mutex.protect lock (fun () ->
        Hashtbl.fold
          (fun _ st acc ->
            acc
            || List.exists
                 (fun e -> List.exists overlaps e.host_ranges)
                 st.pending)
          live
          false)
end

(** Refuse to free device memory that pending stream-ordered work may still
    touch (FR-021): freeing it would put the DMA engine or a kernel on freed
    memory. *)
let refuse_if_in_flight ~(op : string) (vec : (_, _) Vector.t) : unit =
  if Ledger.holds_vector vec.id then
    invalid_arg
      (Printf.sprintf
         "%s: vector #%d is still referenced by stream-ordered work; \
          synchronize the stream (or an event recorded after the work) first"
         op
         vec.id)

(** [true] iff [vec] has no host storage ({!create_device_only} on a non-CPU
    device). *)
let is_device_only = Vector.is_device_only

(** {1 Buffer Cleanup} *)

(** Release the device memory [vec] holds on [dev], draining it to host storage
    first.

    FOUR steps, and not one of them may sit inside [get_buffer]'s [Some] arm.
    Under the SoA ABI the leaves are the only device memory the vector has and
    the packed buffer is never allocated, so on a transparent vector
    [get_buffer] returns [None] ALWAYS — an early return there does not skip a
    corner case, it skips the whole function.

    The previous version of this comment already said "under the SoA ABI they
    are the only device memory this vector has" while the bookkeeping below it
    still lived in the [Some buf] arm and assumed a packed buffer existed. That
    mismatch was the bug, not a wording slip: the location reset is the only
    code here that assigns [CPU], so a transparent SoA vector came out of this
    function with its leaves freed and [location] still naming [dev]. Two
    measured consequences, both on a vector whose data was intact in host
    storage:

    - [to_cpu ~force:true] raised
      [Failure "to_cpu: no device buffer to transfer from"] — [Both dev] plus
      [force] means "read the device back", and there is nothing there to read;
    - [to_device] on the same device took the "skip (Both)" short-circuit and
      allocated nothing, reinstating the exact short-circuit the [Stale_CPU]
      work earlier in this branch exists to eliminate.

    {!free_all_buffers} was never exposed to this because it assigns [CPU]
    unconditionally at the end. This function now does the same, scoped to
    [dev]. *)
let free_buffer (vec : (_, _) Vector.t) (dev : Device.t) : unit =
  refuse_if_in_flight ~op:"Transfer.free_buffer" vec ;
  (* A device-only vector has nothing to drain into: freeing it DISCARDS its
     contents (documented on [create_device_only]). The location reset in step
     4 then leaves [CPU] over [Device_only_storage], which every host and
     upload path refuses — the vector is unusable afterwards, never silently
     re-uploaded from nowhere. *)
  let drain = not (is_device_only vec) in
  (* 1. Read back BEFORE freeing, and through {!read_back_to_host}: an
     SoA-dispatched vector has no packed buffer of its own, so the old
     [None -> ()] early return threw the output of a transparent launch away
     outright. [has_device_data] keeps the no-op for the case that return was
     actually for — an AoS vector with nothing on this device. *)
  (match vec.location with
  | (Vector.GPU d | Vector.Stale_CPU d)
    when drain && d.id = dev.id && has_device_data vec dev ->
      read_back_to_host vec dev ;
      vec.location <- Vector.Both dev
  | _ -> ()) ;
  (* 2. Release the LEAVES. Under the SoA ABI they are the only device memory
     this vector has, so without this the call returned having freed nothing at
     all. Ordered after the read-back above and before the flag it updates is
     needed again. *)
  (match vec.Vector.soa with
  | Some b -> b.Vector.soa_free_leaves (Some dev)
  | None -> ()) ;
  (* 3. Release the packed buffer, if this vector ever had one on [dev]. *)
  (match Vector.get_buffer vec dev with
  | None -> ()
  | Some buf ->
      let (module B : Vector.DEVICE_BUFFER) = buf in
      B.free () ;
      Gpu_memory.track_free (B.size * B.elem_size) ;
      Hashtbl.remove vec.device_buffers dev.id) ;
  (* 4. Location. Outside step 3's [Some] arm, which is the whole fix: steps 2
     and 3 together released everything this vector held on [dev], whether that
     was leaves, a packed buffer, or both, so the state that follows is the same
     in all three cases and [CPU] is it.

     Guarded on the device alone rather than per-constructor: every
     device-naming constructor asserts something about [dev]'s contents, and
     after steps 2-3 [dev] holds nothing, so all four collapse to the same
     answer. A location naming a DIFFERENT device is untouched — that device's
     memory was not freed. *)
  match vec.location with
  | (Vector.GPU d | Vector.Both d | Vector.Stale_CPU d | Vector.Stale_GPU d)
    when d.id = dev.id ->
      vec.location <- Vector.CPU
  | _ -> ()

(** Free all device buffers for a vector *)
let free_all_buffers (vec : (_, _) Vector.t) : unit =
  refuse_if_in_flight ~op:"Transfer.free_all_buffers" vec ;
  (* Device-only: discard, as in {!free_buffer}. *)
  let drain = not (is_device_only vec) in
  (* Same SoA-aware read-back as {!free_buffer}. Pre-fix this arm called
     [copy_device_to_host] on a vector whose packed buffer does not exist after a
     transparent launch, so freeing without an explicit [to_cpu] raised
     "no device buffer to transfer from" from inside cleanup — and, when a packed
     buffer DID exist from an earlier AoS launch, silently overwrote the host
     storage with its pre-SoA-launch contents. *)
  (match vec.location with
  | (Vector.GPU d | Vector.Stale_CPU d) when drain && has_device_data vec d ->
      read_back_to_host vec d ;
      vec.location <- Vector.CPU
  | _ -> ()) ;
  (* Same leaf release as {!free_buffer}, on every device. Pre-fix this function
     was a no-op on a transparent SoA vector's memory: [device_buffers] is EMPTY
     under that ABI, so the iteration below had nothing to free and the call
     reported success having released zero bytes (measured with
     [Gpu_memory.usage()]: 3840 B -> 4224 B across a launch + free). *)
  (match vec.Vector.soa with
  | Some b -> b.Vector.soa_free_leaves None
  | None -> ()) ;
  Hashtbl.iter
    (fun _ buf ->
      let (module B : Vector.DEVICE_BUFFER) = buf in
      B.free () ;
      Gpu_memory.track_free (B.size * B.elem_size))
    vec.device_buffers ;
  Hashtbl.clear vec.device_buffers ;
  vec.location <- Vector.CPU

(** {1 Device Synchronization} *)

(** Synchronize all pending operations on a device *)
let flush (dev : Device.t) : unit =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get dev.backend_id in
      B.Device.synchronize backend_dev ;
      (* Everything enqueued on any stream of [dev] has completed. *)
      Ledger.release_device dev

(** {1 Stream Operations} *)

let backend_of (dev : Device.t) : (module Framework_sig.BACKEND) =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some b -> b

(** Raise [Feature_not_supported] for [backend]: the capability refusal of this
    section, as opposed to [Invalid_argument] for caller errors. *)
let refuse_capability ~(backend : string) (feature : string) : 'a =
  Sarek_backend_error.Backend_error.raise_error
    (Sarek_backend_error.Backend_error.feature_not_supported ~backend feature)

(** Stream handle - packages backend stream with its operations.

    [stream_ref] is the backend stream wrapped by the backend's own
    [wrap_stream], so any other unpacking of the same backend can recover it
    ({!Framework_sig.stream_ref}). [state] is this stream's entry in the
    keep-alive {!Ledger}.

    Streams created by {!create_stream} are BLOCKING streams (they synchronise
    with the device's legacy default stream). The auto-transfer path relies on
    that: after an async ranged write the vector is [Stale_CPU], and the
    implicit read-back runs on the default stream, which then waits for the
    write. Switching to non-blocking streams requires making auto-sync
    stream-aware first (FR-019). *)
module type STREAM = sig
  val device : Device.t

  val synchronize : unit -> unit

  val destroy : unit -> unit

  val stream_ref : Framework_sig.stream_ref

  val state : Ledger.stream_state
end

type stream = (module STREAM)

(** Create a new stream on a device *)
let create_stream (dev : Device.t) : stream =
  let (module B : Framework_sig.BACKEND) = backend_of dev in
  let backend_dev = B.Device.get dev.backend_id in
  let s = B.Stream.create backend_dev in
  let st = Ledger.new_stream dev in
  (module struct
    let device = dev

    let stream_ref = B.wrap_stream s

    let state = st

    let synchronize () =
      Ledger.check_live ~op:"Transfer.synchronize_stream" st ;
      B.Stream.synchronize s ;
      Ledger.release_all st

    (* Synchronise first: the ledger may only let go of what the stream's work
       references once that work has completed. *)
    let destroy () =
      Ledger.check_live ~op:"Transfer.destroy_stream" st ;
      synchronize () ;
      B.Stream.destroy s ;
      Ledger.forget st
  end : STREAM)

(* One ledger state per device for the default stream: every [default_stream]
   handle names the same backend stream, so they must share one ledger, or
   synchronising one handle would not release work enqueued through another. *)
let default_states : (int, Ledger.stream_state) Hashtbl.t = Hashtbl.create 4

let default_states_lock = Mutex.create ()

(** Get default stream for a device. All handles for one device share one ledger
    state. *)
let default_stream (dev : Device.t) : stream =
  let (module B : Framework_sig.BACKEND) = backend_of dev in
  let backend_dev = B.Device.get dev.backend_id in
  let s = B.Stream.default backend_dev in
  let st =
    Mutex.protect default_states_lock (fun () ->
        match Hashtbl.find_opt default_states dev.id with
        | Some st -> st
        | None ->
            let st = Ledger.new_stream dev in
            Hashtbl.replace default_states dev.id st ;
            st)
  in
  (module struct
    let device = dev

    let stream_ref = B.wrap_stream s

    let state = st

    let synchronize () =
      B.Stream.synchronize s ;
      Ledger.release_all st

    let destroy () = () (* Don't destroy default stream *)
  end : STREAM)

let synchronize_stream (s : stream) =
  let (module S : STREAM) = s in
  S.synchronize ()

let destroy_stream (s : stream) =
  let (module S : STREAM) = s in
  S.destroy ()

let stream_device (s : stream) : Device.t =
  let (module S : STREAM) = s in
  S.device

(** The device a stream belongs to and its backend-tagged handle — what
    [Execute] needs to launch on it. *)
let stream_handle (s : stream) : Device.t * Framework_sig.stream_ref =
  let (module S : STREAM) = s in
  Ledger.check_live ~op:"Execute (launch on a stream)" S.state ;
  (S.device, S.stream_ref)

(** Root [keep] (and record [vec_ids] as in use) until the work just enqueued on
    [s] has completed — what [Execute] does for the vectors of a launch it
    enqueued on [s] (the backend's own kernel-argument retention roots the
    argument cells, not the vectors whose buffers the kernel touches). *)
let retain_on_stream (s : stream) ~(keep : Obj.t) ~(vec_ids : int list) : unit =
  let (module S : STREAM) = s in
  Ledger.enqueue S.state ~keep ~vec_ids ~host_ranges:[]

(** Recover the backend's own stream from a core [stream]; refused when it
    belongs to another backend or another device than [dev]. *)
let backend_stream (type s) ~(op : string) ~(dev : Device.t)
    (unwrap : Framework_sig.stream_ref -> s option) (st : stream) : s =
  let (module S : STREAM) = st in
  Ledger.check_live ~op S.state ;
  if S.device.id <> dev.id then
    invalid_arg
      (Printf.sprintf
         "%s: the stream belongs to device %d (%s), the operation targets \
          device %d (%s)"
         op
         S.device.id
         S.device.name
         dev.id
         dev.name) ;
  match unwrap S.stream_ref with
  | Some s -> s
  | None ->
      invalid_arg
        (Printf.sprintf "%s: the stream belongs to another backend" op)

(** {1 Events} *)

(** An event on one device. [ev_recorded] remembers which stream and ledger
    position the latest {!record_event} covered, so synchronising or querying
    the event can release exactly the keep-alive entries it proves complete.
    Recording again replaces it (the latest record is what waits, queries and
    releases refer to, as in CUDA/HIP). *)
type event = {
  ev_device : Device.t;
  ev_ref : Framework_sig.event_ref;
  mutable ev_recorded : (Ledger.stream_state * int) option;
  mutable ev_destroyed : bool;
      (** A destroyed event is refused: the driver may have recycled its handle,
          and a query answering for another event could release ledger entries
          early. *)
}

(** Create an event on [dev]. The device is selected first: HIP and CUDA create
    events on the CURRENT device, which a GC finaliser freeing a buffer may have
    changed. *)
let create_event (dev : Device.t) : event =
  let (module B : Framework_sig.BACKEND) = backend_of dev in
  B.Device.set_current (B.Device.get dev.backend_id) ;
  {
    ev_device = dev;
    ev_ref = B.wrap_event (B.Event.create ());
    ev_recorded = None;
    ev_destroyed = false;
  }

let backend_event (type e) ~(op : string)
    (unwrap : Framework_sig.event_ref -> e option) (ev : event) : e =
  if ev.ev_destroyed then invalid_arg (op ^ ": the event has been destroyed") ;
  match unwrap ev.ev_ref with
  | Some e -> e
  | None ->
      invalid_arg (Printf.sprintf "%s: the event belongs to another backend" op)

(** Record [ev] on [s]: it completes once all work enqueued on [s] so far has
    completed. Refused across devices. *)
let record_event (ev : event) (s : stream) : unit =
  let op = "Transfer.record_event" in
  let (module B : Framework_sig.BACKEND) = backend_of ev.ev_device in
  let bs = backend_stream ~op ~dev:ev.ev_device B.unwrap_stream s in
  let (module S : STREAM) = s in
  (* Mark BEFORE the backend record: ledger entries are added after their
     backend call, so a mark taken first never covers work the event does not
     (another domain may enqueue on the same stream in between). *)
  let mark = Ledger.mark S.state in
  (* The default stream is the NULL handle, which the driver reads as the
     CURRENT device's: select the event's device first. *)
  B.Device.set_current (B.Device.get ev.ev_device.backend_id) ;
  B.Event.record (backend_event ~op B.unwrap_event ev) bs ;
  ev.ev_recorded <- Some (S.state, mark)

(** Make every operation enqueued on [s] after this call wait for the work [ev]
    recorded. Refused across devices. On Native/Interpreter streams are
    synchronous, so the wait is already satisfied. *)
let stream_wait_event (s : stream) (ev : event) : unit =
  let op = "Transfer.stream_wait_event" in
  let (module B : Framework_sig.BACKEND) = backend_of ev.ev_device in
  let bs = backend_stream ~op ~dev:ev.ev_device B.unwrap_stream s in
  B.Async.stream_wait_event bs (backend_event ~op B.unwrap_event ev) ;
  let (module S : STREAM) = s in
  match ev.ev_recorded with
  | Some on -> Ledger.add_wait S.state ~on
  | None -> ()

let release_event (ev : event) : unit =
  match ev.ev_recorded with
  | Some (st, upto) -> Ledger.release st upto
  | None -> ()

(** [true] once the recorded work has completed (or the event was never
    recorded). Never blocks. A [true] answer releases what the event covers. *)
let event_query (ev : event) : bool =
  let (module B : Framework_sig.BACKEND) = backend_of ev.ev_device in
  let done_ =
    B.Async.event_query
      (backend_event ~op:"Transfer.event_query" B.unwrap_event ev)
  in
  if done_ then release_event ev ;
  done_

(** Block until the recorded work has completed, then release what it covers. *)
let synchronize_event (ev : event) : unit =
  let (module B : Framework_sig.BACKEND) = backend_of ev.ev_device in
  B.Event.synchronize
    (backend_event ~op:"Transfer.synchronize_event" B.unwrap_event ev) ;
  release_event ev

(** Destroy the event. Keep-alive entries it would have released stay until
    their stream is synchronised. *)
let destroy_event (ev : event) : unit =
  let (module B : Framework_sig.BACKEND) = backend_of ev.ev_device in
  B.Event.destroy (backend_event ~op:"Transfer.destroy_event" B.unwrap_event ev) ;
  ev.ev_destroyed <- true

(** {1 Device-only vectors} *)

(** Allocate a vector of [length] elements that lives only on [dev].

    On a GPU device no host storage is allocated at all: host element access
    ([Vector.get]/[set]/[fill]/[iter]/[to_bigarray]/...), whole-vector read-back
    ([to_cpu], auto-sync) and host-side copies all raise [Invalid_argument]
    naming the vector as device-only. Data moves in and out only through the
    ranged copies below ({!copy_to_device}, {!copy_bytes_to_device},
    {!copy_ptr_to_device}, {!copy_to_host}, {!copy_bytes_to_host}) or kernels.
    Freeing its buffer ({!free_buffer}, {!free_all_buffers}) discards the
    contents, and the vector is unusable afterwards.

    On Metal the buffer is a shared-storage MTLBuffer (Apple silicon unified
    memory): still no host mirror, and the same refusals apply.

    On a CPU device (Native, Interpreter) device memory IS host memory — kernels
    there read the vector's host storage — so the vector gets ordinary host
    storage, aliased by its device buffer, and host element access works.
    Portable code must not rely on that.

    Refused ([Invalid_argument]): [length] outside [1 .. Int32.max_int] (kernels
    index with int32; split larger caches over several vectors) and custom
    (record) kinds. Refused ([Feature_not_supported]): backends without ranged
    copies (OpenCL, Vulkan, WebGPU), where nothing but a kernel could ever fill
    the vector. *)
let create_device_only (type a b) (dev : Device.t) (kind : (a, b) Vector.kind)
    (length : int) : (a, b) Vector.t =
  if length < 1 || length > Int32.to_int Int32.max_int then
    invalid_arg
      (Printf.sprintf
         "Transfer.create_device_only: length %d outside [1, %ld] (kernels \
          index with int32; split larger caches over several vectors)"
         length
         Int32.max_int) ;
  (match kind with
  | Vector.Scalar _ -> ()
  | Vector.Custom _ ->
      invalid_arg
        "Transfer.create_device_only: custom (record) kinds are not supported; \
         ranged copies are typed over Bigarray element kinds") ;
  let (module B : Framework_sig.BACKEND) = backend_of dev in
  if not B.Async.support.ranged_copy then
    refuse_capability
      ~backend:B.name
      "device-only vectors (they need ranged copies, implemented on HIP, CUDA, \
       Metal, Native and Interpreter only)" ;
  let vec =
    if dev.capabilities.is_cpu then Vector.create kind length
    else Vector.create_device_only_record kind dev length
  in
  ignore (ensure_buffer vec dev) ;
  vec.location <- Vector.GPU dev ;
  vec

(** {1 Ranged copies} *)

(** The device a ranged copy on [vec] targets, and its buffer there, after
    checking the preconditions. A ranged copy needs the device copy to be
    CURRENT: a partial write into a stale device copy would leave neither copy
    authoritative, and a read from one would return stale data.

    Current means: [vec] is device-only (device-current by definition), or its
    location is [GPU d], [Both d] or [Stale_CPU d], or — on a CPU device, where
    the buffer aliases host storage — any location naming [d]. *)
let range_target (type a b) ~(op : string) (vec : (a, b) Vector.t) :
    Device.t * Vector.device_buffer =
  (match vec.kind with
  | Vector.Scalar _ -> ()
  | Vector.Custom _ ->
      invalid_arg
        (op ^ ": custom (record) kinds are not supported by ranged copies")) ;
  let dev =
    match vec.location with
    | Vector.GPU d | Vector.Both d | Vector.Stale_CPU d -> d
    | Vector.Stale_GPU d when is_device_only vec || d.capabilities.is_cpu -> d
    | Vector.Stale_GPU _ | Vector.CPU ->
        if is_device_only vec then
          invalid_arg
            (Printf.sprintf
               "%s: device-only vector #%d has no device buffer (freed)"
               op
               vec.id)
        else
          invalid_arg
            (Printf.sprintf
               "%s: vector #%d's host copy is authoritative (location %s); \
                upload it with Transfer.to_device first"
               op
               vec.id
               (Vector.location_to_string vec.location))
  in
  match Vector.get_buffer vec dev with
  | Some buf -> (dev, buf)
  | None ->
      invalid_arg
        (Printf.sprintf "%s: vector #%d has no buffer on %s" op vec.id dev.name)

let check_bytes ~(op : string) ~(total : int) ~(off : int) ~(byte_size : int) =
  if off < 0 || byte_size < 0 || off + byte_size > total then
    invalid_arg
      (Printf.sprintf
         "%s: byte range [%d, %d) outside the vector's %d bytes"
         op
         off
         (off + byte_size)
         total)

let vector_bytes (vec : (_, _) Vector.t) =
  vec.length * Vector.elem_size vec.kind

(* The one host->device path every public variant goes through. [keep] is
   rooted until the copy completes: synchronously across the backend call, in
   the ledger when enqueued on a stream. *)
let h2d_range (type a b) ~(op : string) ?(stream : stream option)
    ~(src_ptr : nativeint) ~(byte_size : int) ~(keep : Obj.t)
    ~(dst : (a, b) Vector.t) ~(dst_byte_offset : int) () : unit =
  let dev, buf = range_target ~op dst in
  check_bytes ~op ~total:(vector_bytes dst) ~off:dst_byte_offset ~byte_size ;
  let (module Buf : Vector.DEVICE_BUFFER) = buf in
  (match stream with
  | None ->
      Buf.range_from_host
        ~src_ptr
        ~dst_offset:dst_byte_offset
        ~byte_size
        ~stream:None ;
      ignore (Sys.opaque_identity keep)
  | Some s ->
      let (module S : STREAM) = s in
      Ledger.check_live ~op S.state ;
      if S.device.id <> dev.id then
        invalid_arg
          (Printf.sprintf
             "%s: the stream belongs to device %d, the vector to device %d"
             op
             S.device.id
             dev.id) ;
      Buf.range_from_host
        ~src_ptr
        ~dst_offset:dst_byte_offset
        ~byte_size
        ~stream:(Some S.stream_ref) ;
      Ledger.enqueue
        S.state
        ~keep:(Obj.repr (keep, dst))
        ~vec_ids:[dst.id]
        ~host_ranges:[(src_ptr, byte_size)]) ;
  (* The device copy now holds data the host copy lacks. *)
  match dst.location with
  | Vector.Both d when not d.capabilities.is_cpu ->
      dst.location <- Vector.Stale_CPU d
  | Vector.Both _ | Vector.GPU _ | Vector.Stale_CPU _ | Vector.Stale_GPU _
  | Vector.CPU ->
      ()

let bigarray_addr (ba : (_, _, Bigarray.c_layout) Bigarray.Array1.t) : nativeint
    =
  Ctypes.raw_address_of_ptr (Memory.bigarray_void_ptr ba)

(** Copy all of [src] into [dst] starting at element [dst_offset].

    Without [~stream] the copy has completed when the call returns. With it, the
    copy is enqueued on the stream and may still be running; SPOC keeps [src]
    and [dst] alive until the stream (or an event recorded after the copy) is
    synchronised. Writing into [src] before then races with the DMA: ordering
    that is the caller's job.

    [dst] must be device-current (see {!range_target}); a vector in [Both]
    becomes [Stale_CPU], so auto-sync later reads the device copy. Raises
    [Invalid_argument] when the range does not fit, and [Feature_not_supported]
    on backends without ranged copies. Slice the source with
    [Bigarray.Array1.sub].

    Pageable sources: HIP and CUDA stage a stream copy from pageable memory
    through a driver bounce buffer. It is correct, but the host call may block
    and the copy may not overlap compute; source from {!alloc_pinned} or
    {!register_host} memory when overlap matters. Not refused, because the
    result is right and the rate can be equal (28.1 vs 28.6 GB/s measured on an
    RX 7900 XTX, CHANGES.md 2026-10). *)
let copy_to_device (type a b) ?(stream : stream option)
    (src : (a, b, Bigarray.c_layout) Bigarray.Array1.t) ~(dst : (a, b) Vector.t)
    ~(dst_offset : int) : unit =
  let op = "Transfer.copy_to_device" in
  let n = Bigarray.Array1.dim src in
  if dst_offset < 0 || dst_offset + n > dst.length then
    invalid_arg
      (Printf.sprintf
         "%s: elements [%d, %d) outside vector of length %d"
         op
         dst_offset
         (dst_offset + n)
         dst.length) ;
  let esz = Vector.elem_size dst.kind in
  h2d_range
    ~op
    ?stream
    ~src_ptr:(bigarray_addr src)
    ~byte_size:(n * esz)
    ~keep:(Obj.repr src)
    ~dst
    ~dst_byte_offset:(dst_offset * esz)
    ()

(** Byte-level {!copy_to_device}: copy every byte of [src] (any element kind)
    into [dst] at byte offset [dst_byte_offset]. For blocks that mix element
    types (e.g. int32 quant nibbles followed by f32 scales) read into a [char]
    buffer. Same stream, keep-alive and location contract. *)
let copy_bytes_to_device ?(stream : stream option)
    (src : (_, _, Bigarray.c_layout) Bigarray.Array1.t) ~(dst : (_, _) Vector.t)
    ~(dst_byte_offset : int) : unit =
  h2d_range
    ~op:"Transfer.copy_bytes_to_device"
    ?stream
    ~src_ptr:(bigarray_addr src)
    ~byte_size:(Bigarray.Array1.size_in_bytes src)
    ~keep:(Obj.repr src)
    ~dst
    ~dst_byte_offset
    ()

(** Raw-address {!copy_bytes_to_device}. [src_ptr] is NOT rooted: the caller
    keeps [byte_size] bytes at [src_ptr] alive and unmodified until the copy has
    completed — the call's return without [~stream], the stream or a later
    event's completion with it. *)
let copy_ptr_to_device ?(stream : stream option) ~(src_ptr : nativeint)
    ~(byte_size : int) ~(dst : (_, _) Vector.t) ~(dst_byte_offset : int) () :
    unit =
  h2d_range
    ~op:"Transfer.copy_ptr_to_device"
    ?stream
    ~src_ptr
    ~byte_size
    ~keep:(Obj.repr ())
    ~dst
    ~dst_byte_offset
    ()

(* The one device->host path every public variant goes through: copy
   [size_in_bytes dst] bytes of [src] from byte [src_byte_offset] into [dst].
   [dst] and [src] are rooted until the copy completes: synchronously across
   the backend call, in the ledger (with [dst]'s byte range as in flight) when
   enqueued on a stream. [src]'s location is unchanged. *)
let d2h_range (type a b) ~(op : string) ?(stream : stream option)
    ~(src : (a, b) Vector.t) ~(src_byte_offset : int)
    (dst : (_, _, Bigarray.c_layout) Bigarray.Array1.t) : unit =
  let dev, buf = range_target ~op src in
  let byte_size = Bigarray.Array1.size_in_bytes dst in
  check_bytes ~op ~total:(vector_bytes src) ~off:src_byte_offset ~byte_size ;
  let dst_ptr = bigarray_addr dst in
  let (module Buf : Vector.DEVICE_BUFFER) = buf in
  match stream with
  | None ->
      Buf.range_to_host
        ~src_offset:src_byte_offset
        ~dst_ptr
        ~byte_size
        ~stream:None ;
      ignore (Sys.opaque_identity dst)
  | Some s ->
      let (module S : STREAM) = s in
      Ledger.check_live ~op S.state ;
      if S.device.id <> dev.id then
        invalid_arg
          (Printf.sprintf
             "%s: the stream belongs to device %d, the vector to device %d"
             op
             S.device.id
             dev.id) ;
      Buf.range_to_host
        ~src_offset:src_byte_offset
        ~dst_ptr
        ~byte_size
        ~stream:(Some S.stream_ref) ;
      Ledger.enqueue
        S.state
        ~keep:(Obj.repr (dst, src))
        ~vec_ids:[src.id]
        ~host_ranges:[(dst_ptr, byte_size)]

(** Copy [Bigarray.Array1.dim dst] elements of [src], starting at element
    [src_offset], into [dst] — the explicit read-back for device-only vectors.
    [src] must be device-current. With [~stream] the copy may still be running
    on return and SPOC roots [dst] until it has completed; read [dst] only after
    synchronising (see {!copy_bytes_to_host} for the destination's state
    meanwhile). [src]'s location is unchanged. *)
let copy_to_host (type a b) ?(stream : stream option) ~(src : (a, b) Vector.t)
    ~(src_offset : int) (dst : (a, b, Bigarray.c_layout) Bigarray.Array1.t) :
    unit =
  let op = "Transfer.copy_to_host" in
  let n = Bigarray.Array1.dim dst in
  if src_offset < 0 || src_offset + n > src.length then
    invalid_arg
      (Printf.sprintf
         "%s: elements [%d, %d) outside vector of length %d"
         op
         src_offset
         (src_offset + n)
         src.length) ;
  d2h_range
    ~op
    ?stream
    ~src
    ~src_byte_offset:(src_offset * Vector.elem_size src.kind)
    dst

(** Byte-level {!copy_to_host}: copy [Bigarray.Array1.size_in_bytes dst] bytes
    of [src] (any element kind), starting at byte [src_byte_offset], into [dst]
    (any element kind) — e.g. a range of an [int32] vector into a [char] slot.
    Slice the destination with [Bigarray.Array1.sub]. The mirror of
    {!copy_bytes_to_device}; same location rules as {!copy_to_host}: [src] must
    be device-current and its location is unchanged. Raises [Invalid_argument]
    when the byte range does not fit, and [Feature_not_supported] on backends
    without ranged copies.

    Without [~stream] the copy has completed when the call returns. With it, the
    copy is enqueued and may still be running: until the stream (or an event
    recorded after the copy) has been synchronised, or {!flush}, [dst]'s
    contents are undefined — reading it may see old, new or mixed bytes, and
    writing it races with the DMA, which may overwrite it. SPOC keeps [dst] and
    [src] alive meanwhile and records [dst]'s bytes as in flight, so
    {!free_pinned} and {!unregister_host} refuse a buffer they overlap. HIP and
    CUDA may stage a stream copy into pageable memory through a driver bounce
    buffer and block the host call; copy into {!alloc_pinned} or
    {!register_host} memory when overlap matters. *)
let copy_bytes_to_host ?(stream : stream option) ~(src : (_, _) Vector.t)
    ~(src_byte_offset : int) (dst : (_, _, Bigarray.c_layout) Bigarray.Array1.t)
    : unit =
  d2h_range ~op:"Transfer.copy_bytes_to_host" ?stream ~src ~src_byte_offset dst

(** {1 Pinned host memory} *)

type pinned_kind = Allocated | Registered of Obj.t

type pinned_entry = {
  pin_device : Device.t;
  pin_bytes : int;
  pin_kind : pinned_kind;
}

(* Keyed by start address; an entry matches a Bigarray only when its byte
   length matches too, so a sub-array (even at offset 0) is never mistaken for
   the allocation. *)
let pinned_table : (nativeint, pinned_entry) Hashtbl.t = Hashtbl.create 8

let pinned_lock = Mutex.create ()

let pinned_lookup (ba : (_, _, Bigarray.c_layout) Bigarray.Array1.t) =
  let addr = bigarray_addr ba and bytes = Bigarray.Array1.size_in_bytes ba in
  Mutex.protect pinned_lock (fun () ->
      match Hashtbl.find_opt pinned_table addr with
      | Some e when e.pin_bytes = bytes -> Some (addr, e)
      | _ -> None)

(** [true] iff [ba] is exactly a live {!alloc_pinned} allocation or a live
    {!register_host} registration. *)
let is_pinned (ba : (_, _, Bigarray.c_layout) Bigarray.Array1.t) : bool =
  Option.is_some (pinned_lookup ba)

(** Allocate [n] elements of page-locked host memory for DMA from/to [dev], as a
    Bigarray (read into it with [pread]/[O_DIRECT], then copy it with [~stream]
    at full link rate). HIP: [hipHostMalloc] (portable); CUDA: [cuMemHostAlloc]
    (portable); Metal: page-aligned host memory (unified memory has nothing to
    page-lock); Native/Interpreter: ordinary host memory. Refused with
    [Feature_not_supported] elsewhere.

    The memory is released ONLY by {!free_pinned}: there is no finaliser
    (sub-arrays could outlive the parent). Using the Bigarray, or any sub-array
    of it, after {!free_pinned} is undefined behaviour, as after [munmap].
    [Bigarray.float16] is refused: ctypes cannot build that kind over foreign
    memory; use a [char] or [int16_unsigned] buffer and {!copy_bytes_to_device}.
*)
let alloc_pinned (type a b) (dev : Device.t) (kind : (a, b) Bigarray.kind)
    (n : int) : (a, b, Bigarray.c_layout) Bigarray.Array1.t =
  if n < 1 then invalid_arg "Transfer.alloc_pinned: length must be positive" ;
  (match kind with
  | Bigarray.Float16 ->
      invalid_arg
        "Transfer.alloc_pinned: Float16 is not supported (ctypes cannot view \
         foreign memory as f16); allocate a char buffer instead"
  | _ -> ()) ;
  let (module B : Framework_sig.BACKEND) = backend_of dev in
  let bytes = n * Memory.bigarray_elem_size kind in
  let addr =
    B.Async.alloc_pinned (B.Device.get dev.backend_id) ~byte_size:bytes
  in
  let ba =
    try
      Ctypes.bigarray_of_ptr
        Ctypes.array1
        n
        kind
        (Ctypes.from_voidp
           (Ctypes.typ_of_bigarray_kind kind)
           (Ctypes.ptr_of_raw_address addr))
    with e ->
      (* Do not leak the page-locked allocation if the view cannot be built. *)
      B.Async.free_pinned (B.Device.get dev.backend_id) addr ;
      raise e
  in
  Mutex.protect pinned_lock (fun () ->
      Hashtbl.replace
        pinned_table
        addr
        {pin_device = dev; pin_bytes = bytes; pin_kind = Allocated}) ;
  ba

let refuse_pinned_in_flight ~op addr bytes =
  if Ledger.holds_host_range addr bytes then
    invalid_arg
      (op
     ^ ": the buffer is still referenced by stream-ordered copies; synchronize \
        the stream (or an event recorded after them) first")

(** Release a buffer returned by {!alloc_pinned}. Refused ([Invalid_argument])
    for anything else — a sub-array, a registered buffer, a buffer already freed
    — and while a stream-ordered copy may still read or write it. *)
let free_pinned (ba : (_, _, Bigarray.c_layout) Bigarray.Array1.t) : unit =
  let op = "Transfer.free_pinned" in
  let addr = bigarray_addr ba and bytes = Bigarray.Array1.size_in_bytes ba in
  (* Check, backend free and removal in ONE critical section: two domains cannot
     both free the allocation, and the entry stays (retryable) if the backend
     free raises. The lock order is pinned_lock -> Ledger.lock, never the
     reverse. *)
  Mutex.protect pinned_lock (fun () ->
      match Hashtbl.find_opt pinned_table addr with
      | Some ({pin_kind = Allocated; pin_bytes; _} as e) when pin_bytes = bytes
        ->
          refuse_pinned_in_flight ~op addr bytes ;
          let (module B : Framework_sig.BACKEND) = backend_of e.pin_device in
          B.Async.free_pinned (B.Device.get e.pin_device.backend_id) addr ;
          Hashtbl.remove pinned_table addr
      | Some {pin_kind = Registered _; pin_bytes; _} when pin_bytes = bytes ->
          invalid_arg
            (op ^ ": buffer was registered, use Transfer.unregister_host")
      | _ -> invalid_arg (op ^ ": not a live Transfer.alloc_pinned allocation"))

(** Page-lock an existing Bigarray for DMA from/to [dev]. SPOC keeps [ba] alive
    until {!unregister_host}.

    Measured pitfalls on HIP (RX 7900 XTX, ROCm 7.2, Linux 7.1.2, 2026-10-10,
    under memory pressure from other workloads; specs/transfer-pitfalls.md):

    - Copies right after registering can be slow. bench_hip_register_host, 8 GiB
      registrations, synchronous 5.5 MiB copies: 7 of 50 first passes ran at
      0.95-6.17 GB/s (about 26 GB/s otherwise; one repeat pass at 1.66),
      whatever the preparation: memory written before registering, never
      touched, written after registering, or faulted in with
      madvise(MADV_POPULATE_WRITE) first. spoc_llm saw 0.45 GB/s for memory
      registered before it was touched, and attributed it to that; repeated runs
      here do not support a dependence on touching, only an intermittent,
      state-dependent effect. [hipHostRegister] faults every page in itself
      (VmRSS grew by the full 8 GiB for untouched memory).
    - A copy whose host range spans two registrations fails with
      [hipErrorInvalidValue], synchronous or on a stream. Registering a large
      buffer in regions means slicing every copy inside one region.
    - spoc_llm observed copies from ONE 8 GiB registration fall to 2.3 GB/s
      after CPU writes into it, against 26.3 GB/s with 64 MiB registrations.
      That did not reproduce later the same day (26.3 GB/s after an O_DIRECT
      fill of a single 8 GiB registration): treat region size as something to
      measure on the target machine, not a rule.
    - ROCm does not count registered pages as pinned ([VmPin: 0 kB] in
      [/proc/self/status] with 8 GiB registered), so the kernel may still
      migrate or compact them. Whether that is what makes copies slow under
      memory pressure is NOT established. [alloc_pinned] memory is allocated by
      the driver instead. *)
let register_host (dev : Device.t)
    (ba : (_, _, Bigarray.c_layout) Bigarray.Array1.t) : unit =
  let addr = bigarray_addr ba and bytes = Bigarray.Array1.size_in_bytes ba in
  if bytes < 1 then invalid_arg "Transfer.register_host: empty buffer" ;
  let (module B : Framework_sig.BACKEND) = backend_of dev in
  (* Check, backend registration and insert in one critical section. *)
  Mutex.protect pinned_lock (fun () ->
      if Hashtbl.mem pinned_table addr then
        invalid_arg
          "Transfer.register_host: buffer already pinned or registered" ;
      B.Async.register_host (B.Device.get dev.backend_id) addr ~byte_size:bytes ;
      Hashtbl.replace
        pinned_table
        addr
        {
          pin_device = dev;
          pin_bytes = bytes;
          pin_kind = Registered (Obj.repr ba);
        })

(** Undo {!register_host}; refused while stream-ordered copies may use it. *)
let unregister_host (ba : (_, _, Bigarray.c_layout) Bigarray.Array1.t) : unit =
  let op = "Transfer.unregister_host" in
  let addr = bigarray_addr ba and bytes = Bigarray.Array1.size_in_bytes ba in
  Mutex.protect pinned_lock (fun () ->
      match Hashtbl.find_opt pinned_table addr with
      | Some ({pin_kind = Registered _; pin_bytes; _} as e)
        when pin_bytes = bytes ->
          refuse_pinned_in_flight ~op addr bytes ;
          let (module B : Framework_sig.BACKEND) = backend_of e.pin_device in
          B.Async.unregister_host (B.Device.get e.pin_device.backend_id) addr ;
          Hashtbl.remove pinned_table addr
      | Some {pin_kind = Allocated; pin_bytes; _} when pin_bytes = bytes ->
          invalid_arg
            (op ^ ": buffer was allocated pinned, use Transfer.free_pinned")
      | _ -> invalid_arg (op ^ ": not a registered buffer"))

(** {1 Staged small writes} *)

(** Largest source {!stage_to_device} accepts, in bytes (64 KiB). *)
let max_staged_bytes = 64 * 1024

(* Per device: 1 MiB of pinned memory used as a ring. Each staged copy owns
   its range until the event recorded after it completes. *)
let staging_capacity = 1024 * 1024

type staging = {
  st_buffer :
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
  st_base : nativeint;
  st_ring : event Staging_ring.t;
  mutable st_spare : event list;  (** Completed events, ready to record. *)
  st_lock : Mutex.t;
}

let stagings : (int, staging) Hashtbl.t = Hashtbl.create 4

let stagings_lock = Mutex.create ()

(* Allocated on first use, kept for the process lifetime. *)
let staging_for (dev : Device.t) : staging =
  Mutex.protect stagings_lock (fun () ->
      match Hashtbl.find_opt stagings dev.id with
      | Some st -> st
      | None ->
          let buf = alloc_pinned dev Bigarray.char staging_capacity in
          let st =
            {
              st_buffer = buf;
              st_base = bigarray_addr buf;
              st_ring =
                Staging_ring.create ~capacity:staging_capacity ~align:256;
              st_spare = [];
              st_lock = Mutex.create ();
            }
          in
          Hashtbl.replace stagings dev.id st ;
          st)

(* Snapshot [byte_size] bytes at [src_addr] into the ring, enqueue the copy on
   [stream], and record the event that frees the range. Called under
   [st.st_lock]. *)
let staged_copy st ~(dev : Device.t) ~(stream : stream) ~src_addr ~byte_size
    ~dst ~dst_byte_offset =
  let ring = st.st_ring in
  (* Ranges are reclaimed only when a request does not fit, by synchronising
     the oldest event (immediate when it has completed). *)
  (* The event is taken BEFORE the copy is enqueued: once the copy is on the
     stream, nothing may fail before its range is committed, or a retry could
     reuse the range under the pending DMA. *)
  let ev =
    match st.st_spare with
    | ev :: rest ->
        st.st_spare <- rest ;
        ev
    | [] -> create_event dev
  in
  let give_back () = st.st_spare <- ev :: st.st_spare in
  let off =
    try
      Staging_ring.reserve
        ring
        ~size:byte_size
        ~is_done:(fun _ -> false)
        ~wait:synchronize_event
        ~on_free:(fun e -> st.st_spare <- e :: st.st_spare)
    with e ->
      give_back () ;
      raise e
  in
  (match
     Host_async.blit
       ~src:src_addr
       ~dst:(Nativeint.add st.st_base (Nativeint.of_int off))
       ~byte_size ;
     copy_bytes_to_device
       ~stream
       (Bigarray.Array1.sub st.st_buffer off byte_size)
       ~dst
       ~dst_byte_offset
   with
  | () -> ()
  | exception e ->
      (* Refused before anything was enqueued (h2d_range checks first). *)
      give_back () ;
      raise e) ;
  match record_event ev stream with
  | () -> Staging_ring.commit ring ~off ~size:byte_size ev
  | exception e ->
      (* The copy is in flight but nothing marks its range busy: wait for it
         before the range can be handed out again, and keep [e] whatever the
         wait does. *)
      (try synchronize_stream stream with _ -> ()) ;
      give_back () ;
      raise e

(** Write all of [src] (at most {!max_staged_bytes}) into [dst] at element
    [dst_offset], WITHOUT waiting for the GPU: the cheap way to update a small
    vector that kernels read, e.g. per-layer routing indices.

    [Vector.set] on such a vector is expensive. After a launch the vector is
    [Stale_CPU], so the first [set] downloads it (waiting for every queued
    kernel), and the next launch uploads it again with a synchronous copy that
    waits for the queue once more. Measured (bench_hip_small_writes, RX 7900
    XTX, 8 int32 + 8 float32 rewritten, then 300 us of host work, then 4
    launches, three runs): 709-712 us per iteration with [Vector.set] (339-344
    us of it in the writes), 333-343 us with [stage_to_device] (10-13 us in the
    writes), 320-324 us with no rewrite at all.

    [src] is copied into SPOC-owned pinned staging memory before the call
    returns, so the caller may overwrite it immediately. The copy to [dst] is
    enqueued on [stream] (default: [dev]'s default stream), after the work
    already there and before what follows; streams from {!create_stream} are
    blocking, so they order with it too. SPOC keeps [dst] alive until the copy
    has completed, and refuses to free it until the stream or device is
    synchronised (or the ring has reused the range).

    Staging is 1 MiB of pinned memory per device, allocated on first use and
    kept for the process lifetime; when every byte of it is still in flight, the
    call waits for the oldest staged copy. Same preconditions as
    {!copy_to_device}: [dst] device-current, [Both] becomes [Stale_CPU]; out of
    range, or a source above {!max_staged_bytes}, raises [Invalid_argument]
    before any device call.

    Known issue: on a Ryzen 7950X iGPU (HIP, ROCm 7.2) about 1 run in 7 of 300
    staged writes interleaved with launches lost one launch (its kernel never
    wrote). A caller-managed pinned ring using only copy_to_device and events
    shows it too, so the cause lies below this function. It depends on GC
    timing: running pending finalisers first ([Gc.full_major ()]) removed it in
    every run (0 of 20 against 4 of 20), so finaliser work during the loop is
    the lead; the mechanism is not established. The RX 7900 XTX never showed it
    (specs/transfer-pitfalls.md C-9, issue #427,
    sarek-hip/test/stress_hip_staged_launches.ml).

    HIP, CUDA: as above. Native, Interpreter, Metal (no asynchronous copies): a
    synchronous {!copy_to_device}, complete on return (on Metal it first waits
    for kernels that use [dst]). OpenCL, Vulkan, WebGPU:
    [Feature_not_supported]. *)
let stage_to_device (type a b) ?(stream : stream option)
    (src : (a, b, Bigarray.c_layout) Bigarray.Array1.t) ~(dst : (a, b) Vector.t)
    ~(dst_offset : int) : unit =
  let op = "Transfer.stage_to_device" in
  let n = Bigarray.Array1.dim src in
  if dst_offset < 0 || dst_offset + n > dst.length then
    invalid_arg
      (Printf.sprintf
         "%s: elements [%d, %d) outside vector of length %d"
         op
         dst_offset
         (dst_offset + n)
         dst.length) ;
  let byte_size = Bigarray.Array1.size_in_bytes src in
  if byte_size > max_staged_bytes then
    invalid_arg
      (Printf.sprintf
         "%s: %d bytes exceed max_staged_bytes (%d); use alloc_pinned and \
          copy_to_device for large transfers"
         op
         byte_size
         max_staged_bytes) ;
  let dev, _ = range_target ~op dst in
  let (module B : Framework_sig.BACKEND) = backend_of dev in
  if not B.Async.support.ranged_copy then
    refuse_capability ~backend:B.name "staged writes (they need ranged copies)" ;
  if byte_size = 0 then ()
  else if not B.Async.support.async_copy then
    copy_to_device ?stream src ~dst ~dst_offset
  else
    let stream = match stream with Some s -> s | None -> default_stream dev in
    (* A destroyed stream or one of another device is refused here, before
       staging is allocated or the ring waits on anything (FR-005). *)
    (let (module S : STREAM) = stream in
     Ledger.check_live ~op S.state ;
     if S.device.id <> dev.id then
       invalid_arg
         (Printf.sprintf
            "%s: the stream belongs to device %d, the vector to device %d"
            op
            S.device.id
            dev.id)) ;
    let st = staging_for dev in
    Mutex.protect st.st_lock (fun () ->
        staged_copy
          st
          ~dev
          ~stream
          ~src_addr:(bigarray_addr src)
          ~byte_size
          ~dst
          ~dst_byte_offset:(dst_offset * Vector.elem_size dst.kind)) ;
    ignore (Sys.opaque_identity src)

(** {1 Batch Operations} *)

let to_device_all (vecs : (_, _) Vector.t list) (dev : Device.t) : unit =
  List.iter (fun v -> to_device v dev) vecs

let to_cpu_all (vecs : (_, _) Vector.t list) : unit = List.iter to_cpu vecs

let sync_all (vecs : (_, _) Vector.t list) : unit = List.iter sync vecs

(** {1 Auto-sync Callback Registration} *)

(** Register auto-sync callback with Vector module. The callback respects the
    global auto_mode setting. *)
let () =
  Vector.register_sync_callback
    {
      Vector.sync =
        (fun (type a b) (vec : (a, b) Vector.t) ->
          if not !auto_mode then false
          else begin
            to_cpu vec ;
            true
          end);
    }
