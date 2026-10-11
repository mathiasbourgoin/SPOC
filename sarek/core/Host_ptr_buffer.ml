(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Host-pointer device buffers: the explicit opt-in to share a vector's host
 * memory with a CPU OpenCL device (CL_MEM_USE_HOST_PTR) instead of copying it.
 *
 * Nothing in SPOC creates such a buffer implicitly. A vector gets one only
 * through [attach] or [of_bigarray]; every other vector on an OpenCL device,
 * CPU devices included, has memory of its own.
 *
 * Contract:
 * - Where: CPU-type OpenCL devices. GPU OpenCL devices are refused
 *   ([Host_ptr_not_cpu_device]); CUDA, HIP, Vulkan, Metal and WebGPU refuse
 *   ([Host_ptr_unsupported]). Native and Interpreter accept: their device
 *   memory is the host array already ([Memory.Host_storage]).
 * - Alignment: the array's first byte must be a multiple of the device's
 *   CL_DEVICE_MEM_BASE_ADDR_ALIGN (128 bytes on the Intel CPU runtime, so a
 *   4096-aligned region always qualifies); otherwise [Host_ptr_misaligned].
 *   Without it the runtime would keep a private copy.
 * - Size: at most CL_DEVICE_MAX_MEM_ALLOC_SIZE bytes per buffer
 *   ([Host_ptr_too_large]); a large region is exposed as several vectors over
 *   [Bigarray.Array1.sub] slices, one buffer each.
 * - Lifetime: the vector roots the Bigarray (and a [sub] roots its parent), so
 *   the memory stays alive as long as the vector does. The memory must also
 *   stay in place and mapped: never pass memory that is unmapped or resized
 *   while the vector exists. The buffer is released by [Transfer.free_buffer],
 *   [Transfer.free_all_buffers] or the vector's finaliser; keep the vector
 *   reachable until the launches that use it have completed, as for any
 *   vector.
 * - Synchronisation (OpenCL: host access to a USE_HOST_PTR
 *   region is defined only while it is mapped):
 *   - the array's contents at [attach] are what the first launch sees;
 *   - a kernel's writes are visible in the Bigarray after
 *     [acquire_device_writes] (or any Vector read, [Transfer.to_cpu]), which
 *     maps the buffer for reading: a blocking map that waits for the queued
 *     kernels;
 *   - host writes made directly to the Bigarray after [attach] are visible to
 *     launches enqueued after [publish_host_writes] (map with
 *     WRITE_INVALIDATE_REGION, then unmap); [Vector.set] and
 *     [Transfer.to_device] publish through the same path.
 *   The Intel CPU runtime happens to alias the memory without a map; that is
 *   not part of the contract and SPOC does not rely on it.
 ******************************************************************************)

open Spoc_framework

(** Why a vector could not be given a host-pointer buffer. *)
type error =
  | Refused of {device : string; reason : Framework_sig.host_ptr_refusal}
      (** The backend refused (device type, alignment, size, backend). *)
  | Unknown_framework of {framework : string}
  | Not_bigarray_backed of {vector_id : int}
      (** Custom-type or device-only vector: no Bigarray to share. *)
  | Buffer_exists of {vector_id : int; device : string}
      (** The vector already has a buffer on the device; sharing would leave two
          device copies. Free it first ([Transfer.free_buffer]). *)

let refusal_message : Framework_sig.host_ptr_refusal -> string = function
  | Host_ptr_unsupported -> "backend has no host-pointer buffers"
  | Host_ptr_not_cpu_device -> "not a CPU device"
  | Host_ptr_empty -> "empty array"
  | Host_ptr_misaligned {address; required_alignment} ->
      Printf.sprintf
        "address 0x%nx is not a multiple of %d bytes"
        address
        required_alignment
  | Host_ptr_too_large {byte_size; max_byte_size} ->
      Printf.sprintf
        "%d bytes exceed the device's largest allocation (%d bytes)"
        byte_size
        max_byte_size

let error_message = function
  | Refused {device; reason} ->
      Printf.sprintf
        "Host_ptr_buffer: %s refused: %s"
        device
        (refusal_message reason)
  | Unknown_framework {framework} ->
      "Host_ptr_buffer: unknown framework " ^ framework
  | Not_bigarray_backed {vector_id} ->
      Printf.sprintf
        "Host_ptr_buffer: vector #%d has no Bigarray host storage"
        vector_id
  | Buffer_exists {vector_id; device} ->
      Printf.sprintf
        "Host_ptr_buffer: vector #%d already has a buffer on %s"
        vector_id
        device

let install (type a b) (vec : (a, b) Vector.t) (dev : Device.t)
    (ba : (a, b, Bigarray.c_layout) Bigarray.Array1.t)
    (sk : (a, b) Vector.scalar_kind) : (unit, error) result =
  match Transfer.alloc_scalar_buffer_host_ptr dev ba sk with
  | None -> Error (Unknown_framework {framework = dev.framework})
  | Some (Error reason) -> Error (Refused {device = dev.name; reason})
  | Some (Ok buf) ->
      Transfer.install_buffer vec dev buf ;
      (* The buffer starts with the array's contents: when the host copy is
         the only one, the device copy is now current too. *)
      (match vec.location with
      | Vector.CPU -> vec.location <- Vector.Both dev
      | Vector.GPU _ | Vector.Both _ | Vector.Stale_CPU _ | Vector.Stale_GPU _
        ->
          ()) ;
      Ok ()

(** [attach vec dev] gives [vec] a buffer on [dev] created over its Bigarray's
    memory. See the module header for the lifetime, alignment and
    synchronisation contract. *)
let attach (type a b) (vec : (a, b) Vector.t) (dev : Device.t) :
    (unit, error) result =
  if Vector.has_buffer vec dev then
    Error (Buffer_exists {vector_id = vec.id; device = dev.name})
  else
    match (vec.kind, vec.host) with
    | Vector.Scalar sk, Vector.Bigarray_storage ba -> install vec dev ba sk
    | Vector.Scalar _, Vector.Custom_storage _ -> .
    | Vector.Scalar _, Vector.Device_only_storage | Vector.Custom _, _ ->
        Error (Not_bigarray_backed {vector_id = vec.id})

(** A vector over [ba] (no copy), attached to [dev]. *)
let of_bigarray (dev : Device.t) (sk : ('a, 'b) Vector.scalar_kind)
    (ba : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t) :
    (('a, 'b) Vector.t, error) result =
  let vec = Vector.of_bigarray sk ba in
  Result.map (fun () -> vec) (attach vec dev)

(** How [vec]'s buffer on [dev] relates to its host storage; [None] when it has
    no buffer there. *)
let host_link (vec : (_, _) Vector.t) (dev : Device.t) : Memory.host_link option
    =
  Option.map
    (fun (module B : Vector.DEVICE_BUFFER) -> B.host_link)
    (Vector.get_buffer vec dev)

(** Declares the Bigarray authoritative and makes its contents visible to
    launches on [dev] enqueued afterwards (map/unmap for a host-pointer buffer,
    a copy for any other). Any device-side change not yet read back is
    discarded. *)
let publish_host_writes (vec : (_, _) Vector.t) (dev : Device.t) : unit =
  vec.location <- Vector.Stale_GPU dev ;
  Transfer.to_device vec dev

(** Makes the writes of the launches on [dev] issued so far visible in the
    Bigarray (blocking map for reading; waits for them). No-op when [vec]'s
    device copy is not on [dev]. *)
let acquire_device_writes (vec : (_, _) Vector.t) (dev : Device.t) : unit =
  match vec.location with
  | (Vector.GPU d | Vector.Stale_CPU d | Vector.Both d) when d.id = dev.id ->
      Transfer.to_cpu ~force:true vec
  | Vector.GPU _ | Vector.Stale_CPU _ | Vector.Both _ | Vector.Stale_GPU _
  | Vector.CPU ->
      ()
