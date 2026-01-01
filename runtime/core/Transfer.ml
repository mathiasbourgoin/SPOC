(******************************************************************************
 * Sarek Runtime - Transfer Control & Streams
 *
 * Provides explicit and async data transfer API with stream management.
 * Phase 2 of runtime V2 feature parity roadmap.
 *
 * Uses the framework plugin system directly - no Obj.t.
 ******************************************************************************)

open Sarek_framework

(** {1 Auto-Transfer Mode} *)

let auto_mode = ref true

let enable_auto () = auto_mode := true
let disable_auto () = auto_mode := false
let is_auto () = !auto_mode
let set_auto enabled = auto_mode := enabled

(** {1 Device Buffer Allocation} *)

(** Allocate a device buffer for a scalar vector, returning a DEVICE_BUFFER module.
    The buffer is packaged with its backend for type-safe operations. *)
let alloc_scalar_buffer (type a b) (dev : Device.t) (length : int)
    (sk : Vector.scalar_kind) : Vector.device_buffer =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get dev.backend_id in
      let ba_kind = Vector.to_bigarray_kind sk in
      let buf = B.Memory.alloc backend_dev length ba_kind in
      let elem_sz = Vector.scalar_elem_size sk in
      (module struct
        let device = dev
        let size = length
        let elem_size = elem_sz
        let ptr = B.Memory.device_ptr buf

        let from_bigarray ba = B.Memory.host_to_device ~src:ba ~dst:buf
        let to_bigarray ba = B.Memory.device_to_host ~src:buf ~dst:ba
        let from_ptr src_ptr ~byte_size =
          B.Memory.host_ptr_to_device ~src_ptr ~byte_size ~dst:buf
        let to_ptr dst_ptr ~byte_size =
          B.Memory.device_to_host_ptr ~src:buf ~dst_ptr ~byte_size
        let free () = B.Memory.free buf
      end : Vector.DEVICE_BUFFER)

(** Allocate a device buffer for a custom vector *)
let alloc_custom_buffer (dev : Device.t) (length : int) (elem_sz : int) :
    Vector.device_buffer =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get dev.backend_id in
      let buf = B.Memory.alloc_custom backend_dev ~size:length ~elem_size:elem_sz in
      (module struct
        let device = dev
        let size = length
        let elem_size = elem_sz
        let ptr = B.Memory.device_ptr buf

        let from_bigarray _ba =
          failwith "from_bigarray: not supported for custom buffers"
        let to_bigarray _ba =
          failwith "to_bigarray: not supported for custom buffers"
        let from_ptr src_ptr ~byte_size =
          B.Memory.host_ptr_to_device ~src_ptr ~byte_size ~dst:buf
        let to_ptr dst_ptr ~byte_size =
          B.Memory.device_to_host_ptr ~src:buf ~dst_ptr ~byte_size
        let free () = B.Memory.free buf
      end : Vector.DEVICE_BUFFER)

(** {1 Buffer Management for Vectors} *)

(** Ensure vector has a device buffer, allocating if needed *)
let ensure_buffer (type a b) (vec : (a, b) Vector.t) (dev : Device.t) :
    Vector.device_buffer =
  match Vector.get_buffer vec dev with
  | Some buf -> buf
  | None ->
      let buf =
        match vec.kind with
        | Vector.Scalar sk -> alloc_scalar_buffer dev vec.length sk
        | Vector.Custom c -> alloc_custom_buffer dev vec.length c.elem_size
      in
      Hashtbl.replace vec.device_buffers dev.id buf ;
      buf

(** {1 Transfer Operations} *)

(** Transfer vector data to a device *)
let to_device (type a b) (vec : (a, b) Vector.t) (dev : Device.t) : unit =
  (* Check if already up-to-date on this device *)
  match vec.location with
  | Vector.GPU d when d.id = dev.id -> ()
  | Vector.Both d when d.id = dev.id -> ()
  | Vector.Stale_CPU d when d.id = dev.id -> ()  (* GPU already authoritative *)
  | _ ->
      (* Ensure buffer exists and transfer *)
      let buf = ensure_buffer vec dev in
      let (module B : Vector.DEVICE_BUFFER) = buf in
      (match vec.host with
      | Vector.Bigarray_storage ba -> B.from_bigarray ba
      | Vector.Custom_storage {ptr; custom; length} ->
          B.from_ptr ptr ~byte_size:(length * custom.elem_size)) ;
      vec.location <- Vector.Both dev

(** Transfer vector data from device to CPU *)
let to_cpu (type a b) (vec : (a, b) Vector.t) : unit =
  match vec.location with
  | Vector.CPU -> ()  (* Already on CPU only *)
  | Vector.Both _ -> ()  (* Already synced *)
  | Vector.Stale_GPU _ -> ()  (* CPU already authoritative *)
  | Vector.GPU dev | Vector.Stale_CPU dev ->
      (* Transfer from GPU to CPU *)
      (match Vector.get_buffer vec dev with
      | None -> failwith "to_cpu: no device buffer to transfer from"
      | Some buf ->
          let (module B : Vector.DEVICE_BUFFER) = buf in
          (match vec.host with
          | Vector.Bigarray_storage ba -> B.to_bigarray ba
          | Vector.Custom_storage {ptr; custom; length} ->
              B.to_ptr ptr ~byte_size:(length * custom.elem_size))) ;
      vec.location <-
        (match vec.location with
        | Vector.GPU dev -> Vector.Both dev
        | Vector.Stale_CPU dev -> Vector.Both dev
        | other -> other)

(** Ensure vector is fully synchronized *)
let sync (type a b) (vec : (a, b) Vector.t) : unit =
  match vec.location with
  | Vector.CPU -> ()
  | Vector.Both _ -> ()
  | Vector.GPU dev -> to_cpu vec ; vec.location <- Vector.Both dev
  | Vector.Stale_CPU dev -> to_cpu vec ; vec.location <- Vector.Both dev
  | Vector.Stale_GPU dev -> to_device vec dev ; vec.location <- Vector.Both dev

(** {1 Buffer Cleanup} *)

(** Free device buffer for a vector *)
let free_buffer (vec : (_, _) Vector.t) (dev : Device.t) : unit =
  match Vector.get_buffer vec dev with
  | None -> ()
  | Some buf ->
      let (module B : Vector.DEVICE_BUFFER) = buf in
      B.free () ;
      Hashtbl.remove vec.device_buffers dev.id ;
      (* Update location *)
      (match vec.location with
      | Vector.GPU d when d.id = dev.id -> vec.location <- Vector.CPU
      | Vector.Both d when d.id = dev.id -> vec.location <- Vector.CPU
      | Vector.Stale_CPU d when d.id = dev.id -> vec.location <- Vector.CPU
      | Vector.Stale_GPU d when d.id = dev.id -> vec.location <- Vector.CPU
      | _ -> ())

(** Free all device buffers for a vector *)
let free_all_buffers (vec : (_, _) Vector.t) : unit =
  Hashtbl.iter
    (fun _ buf ->
      let (module B : Vector.DEVICE_BUFFER) = buf in
      B.free ())
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
      B.Device.synchronize backend_dev

(** {1 Stream Operations} *)

(** Stream handle - packages backend stream with its operations *)
module type STREAM = sig
  val device : Device.t
  val synchronize : unit -> unit
  val destroy : unit -> unit
end

type stream = (module STREAM)

(** Create a new stream on a device *)
let create_stream (dev : Device.t) : stream =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get dev.backend_id in
      let s = B.Stream.create backend_dev in
      (module struct
        let device = dev
        let synchronize () = B.Stream.synchronize s
        let destroy () = B.Stream.destroy s
      end : STREAM)

(** Get default stream for a device *)
let default_stream (dev : Device.t) : stream =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get dev.backend_id in
      let s = B.Stream.default backend_dev in
      (module struct
        let device = dev
        let synchronize () = B.Stream.synchronize s
        let destroy () = ()  (* Don't destroy default stream *)
      end : STREAM)

let synchronize_stream (s : stream) =
  let (module S : STREAM) = s in
  S.synchronize ()

let destroy_stream (s : stream) =
  let (module S : STREAM) = s in
  S.destroy ()

(** {1 Batch Operations} *)

let to_device_all (vecs : (_, _) Vector.t list) (dev : Device.t) : unit =
  List.iter (fun v -> to_device v dev) vecs

let to_cpu_all (vecs : (_, _) Vector.t list) : unit =
  List.iter to_cpu vecs

let sync_all (vecs : (_, _) Vector.t list) : unit =
  List.iter sync vecs
