(******************************************************************************
 * Sarek Runtime - Transfer Control & Streams
 *
 * Provides explicit and async data transfer API with stream management.
 * Phase 2 of runtime V2 feature parity roadmap.
 *
 * Uses the framework plugin system directly - no Obj.t.
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
      let buf_id = Obj.repr buf |> Obj.obj |> Hashtbl.hash in
      Log.debugf Log.Transfer "alloc_scalar_buffer: buf_id=%d" buf_id ;
      (module struct
        let device = dev

        let size = length

        let elem_size = elem_sz

        let ptr = B.Memory.device_ptr buf

        let bind_to_kernel ~kargs ~idx =
          (* Cast kargs back to the backend's kernel args type and bind buffer *)
          let cur_id = Obj.repr buf |> Obj.obj |> Hashtbl.hash in
          Log.debugf
            Log.Transfer
            "bind_to_kernel: buf_id=%d (was %d)"
            cur_id
            buf_id ;
          Log.debugf
            Log.Transfer
            "bind_to_kernel: idx=%d ptr=%Ld"
            idx
            (Int64.of_nativeint ptr) ;
          let kargs : B.Kernel.args = Obj.obj kargs in
          B.Kernel.set_arg_buffer kargs idx buf

        let from_ptr src_ptr ~byte_size =
          let cur_id = Obj.repr buf |> Obj.obj |> Hashtbl.hash in
          Log.debugf Log.Transfer "from_ptr: buf_id=%d (was %d)" cur_id buf_id ;
          if B.Memory.is_zero_copy buf then () (* Skip for zero-copy *)
          else B.Memory.host_ptr_to_device ~src_ptr ~byte_size ~dst:buf

        let to_ptr dst_ptr ~byte_size =
          if B.Memory.is_zero_copy buf then () (* Skip for zero-copy *)
          else B.Memory.device_to_host_ptr ~src:buf ~dst_ptr ~byte_size

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
          Some
            (module struct
              let device = dev

              let size = length

              let elem_size = elem_sz

              let ptr = B.Memory.device_ptr buf

              let bind_to_kernel ~kargs ~idx =
                Log.debugf
                  Log.Transfer
                  "bind_to_kernel (zero-copy): idx=%d ptr=%Ld"
                  idx
                  (Int64.of_nativeint ptr) ;
                let kargs : B.Kernel.args = Obj.obj kargs in
                B.Kernel.set_arg_buffer kargs idx buf

              let from_ptr _src_ptr ~byte_size:_ = () (* No-op for zero-copy *)

              let to_ptr _dst_ptr ~byte_size:_ = () (* No-op for zero-copy *)

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
      (module struct
        let device = dev

        let size = length

        let elem_size = elem_sz

        let ptr = B.Memory.device_ptr buf

        let bind_to_kernel ~kargs ~idx =
          (* Cast kargs back to the backend's kernel args type and bind buffer *)
          Log.debugf
            Log.Transfer
            "bind_to_kernel: idx=%d ptr=%Ld"
            idx
            (Int64.of_nativeint ptr) ;
          let kargs : B.Kernel.args = Obj.obj kargs in
          B.Kernel.set_arg_buffer kargs idx buf

        let from_ptr src_ptr ~byte_size =
          B.Memory.host_ptr_to_device ~src_ptr ~byte_size ~dst:buf

        let to_ptr dst_ptr ~byte_size =
          B.Memory.device_to_host_ptr ~src:buf ~dst_ptr ~byte_size

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
        | Vector.Custom c, _ ->
            Log.debug Log.Transfer "  -> custom alloc" ;
            alloc_custom_buffer dev vec.length c.elem_size
      in
      Log.debugf
        Log.Transfer
        "ensure_buffer: storing buffer for dev=%d (hashtbl key=%d)"
        dev.id
        dev.id ;
      Hashtbl.replace vec.device_buffers dev.id buf ;
      let (module B : Vector.DEVICE_BUFFER) = buf in
      Log.debugf
        Log.Transfer
        "ensure_buffer: stored buffer ptr=%Ld size=%d"
        (Int64.of_nativeint B.ptr)
        B.size ;
      buf

(** {1 Transfer Operations} *)

(** Convert bigarray to raw pointer and byte size *)
let bigarray_to_ptr (type a b)
    (ba : (a, b, Bigarray.c_layout) Bigarray.Array1.t) (elem_size : int) :
    unit Ctypes.ptr * int =
  let len = Bigarray.Array1.dim ba in
  let byte_size = len * elem_size in
  let ptr = Ctypes.bigarray_start Ctypes.array1 ba in
  (Ctypes.to_voidp ptr, byte_size)

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
  (* Check if already up-to-date on this device *)
  match vec.location with
  | Vector.GPU d when d.id = dev.id -> Log.debug Log.Transfer "-> skip (GPU)"
  | Vector.Both d when d.id = dev.id -> Log.debug Log.Transfer "-> skip (Both)"
  | Vector.Stale_CPU d when d.id = dev.id ->
      Log.debug Log.Transfer "-> skip (Stale_CPU)"
  | _ ->
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
          let ptr, byte_size = bigarray_to_ptr ba B.elem_size in
          Log.debugf
            Log.Transfer
            "to_device: calling from_ptr byte_size=%d"
            byte_size ;
          B.from_ptr ptr ~byte_size
      | Vector.Custom_storage {ptr; custom; length} ->
          B.from_ptr ptr ~byte_size:(length * custom.elem_size)) ;
      vec.location <- Vector.Both dev

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
    match Vector.get_buffer vec dev with
    | None -> failwith "to_cpu: no device buffer to transfer from"
    | Some buf ->
        let (module B : Vector.DEVICE_BUFFER) = buf in
        Log.debugf
          Log.Transfer
          "to_cpu: got buffer ptr=%Ld size=%d"
          (Int64.of_nativeint B.ptr)
          B.size ;
        (match vec.host with
        | Vector.Bigarray_storage ba ->
            let ptr, byte_size = bigarray_to_ptr ba B.elem_size in
            B.to_ptr ptr ~byte_size
        | Vector.Custom_storage {ptr; custom; length} ->
            B.to_ptr ptr ~byte_size:(length * custom.elem_size)) ;
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

(** {1 Buffer Cleanup} *)

(** Free device buffer for a vector *)
let free_buffer (vec : (_, _) Vector.t) (dev : Device.t) : unit =
  match Vector.get_buffer vec dev with
  | None -> ()
  | Some buf -> (
      let (module B : Vector.DEVICE_BUFFER) = buf in
      B.free () ;
      Hashtbl.remove vec.device_buffers dev.id ;
      (* Update location *)
      match vec.location with
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

        let destroy () = () (* Don't destroy default stream *)
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
