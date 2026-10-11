(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - Unified Memory Abstraction
 *
 * Provides a unified interface for GPU memory allocation and data transfer.
 * Uses first-class modules to wrap backend-specific buffers.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** {1 Pure element-size helper} *)

(** Byte size of each Bigarray element kind on a 64-bit target. Replaces
    [Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind)] so that the
    numeric-only path is free of ctypes. Sizes are identical to what ctypes
    sizeof returns on 64-bit: the underlying C representation widths. *)
let bigarray_elem_size : type a b. (a, b) Bigarray.kind -> int = function
  | Bigarray.Float16 -> 2
  | Bigarray.Float32 -> 4
  | Bigarray.Float64 -> 8
  | Bigarray.Int8_signed -> 1
  | Bigarray.Int8_unsigned -> 1
  | Bigarray.Int16_signed -> 2
  | Bigarray.Int16_unsigned -> 2
  | Bigarray.Int32 -> 4
  | Bigarray.Int64 -> 8
  (* OCaml int / nativeint are word-sized: 8 on 64-bit, 4 on 32-bit — matches
     what Ctypes_static.sizeof returned per platform. *)
  | Bigarray.Int -> Sys.word_size / 8
  | Bigarray.Nativeint -> Sys.word_size / 8
  | Bigarray.Complex32 -> 8
  | Bigarray.Complex64 -> 16
  | Bigarray.Char -> 1

(** {1 Host bigarray -> raw pointer} *)

(** Data pointer of a host bigarray, valid only while [ba] is reachable.

    [Ctypes.bigarray_start] cannot be used for f16: ctypes maps a
    [Bigarray.kind] to its own [Ctypes_bigarray.kind] GADT, and that mapping has
    no [Float16] arm (it ends in [failwith "Unsupported bigarray kind"]). For
    f16 the pointer is built over [Ctypes_bigarray.unsafe_address], the
    kind-independent [Caml_ba_data_val]; every other kind keeps the
    [bigarray_start] path.

    GC roots. [bigarray_start] returns a MANAGED fat pointer whose value roots
    the bigarray; ctypes' FFI keeps that root alive across a call. A managed
    pointer with an arbitrary root can only be built through [Obj], so the f16
    pointer here is UNMANAGED: it does not root [ba], and an FFI call holding
    only that pointer can see [ba] freed mid-call. The pointer returned here
    therefore carries an obligation for every kind: the caller keeps [ba] (or
    the vector that owns it) reachable for as long as the pointer, or any
    [nativeint] taken from it, is in use. On transfer paths that is a trailing
    [ignore (Sys.opaque_identity ba)] after the FFI call: [ba] is then live
    across the call, and if the call raises it has already returned from C, so
    no use of the address remains. *)
let borrow_bigarray_ptr : type a b.
    (a, b, Bigarray.c_layout) Bigarray.Array1.t -> unit Ctypes.ptr =
 fun ba ->
  match Bigarray.Array1.kind ba with
  | Bigarray.Float16 ->
      Ctypes.ptr_of_raw_address (Ctypes_bigarray.unsafe_address ba)
  | _ -> Ctypes.(bigarray_start array1 ba |> to_voidp)

(** [with_bigarray_ptr ba f] calls [f] with the data pointer of [ba]. [ba] stays
    reachable until [f] returns, on every element kind, so a synchronous FFI
    call made inside [f] may read or write through the pointer. The pointer must
    not be used after [f] returns: an asynchronous consumer of the address needs
    its own root. A capturing [f] costs a closure allocation per call, so
    transfer paths use {!borrow_bigarray_ptr} with a trailing
    [Sys.opaque_identity] instead; this helper is for cold paths such as buffer
    allocation. *)
let with_bigarray_ptr : type a b r.
    (a, b, Bigarray.c_layout) Bigarray.Array1.t -> (unit Ctypes.ptr -> r) -> r =
 fun ba f ->
  let result = f (borrow_bigarray_ptr ba) in
  ignore (Sys.opaque_identity ba) ;
  result

(** {1 Buffer Module Type} *)

(** A buffer packages backend-specific buffer with its operations. All transfers
    use raw pointers with byte sizes to avoid type parameter escaping issues in
    first-class modules. *)
module type BUFFER = sig
  (** The device this buffer is allocated on *)
  val device : Device.t

  (** Number of elements *)
  val size : int

  (** Size of each element in bytes *)
  val elem_size : int

  (** Get raw device pointer (for kernel arg binding) *)
  val device_ptr : nativeint

  (** Transfer from host pointer to device *)
  val host_ptr_to_device : unit Ctypes.ptr -> byte_size:int -> unit

  (** Transfer from device to host pointer *)
  val device_to_host_ptr : unit Ctypes.ptr -> byte_size:int -> unit

  (** Bind this buffer to kernel args at given index *)
  val bind_to_kargs : Framework_sig.kargs -> int -> unit

  (** Ranged host->device copy into this buffer at [dst_offset] bytes.
      [~stream:None] is synchronous; [Some r] enqueues on the stream [r] unwraps
      to (refused if it belongs to another backend). [src_ptr] is not a GC root:
      the caller keeps it alive until the copy has completed. Backends without
      ranged copies raise [Feature_not_supported]. *)
  val range_from_host :
    src_ptr:nativeint ->
    dst_offset:int ->
    byte_size:int ->
    stream:Framework_sig.stream_ref option ->
    unit

  (** Ranged device->host copy out of this buffer, same contract. *)
  val range_to_host :
    src_offset:int ->
    dst_ptr:nativeint ->
    byte_size:int ->
    stream:Framework_sig.stream_ref option ->
    unit

  (** Free the buffer *)
  val free : unit -> unit
end

(** Unwrap a core stream handle for backend [backend]; a handle from another
    backend is a caller error, refused rather than ignored. *)
let unwrap_stream_for ~(backend : string)
    (unwrap : Framework_sig.stream_ref -> 's option)
    (r : Framework_sig.stream_ref option) : 's option =
  match r with
  | None -> None
  | Some r -> (
      match unwrap r with
      | Some s -> Some s
      | None ->
          invalid_arg
            (Printf.sprintf
               "stream belongs to another backend than %s (the buffer's)"
               backend))

(** Buffer with phantom type parameter for element type safety. The 'a parameter
    is not used at runtime but ensures type-safe transfers. *)
type _ buffer = Buffer : (module BUFFER) -> 'a buffer

(** {1 Allocation} *)

(** Allocate a buffer on a device for standard Bigarray types *)
let alloc (device : Device.t) (size : int) (kind : ('a, 'b) Bigarray.kind) :
    'a buffer =
  match Framework_registry.find_backend device.framework with
  | None -> failwith ("Unknown framework: " ^ device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let buf = B.Memory.alloc dev size kind in
      let elem_size = bigarray_elem_size kind in
      Buffer
        (module struct
          let device = device

          let size = size

          let elem_size = elem_size

          let device_ptr = B.Memory.device_ptr buf

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

          let bind_to_kargs kargs idx =
            match B.unwrap_kargs kargs with
            | Some args -> B.Kernel.set_arg_buffer args idx buf
            | None -> failwith "bind_to_kargs: backend mismatch"

          let range_from_host ~src_ptr ~dst_offset ~byte_size ~stream =
            B.Async.host_to_device
              ~src_ptr
              ~dst:buf
              ~dst_offset
              ~byte_size
              ~stream:(unwrap_stream_for ~backend:B.name B.unwrap_stream stream)

          let range_to_host ~src_offset ~dst_ptr ~byte_size ~stream =
            B.Async.device_to_host
              ~src:buf
              ~src_offset
              ~dst_ptr
              ~byte_size
              ~stream:(unwrap_stream_for ~backend:B.name B.unwrap_stream stream)

          let free () = B.Memory.free buf
        end : BUFFER)

(** Allocate a buffer for custom types with explicit element size in bytes *)
let alloc_custom (device : Device.t) ~(size : int) ~(elem_size : int) :
    'a buffer =
  match Framework_registry.find_backend device.framework with
  | None -> failwith ("Unknown framework: " ^ device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let buf = B.Memory.alloc_custom dev ~size ~elem_size in
      Buffer
        (module struct
          let device = device

          let size = size

          let elem_size = elem_size

          let device_ptr = B.Memory.device_ptr buf

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

          let bind_to_kargs kargs idx =
            match B.unwrap_kargs kargs with
            | Some args -> B.Kernel.set_arg_buffer args idx buf
            | None -> failwith "bind_to_kargs: backend mismatch"

          let range_from_host ~src_ptr ~dst_offset ~byte_size ~stream =
            B.Async.host_to_device
              ~src_ptr
              ~dst:buf
              ~dst_offset
              ~byte_size
              ~stream:(unwrap_stream_for ~backend:B.name B.unwrap_stream stream)

          let range_to_host ~src_offset ~dst_ptr ~byte_size ~stream =
            B.Async.device_to_host
              ~src:buf
              ~src_offset
              ~dst_ptr
              ~byte_size
              ~stream:(unwrap_stream_for ~backend:B.name B.unwrap_stream stream)

          let free () = B.Memory.free buf
        end : BUFFER)

(** {1 Buffer Operations} *)

(** Free a buffer *)
let free : type a. a buffer -> unit = fun (Buffer (module B)) -> B.free ()

(** Size a bigarray<->buffer copy from the bigarray's own bytes. ['a buffer]
    only fixes the OCaml element type, so a float32 bigarray type-checks against
    a float64 buffer; sizing the copy as [dim x buffer elem_size] then reads or
    writes past the bigarray. Refused ([Invalid_argument]) when the element
    sizes differ or the bigarray holds more bytes than the buffer. *)
let checked_host_copy_bytes : type a b.
    op:string ->
    ba:(a, b, Bigarray.c_layout) Bigarray.Array1.t ->
    buffer_elem_size:int ->
    buffer_size:int ->
    int =
 fun ~op ~ba ~buffer_elem_size ~buffer_size ->
  let ba_elem_size = bigarray_elem_size (Bigarray.Array1.kind ba) in
  let ba_bytes = Bigarray.Array1.size_in_bytes ba in
  let buffer_bytes = buffer_size * buffer_elem_size in
  if ba_elem_size <> buffer_elem_size then
    invalid_arg
      (Printf.sprintf
         "%s: bigarray element size (%d bytes) differs from the buffer's (%d \
          bytes)"
         op
         ba_elem_size
         buffer_elem_size)
  else if ba_bytes > buffer_bytes then
    invalid_arg
      (Printf.sprintf
         "%s: bigarray (%d bytes) does not fit the buffer (%d bytes)"
         op
         ba_bytes
         buffer_bytes)
  else ba_bytes

(** Copy data from host bigarray to device. Converts bigarray to pointer
    internally. The type parameter fixes only the OCaml element type; the copy
    is sized from the bigarray's bytes and [Invalid_argument] is raised if its
    element size differs from the buffer's or it exceeds the buffer. *)
let host_to_device : type a b.
    src:(a, b, Bigarray.c_layout) Bigarray.Array1.t -> dst:a buffer -> unit =
 fun ~src ~dst ->
  let (Buffer (module B)) = dst in
  let byte_size =
    checked_host_copy_bytes
      ~op:"host_to_device"
      ~ba:src
      ~buffer_elem_size:B.elem_size
      ~buffer_size:B.size
  in
  (* Neither the borrowed pointer nor the bare [nativeint] that
     [B.host_ptr_to_device] takes from it roots [src] (see [host_ptr_to_device]
     in Framework_sig.ml); the trailing [Sys.opaque_identity] keeps [src]
     reachable until the transfer has consumed the address. *)
  B.host_ptr_to_device (borrow_bigarray_ptr src) ~byte_size ;
  ignore (Sys.opaque_identity src)

(** Copy data from device to host bigarray. Converts bigarray to pointer
    internally. The type parameter fixes only the OCaml element type; the copy
    is sized from the bigarray's bytes and [Invalid_argument] is raised if its
    element size differs from the buffer's or it exceeds the buffer. *)
let device_to_host : type a b.
    src:a buffer -> dst:(a, b, Bigarray.c_layout) Bigarray.Array1.t -> unit =
 fun ~src ~dst ->
  let (Buffer (module B)) = src in
  let byte_size =
    checked_host_copy_bytes
      ~op:"device_to_host"
      ~ba:dst
      ~buffer_elem_size:B.elem_size
      ~buffer_size:B.size
  in
  B.device_to_host_ptr (borrow_bigarray_ptr dst) ~byte_size ;
  ignore (Sys.opaque_identity dst)

(** Copy data from raw pointer to device buffer (for custom types) *)
let host_ptr_to_device : type a. src_ptr:unit Ctypes.ptr -> dst:a buffer -> unit
    =
 fun ~src_ptr ~dst ->
  let (Buffer (module B)) = dst in
  let byte_size = B.size * B.elem_size in
  B.host_ptr_to_device src_ptr ~byte_size

(** Copy data from device buffer to raw pointer (for custom types) *)
let device_to_host_ptr : type a. src:a buffer -> dst_ptr:unit Ctypes.ptr -> unit
    =
 fun ~src ~dst_ptr ->
  let (Buffer (module B)) = src in
  let byte_size = B.size * B.elem_size in
  B.device_to_host_ptr dst_ptr ~byte_size

(** Copy data between device buffers (same device). The type parameter fixes
    only the OCaml element type; [Invalid_argument] is raised if the element
    sizes differ or the source holds more bytes than the destination. *)
let device_to_device : type a. src:a buffer -> dst:a buffer -> unit =
 fun ~src ~dst ->
  let (Buffer (module Src)) = src in
  let (Buffer (module Dst)) = dst in
  if Src.device.id <> Dst.device.id then
    failwith "device_to_device requires buffers on same device"
  else begin
    let byte_size = Src.size * Src.elem_size in
    let dst_bytes = Dst.size * Dst.elem_size in
    if Src.elem_size <> Dst.elem_size then
      invalid_arg
        (Printf.sprintf
           "device_to_device: element size differs (source %d bytes, \
            destination %d bytes)"
           Src.elem_size
           Dst.elem_size) ;
    if byte_size > dst_bytes then
      invalid_arg
        (Printf.sprintf
           "device_to_device: source (%d bytes) does not fit the destination \
            (%d bytes)"
           byte_size
           dst_bytes) ;
    (* Transfer via host. BUFFER has no backend D2D copy, so the bounce stays. *)
    let tmp = Ctypes.(allocate_n uint8_t ~count:byte_size) in
    let tmp_ptr = Ctypes.to_voidp tmp in
    Src.device_to_host_ptr tmp_ptr ~byte_size ;
    Dst.host_ptr_to_device tmp_ptr ~byte_size ;
    (* The backends take a bare nativeint: keep [tmp] alive past the copy. *)
    ignore (Sys.opaque_identity tmp)
  end

(** {1 Accessors} *)

(** Get buffer size in elements *)
let size : type a. a buffer -> int = fun (Buffer (module B)) -> B.size

(** Get buffer element size in bytes *)
let elem_size : type a. a buffer -> int = fun (Buffer (module B)) -> B.elem_size

(** Get buffer device *)
let device : type a. a buffer -> Device.t = fun (Buffer (module B)) -> B.device

(** Get raw device pointer *)
let device_ptr : type a. a buffer -> nativeint =
 fun (Buffer (module B)) -> B.device_ptr

(** Bind buffer to kernel args *)
let bind_to_kargs : type a. a buffer -> Framework_sig.kargs -> int -> unit =
 fun (Buffer (module B)) kargs idx -> B.bind_to_kargs kargs idx
