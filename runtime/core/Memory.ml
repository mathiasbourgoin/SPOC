(******************************************************************************
 * Sarek Runtime - Unified Memory Abstraction
 *
 * Provides a unified interface for GPU memory allocation and data transfer.
 * Wraps backend-specific buffers with a common type.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** Unified buffer type - wraps backend-specific buffers *)
type 'a buffer = {
  device : Device.t;
  size : int;  (** Number of elements *)
  elem_size : int;  (** Size of each element in bytes *)
  handle : Obj.t;  (** Backend-specific buffer handle *)
}

(** Allocate a buffer on a device for standard Bigarray types *)
let alloc (device : Device.t) (size : int) (kind : ('a, 'b) Bigarray.kind) :
    'a buffer =
  match Framework_registry.find_backend device.framework with
  | None -> failwith ("Unknown framework: " ^ device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let buf = B.Memory.alloc dev size kind in
      let elem_size = Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind) in
      {device; size; elem_size; handle = Obj.repr buf}

(** Allocate a buffer for custom types with explicit element size in bytes *)
let alloc_custom (device : Device.t) ~(size : int) ~(elem_size : int) :
    'a buffer =
  match Framework_registry.find_backend device.framework with
  | None -> failwith ("Unknown framework: " ^ device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let buf = B.Memory.alloc_custom dev ~size ~elem_size in
      {device; size; elem_size; handle = Obj.repr buf}

(** Free a buffer *)
let free (buf : 'a buffer) : unit =
  match Framework_registry.find_backend buf.device.framework with
  | None -> ()
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_buf : 'a B.Memory.buffer = Obj.obj buf.handle in
      B.Memory.free backend_buf

(** Copy data from host to device *)
let host_to_device ~(src : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t)
    ~(dst : 'a buffer) : unit =
  match Framework_registry.find_backend dst.device.framework with
  | None -> failwith ("Unknown framework: " ^ dst.device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_buf : 'a B.Memory.buffer = Obj.obj dst.handle in
      B.Memory.host_to_device ~src ~dst:backend_buf

(** Copy data from device to host *)
let device_to_host ~(src : 'a buffer)
    ~(dst : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t) : unit =
  match Framework_registry.find_backend src.device.framework with
  | None -> failwith ("Unknown framework: " ^ src.device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_buf : 'a B.Memory.buffer = Obj.obj src.handle in
      B.Memory.device_to_host ~src:backend_buf ~dst

(** Copy data from raw pointer to device buffer (for custom types) *)
let host_ptr_to_device ~(src_ptr : unit Ctypes.ptr) ~(dst : 'a buffer) : unit =
  let byte_size = dst.size * dst.elem_size in
  match Framework_registry.find_backend dst.device.framework with
  | None -> failwith ("Unknown framework: " ^ dst.device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_buf : 'a B.Memory.buffer = Obj.obj dst.handle in
      B.Memory.host_ptr_to_device ~src_ptr ~byte_size ~dst:backend_buf

(** Copy data from device buffer to raw pointer (for custom types) *)
let device_to_host_ptr ~(src : 'a buffer) ~(dst_ptr : unit Ctypes.ptr) : unit =
  let byte_size = src.size * src.elem_size in
  match Framework_registry.find_backend src.device.framework with
  | None -> failwith ("Unknown framework: " ^ src.device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_buf : 'a B.Memory.buffer = Obj.obj src.handle in
      B.Memory.device_to_host_ptr ~src:backend_buf ~dst_ptr ~byte_size

(** Copy data between device buffers (same device) *)
let device_to_device ~(src : 'a buffer) ~(dst : 'a buffer) : unit =
  if src.device.id <> dst.device.id then
    failwith "device_to_device requires buffers on same device"
  else
    match Framework_registry.find_backend src.device.framework with
    | None -> failwith ("Unknown framework: " ^ src.device.framework)
    | Some (module B : Framework_sig.BACKEND) ->
        let src_buf : 'a B.Memory.buffer = Obj.obj src.handle in
        let dst_buf : 'a B.Memory.buffer = Obj.obj dst.handle in
        B.Memory.device_to_device ~src:src_buf ~dst:dst_buf

(** Get buffer size in elements *)
let size buf = buf.size

(** Get buffer device *)
let device buf = buf.device
