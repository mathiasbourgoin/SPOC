(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Pinned and registered host memory (specs/vulkan-async-api.md).

    Two sources of a host range that a [VkBuffer] covers:

    - {!alloc_pinned}: memory the driver allocates ([vkAllocateMemory] from a
      HOST_VISIBLE | HOST_COHERENT type, HOST_CACHED when there is one), mapped
      once for the life of the block. The mapped pointer is the pinned address.
      No host page reaches the driver, so there is no userptr object and no
      kernel invalidation callback (the Sarek #503 path), and no extension is
      needed.
    - {!register}: the pages of a caller's range imported into the driver as
      device memory ([VK_EXT_external_memory_host]). The import covers whole
      pages: the range rounded outwards to the device's
      [minImportedHostPointerAlignment]. Opt-in, and refused on AMD drivers
      ({!Vulkan_host_import_policy}); a device without the extension refuses it.

    A copy between such a range and a device buffer is one [vkCmdCopyBuffer]
    with no staging buffer and no host [memcpy]: the DMA engine reads or writes
    the host pages. A copy whose host range lies inside a buffer's range uses
    it, whether or not it names the allocated or registered bytes exactly. *)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base
module Device = Vulkan_api_device
module Ffi = Vulkan_api_host_ffi

type import = {
  device : Device.t;
  base : nativeint;
      (** First covered byte: the mapped pointer of a driver allocation, or the
          first imported byte (a multiple of the alignment). *)
  length : int;  (** Covered bytes. *)
  buffer : vk_buffer;
  memory : vk_device_memory;
}

(** One buffer per (device, address). [import = None] once the device is gone: a
    [registered] entry then stays as a tombstone so a late {!unregister}
    succeeds quietly. *)
type entry = {mutable import : import option; registered : bool}

let entries : (int * nativeint, entry) Hashtbl.t = Hashtbl.create 8

(** Live {!alloc_pinned} blocks. A block is driver memory owned by the device
    that allocated it: it is unmapped when that device is destroyed, but stays
    listed until {!free_pinned}, which then succeeds without touching it. *)
let owned_blocks : (nativeint, unit) Hashtbl.t = Hashtbl.create 8

let lock = Mutex.create ()

let refuse reason =
  Vulkan_error.raise_error (Vulkan_error.feature_not_supported reason)

let unavailable_reason (device : Device.t) =
  match
    Vulkan_host_import_policy.refusal_reason device.Device.host_import_policy
  with
  | Some reason -> reason
  | None ->
      "registered host memory (the device does not offer \
       VK_EXT_external_memory_host)"

let alignment_of (device : Device.t) =
  match device.Device.host_import_alignment with
  | Some a -> a
  | None -> refuse (unavailable_reason device)

let properties_fn_of (device : Device.t) =
  match device.Device.host_pointer_properties_fn with
  | Some fn -> fn
  | None -> refuse (unavailable_reason device)

let round_down addr alignment =
  Nativeint.sub addr (Nativeint.rem addr (Nativeint.of_int alignment))

let round_up n alignment = (n + alignment - 1) / alignment * alignment

let create_import_buffer (device : Device.t) length =
  let ext = make Ffi.external_buffer_info in
  setf
    ext
    Ffi.ext_buf_sType
    (u32 Ffi.structure_type_external_memory_buffer_create_info) ;
  setf ext Ffi.ext_buf_pNext null ;
  setf ext Ffi.ext_buf_handleTypes (u32 Ffi.handle_type_host_allocation) ;
  let info = make vk_buffer_create_info in
  setf info buf_create_sType (u32 vk_structure_type_buffer_create_info) ;
  setf info buf_create_pNext (to_voidp (addr ext)) ;
  setf info buf_create_flags (u32 0) ;
  setf info buf_create_size (Unsigned.UInt64.of_int length) ;
  setf
    info
    buf_create_usage
    (u32
       (vk_buffer_usage_transfer_src_bit lor vk_buffer_usage_transfer_dst_bit)) ;
  setf info buf_create_sharingMode (u32 0) ;
  setf info buf_create_queueFamilyIndexCount (u32 0) ;
  setf info buf_create_pQueueFamilyIndices (from_voidp uint32_t null) ;
  let buffer = allocate vk_buffer vk_null_handle in
  check
    "vkCreateBuffer (host import)"
    (vkCreateBuffer device.Device.device (addr info) null buffer) ;
  ignore (Sys.opaque_identity ext) ;
  !@buffer

(** Memory type bits both the buffer and the host pointer allow, and the size
    the buffer requires. *)
let import_requirements (device : Device.t) buffer host_ptr =
  let reqs = make vk_memory_requirements in
  vkGetBufferMemoryRequirements device.Device.device buffer (addr reqs) ;
  let props = make Ffi.host_pointer_properties in
  setf
    props
    Ffi.hp_sType
    (u32 Ffi.structure_type_memory_host_pointer_properties) ;
  setf props Ffi.hp_pNext null ;
  setf props Ffi.hp_memoryTypeBits (u32 0) ;
  check
    "vkGetMemoryHostPointerPropertiesEXT"
    (Ffi.get_memory_host_pointer_properties
       (properties_fn_of device)
       device.Device.device
       host_ptr
       (addr props)) ;
  ( Unsigned.UInt32.to_int (getf reqs mem_req_memoryTypeBits)
    land Unsigned.UInt32.to_int (getf props Ffi.hp_memoryTypeBits),
    Unsigned.UInt64.to_int (getf reqs mem_req_size) )

let allocate_import (device : Device.t) ~type_index ~length host_ptr =
  let import_info = make Ffi.import_host_pointer_info in
  setf
    import_info
    Ffi.import_sType
    (u32 Ffi.structure_type_import_memory_host_pointer_info) ;
  setf import_info Ffi.import_pNext null ;
  setf import_info Ffi.import_handleType (u32 Ffi.handle_type_host_allocation) ;
  setf import_info Ffi.import_pHostPointer host_ptr ;
  let info = make vk_memory_allocate_info in
  setf info mem_alloc_sType (u32 vk_structure_type_memory_allocate_info) ;
  setf info mem_alloc_pNext (to_voidp (addr import_info)) ;
  setf info mem_alloc_allocationSize (Unsigned.UInt64.of_int length) ;
  setf info mem_alloc_memoryTypeIndex (u32 type_index) ;
  let memory = allocate vk_device_memory vk_null_handle in
  check
    "vkAllocateMemory (host import)"
    (vkAllocateMemory device.Device.device (addr info) null memory) ;
  ignore (Sys.opaque_identity import_info) ;
  !@memory

let pick_import_type (device : Device.t) bits =
  match
    Vulkan_api_memory.find_memory_type
      device
      bits
      (vk_memory_property_host_visible_bit
     lor vk_memory_property_host_coherent_bit)
  with
  | Some index -> index
  | None ->
      Vulkan_error.raise_error
        (Vulkan_error.context_error
           "host import"
           "no HOST_VISIBLE | HOST_COHERENT memory type accepts this host \
            pointer")

(** Import the pages covering the range from [addr] of [byte_size] bytes on
    [device]. The buffer's own size requirement must fit the rounded range. *)
let import_range (device : Device.t) ~addr ~byte_size =
  let alignment = alignment_of device in
  let base = round_down addr alignment in
  let length =
    round_up (Nativeint.to_int (Nativeint.sub addr base) + byte_size) alignment
  in
  let host_ptr = ptr_of_raw_address base in
  let buffer = create_import_buffer device length in
  let memory = ref vk_null_handle in
  try
    let bits, required = import_requirements device buffer host_ptr in
    if required > length then
      Vulkan_error.raise_error
        (Vulkan_error.context_error
           "host import"
           (Printf.sprintf
              "the buffer needs %d bytes, the imported range has %d"
              required
              length)) ;
    memory :=
      allocate_import
        device
        ~type_index:(pick_import_type device bits)
        ~length
        host_ptr ;
    check
      "vkBindBufferMemory (host import)"
      (vkBindBufferMemory
         device.Device.device
         buffer
         !memory
         (Unsigned.UInt64.of_int 0)) ;
    {device; base; length; buffer; memory = !memory}
  with e ->
    if !memory <> vk_null_handle then
      vkFreeMemory device.Device.device !memory null ;
    vkDestroyBuffer device.Device.device buffer null ;
    raise e

(** Launches and copies that name the import have completed (every copy through
    this module is synchronous), so the objects can go at once. *)
let destroy_import import =
  vkDestroyBuffer import.device.Device.device import.buffer null ;
  vkFreeMemory import.device.Device.device import.memory null

let release_import import =
  !Device.drain_pending import.device ;
  destroy_import import

let take_entry (device : Device.t) addr =
  Mutex.protect lock (fun () ->
      let key = (device.Device.id, addr) in
      let found = Hashtbl.find_opt entries key in
      Hashtbl.remove entries key ;
      found)

let add_entry (device : Device.t) addr entry =
  Mutex.protect lock (fun () ->
      Hashtbl.replace entries (device.Device.id, addr) entry)

let release_entry entry =
  Option.iter release_import entry.import ;
  entry.import <- None

(* The device is being destroyed and has been drained already; draining again
   could raise again and skip the destruction. *)
let destroy_entry entry =
  Option.iter destroy_import entry.import ;
  entry.import <- None

(** The buffer and byte offset that address the range from [addr] of [byte_size]
    bytes of host memory on [device], when an import covers it. *)
let find (device : Device.t) ~addr ~byte_size =
  Mutex.protect lock (fun () ->
      Hashtbl.fold
        (fun _ entry found ->
          match (found, entry.import) with
          | None, Some imp
            when imp.device.Device.id = device.Device.id
                 && imp.base <= addr
                 && Nativeint.add addr (Nativeint.of_int byte_size)
                    <= Nativeint.add imp.base (Nativeint.of_int imp.length) ->
              Some (imp.buffer, Nativeint.to_int (Nativeint.sub addr imp.base))
          | _ -> found)
        entries
        None)

let page_size =
  lazy ((Foreign.foreign "getpagesize" (void @-> returning int)) ())

(** A registered range is imported as the pages around it; the extension may ask
    for a coarser granule than a page, and rounding to it would import memory
    the process may not own. *)
let check_registrable (device : Device.t) =
  let alignment = alignment_of device in
  if alignment > Lazy.force page_size then
    refuse
      (Printf.sprintf
         "register_host (the device imports host memory in %d-byte granules, \
          larger than a page)"
         alignment)

let register (device : Device.t) addr ~byte_size =
  if byte_size <= 0 then
    invalid_arg "Vulkan.Async.register_host: byte_size must be positive" ;
  check_registrable device ;
  let live =
    Mutex.protect lock (fun () ->
        match Hashtbl.find_opt entries (device.Device.id, addr) with
        | Some {import = Some _; _} -> true
        | Some {import = None; _} | None -> false)
  in
  if live then
    invalid_arg "Vulkan.Async.register_host: address already registered" ;
  let import = import_range device ~addr ~byte_size in
  add_entry device addr {import = Some import; registered = true}

let unregister (device : Device.t) addr =
  match take_entry device addr with
  | Some ({registered = true; _} as entry) -> release_entry entry
  | Some entry ->
      add_entry device addr entry ;
      invalid_arg "Vulkan.Async.unregister_host: not a registered range"
  | None -> invalid_arg "Vulkan.Async.unregister_host: not a registered range"

(** The first memory type of [bits] that is HOST_VISIBLE | HOST_COHERENT, with
    [extra] flags, and not DEVICE_LOCAL: on a discrete GPU the DEVICE_LOCAL
    host-visible type is the BAR window onto VRAM, write-combined and read at
    0.03 GB/s. A UMA device has only DEVICE_LOCAL types; those are the last
    resort. *)
let find_host_memory_type (device : Device.t) bits ~extra =
  let required =
    vk_memory_property_host_visible_bit lor vk_memory_property_host_coherent_bit
    lor extra
  in
  match
    Vulkan_api_memory.find_memory_type_excluding
      device
      bits
      ~required
      ~excluded:vk_memory_property_device_local_bit
  with
  | Some _ as found -> found
  | None -> Vulkan_api_memory.find_memory_type device bits required

let pinned_memory_type device bits =
  let cached =
    find_host_memory_type device bits ~extra:vk_memory_property_host_cached_bit
  in
  match cached with
  | Some _ -> cached
  | None -> find_host_memory_type device bits ~extra:0

let create_pinned_buffer (device : Device.t) byte_size =
  let info = make vk_buffer_create_info in
  setf info buf_create_sType (u32 vk_structure_type_buffer_create_info) ;
  setf info buf_create_pNext null ;
  setf info buf_create_flags (u32 0) ;
  setf info buf_create_size (Unsigned.UInt64.of_int byte_size) ;
  setf
    info
    buf_create_usage
    (u32
       (vk_buffer_usage_transfer_src_bit lor vk_buffer_usage_transfer_dst_bit)) ;
  setf info buf_create_sharingMode (u32 0) ;
  setf info buf_create_queueFamilyIndexCount (u32 0) ;
  setf info buf_create_pQueueFamilyIndices (from_voidp uint32_t null) ;
  let buffer = allocate vk_buffer vk_null_handle in
  check
    "vkCreateBuffer (pinned)"
    (vkCreateBuffer device.Device.device (addr info) null buffer) ;
  !@buffer

(** Driver memory for [buffer], bound and mapped. The mapping lasts until the
    memory is freed. *)
let allocate_mapped_memory (device : Device.t) buffer =
  let reqs = make vk_memory_requirements in
  vkGetBufferMemoryRequirements device.Device.device buffer (addr reqs) ;
  let bits = Unsigned.UInt32.to_int (getf reqs mem_req_memoryTypeBits) in
  let type_index =
    match pinned_memory_type device bits with
    | Some index -> index
    | None ->
        Vulkan_error.raise_error
          (Vulkan_error.context_error
             "alloc_pinned"
             "no HOST_VISIBLE | HOST_COHERENT memory type accepts a transfer \
              buffer")
  in
  let info = make vk_memory_allocate_info in
  setf info mem_alloc_sType (u32 vk_structure_type_memory_allocate_info) ;
  setf info mem_alloc_pNext null ;
  setf info mem_alloc_allocationSize (getf reqs mem_req_size) ;
  setf info mem_alloc_memoryTypeIndex (u32 type_index) ;
  let memory = allocate vk_device_memory vk_null_handle in
  check
    "vkAllocateMemory (pinned)"
    (vkAllocateMemory device.Device.device (addr info) null memory) ;
  let memory = !@memory in
  try
    check
      "vkBindBufferMemory (pinned)"
      (vkBindBufferMemory
         device.Device.device
         buffer
         memory
         (Unsigned.UInt64.of_int 0)) ;
    let mapped = allocate (ptr void) null in
    check
      "vkMapMemory (pinned)"
      (vkMapMemory
         device.Device.device
         memory
         (Unsigned.UInt64.of_int 0)
         Unsigned.UInt64.max_int
         (u32 0)
         mapped) ;
    (memory, raw_address_of_ptr !@mapped)
  with e ->
    vkFreeMemory device.Device.device memory null ;
    raise e

(** A driver allocation of [byte_size] bytes, mapped: no host memory crosses to
    the driver, whatever the device or the import policy. *)
let allocate_pinned (device : Device.t) ~byte_size =
  let buffer = create_pinned_buffer device byte_size in
  match allocate_mapped_memory device buffer with
  | memory, base -> {device; base; length = byte_size; buffer; memory}
  | exception e ->
      vkDestroyBuffer device.Device.device buffer null ;
      raise e

(** Driver-allocated, mapped host memory that DMA copies address directly. *)
let alloc_pinned (device : Device.t) ~byte_size =
  if byte_size <= 0 then
    invalid_arg "Vulkan.Async.alloc_pinned: byte_size must be positive" ;
  let import = allocate_pinned device ~byte_size in
  Mutex.protect lock (fun () -> Hashtbl.replace owned_blocks import.base ()) ;
  add_entry device import.base {import = Some import; registered = false} ;
  import.base

(** Releases the block's buffer and memory, which unmaps it. A block whose
    device was destroyed has nothing left to release. *)
let free_pinned (_device : Device.t) addr =
  let owned =
    Mutex.protect lock (fun () ->
        let owned = Hashtbl.mem owned_blocks addr in
        Hashtbl.remove owned_blocks addr ;
        owned)
  in
  if not owned then
    invalid_arg "Vulkan.Async.free_pinned: not a pinned allocation" ;
  let imports =
    Mutex.protect lock (fun () ->
        let found =
          Hashtbl.fold
            (fun key entry acc ->
              if snd key = addr && not entry.registered then (key, entry) :: acc
              else acc)
            entries
            []
        in
        List.iter (fun (key, _) -> Hashtbl.remove entries key) found ;
        found)
  in
  List.iter (fun (_, entry) -> release_entry entry) imports

(* The buffers are children of the VkDevice and go before it. A registered
   range keeps a tombstone (a late [unregister] succeeds); a pinned block is
   unmapped with its device and stays listed, so a late [free_pinned]
   succeeds. *)
let () =
  Spoc_framework.Cache_hooks.on_device_destroy (fun ~backend index ->
      if String.equal backend "Vulkan" then begin
        let doomed =
          Mutex.protect lock (fun () ->
              Hashtbl.fold
                (fun (id, _) entry acc ->
                  if id = index then entry :: acc else acc)
                entries
                [])
        in
        let first_error = ref None in
        List.iter
          (fun entry ->
            try destroy_entry entry
            with e -> if !first_error = None then first_error := Some e)
          doomed ;
        Mutex.protect lock (fun () ->
            Hashtbl.filter_map_inplace
              (fun (id, _) entry ->
                if id = index && not entry.registered then None else Some entry)
              entries) ;
        Option.iter raise !first_error
      end)
