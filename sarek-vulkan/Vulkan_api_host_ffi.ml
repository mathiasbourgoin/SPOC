(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** [VK_EXT_external_memory_host]: structures and entry points for importing a
    range of host memory as device memory (specs/vulkan-async-api.md).

    The extension's entry point is not exported by the loader, so it is resolved
    through [vkGetDeviceProcAddr], the way {!Vulkan_bindings} resolves the
    cooperative-matrix query. *)

open Ctypes
open Vulkan_types
open Vulkan_bindings

let extension_name = "VK_EXT_external_memory_host"

let structure_type_external_memory_buffer_create_info = 1000072000

let structure_type_import_memory_host_pointer_info = 1000178000

let structure_type_memory_host_pointer_properties = 1000178001

let structure_type_physical_device_external_memory_host_properties = 1000178002

(** [VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT]. *)
let handle_type_host_allocation = 0x80

let u32 = Unsigned.UInt32.of_int

(** VkExternalMemoryBufferCreateInfo. *)
type external_buffer_info

let external_buffer_info : external_buffer_info structure typ =
  structure "VkExternalMemoryBufferCreateInfo"

let ext_buf_sType = field external_buffer_info "sType" uint32_t

let ext_buf_pNext = field external_buffer_info "pNext" (ptr void)

let ext_buf_handleTypes = field external_buffer_info "handleTypes" uint32_t

let () = seal external_buffer_info

(** VkImportMemoryHostPointerInfoEXT. *)
type import_host_pointer_info

let import_host_pointer_info : import_host_pointer_info structure typ =
  structure "VkImportMemoryHostPointerInfoEXT"

let import_sType = field import_host_pointer_info "sType" uint32_t

let import_pNext = field import_host_pointer_info "pNext" (ptr void)

let import_handleType = field import_host_pointer_info "handleType" uint32_t

let import_pHostPointer =
  field import_host_pointer_info "pHostPointer" (ptr void)

let () = seal import_host_pointer_info

(** VkMemoryHostPointerPropertiesEXT. *)
type host_pointer_properties

let host_pointer_properties : host_pointer_properties structure typ =
  structure "VkMemoryHostPointerPropertiesEXT"

let hp_sType = field host_pointer_properties "sType" uint32_t

let hp_pNext = field host_pointer_properties "pNext" (ptr void)

let hp_memoryTypeBits = field host_pointer_properties "memoryTypeBits" uint32_t

let () = seal host_pointer_properties

(** VkPhysicalDeviceExternalMemoryHostPropertiesEXT. *)
type external_memory_host_properties

let external_memory_host_properties :
    external_memory_host_properties structure typ =
  structure "VkPhysicalDeviceExternalMemoryHostPropertiesEXT"

let emh_sType = field external_memory_host_properties "sType" uint32_t

let emh_pNext = field external_memory_host_properties "pNext" (ptr void)

let emh_minImportedHostPointerAlignment =
  field
    external_memory_host_properties
    "minImportedHostPointerAlignment"
    vk_device_size

let () = seal external_memory_host_properties

(** [minImportedHostPointerAlignment] of [phys_dev]: the alignment required of
    both the address and the size of an imported range. Only meaningful when the
    device advertises {!extension_name}. *)
let query_import_alignment phys_dev =
  let props = make external_memory_host_properties in
  setf
    props
    emh_sType
    (u32 structure_type_physical_device_external_memory_host_properties) ;
  setf props emh_pNext null ;
  setf props emh_minImportedHostPointerAlignment (Unsigned.UInt64.of_int 0) ;
  let props2 = make vk_physical_device_properties_2 in
  setf
    props2
    properties2_sType
    (u32 vk_structure_type_physical_device_properties_2) ;
  setf props2 properties2_pNext (to_voidp (addr props)) ;
  vkGetPhysicalDeviceProperties2 phys_dev (addr props2) ;
  ignore (Sys.opaque_identity props2) ;
  Unsigned.UInt64.to_int (getf props emh_minImportedHostPointerAlignment)

let vkGetDeviceProcAddr_lazy =
  foreign_vk_lazy
    "vkGetDeviceProcAddr"
    (vk_device_ptr @-> string @-> returning (ptr void))

(** [vkGetMemoryHostPointerPropertiesEXT]: the memory types [host_ptr] can be
    imported as. The device must have been created with {!extension_name}
    enabled. *)
let get_memory_host_pointer_properties device host_ptr props =
  let fn =
    Lazy.force
      vkGetDeviceProcAddr_lazy
      device
      "vkGetMemoryHostPointerPropertiesEXT"
  in
  if is_null fn then
    Vulkan_error.raise_error
      (Vulkan_error.feature_not_supported
         "vkGetMemoryHostPointerPropertiesEXT (the loader does not know it)")
  else
    (coerce
       (ptr void)
       (Foreign.funptr
          (vk_device_ptr @-> uint32_t @-> ptr void
          @-> ptr host_pointer_properties
          @-> returning vk_result))
       fn)
      device
      (u32 handle_type_host_allocation)
      host_ptr
      props
