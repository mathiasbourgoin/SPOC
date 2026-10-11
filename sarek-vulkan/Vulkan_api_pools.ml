(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Descriptor pools of one launch-batch slot. A set allocated here stays valid
    until {!reset}, which the owner calls only after the fence of the command
    buffer that bound the sets has been waited for. *)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base
module Device = Vulkan_api_device

type pool = {
  handle : vk_descriptor_pool;
  max_sets : int;
  max_descriptors : int;
  mutable used_sets : int;
  mutable used_descriptors : int;
}

type t = {mutable pools : pool list}

let create () = {pools = []}

let pool_sets = 64

let pool_descriptors = 512

let create_pool device ~max_sets ~max_descriptors =
  let pool_size = make vk_descriptor_pool_size in
  setf pool_size pool_size_type (u32 vk_descriptor_type_storage_buffer) ;
  setf pool_size pool_size_descriptorCount (u32 max_descriptors) ;
  let pool_info = make vk_descriptor_pool_create_info in
  setf
    pool_info
    desc_pool_sType
    (u32 vk_structure_type_descriptor_pool_create_info) ;
  setf pool_info desc_pool_pNext null ;
  setf pool_info desc_pool_flags (u32 0) ;
  setf pool_info desc_pool_maxSets (u32 max_sets) ;
  setf pool_info desc_pool_poolSizeCount (u32 1) ;
  setf pool_info desc_pool_pPoolSizes (addr pool_size) ;
  let handle = allocate vk_descriptor_pool vk_null_handle in
  check
    "vkCreateDescriptorPool"
    (vkCreateDescriptorPool device.Device.device (addr pool_info) null handle) ;
  ignore (Sys.opaque_identity pool_size) ;
  {
    handle = !@handle;
    max_sets;
    max_descriptors;
    used_sets = 0;
    used_descriptors = 0;
  }

let has_room ~descriptors pool =
  pool.used_sets < pool.max_sets
  && pool.used_descriptors + descriptors <= pool.max_descriptors

(** A set of [layout] ([descriptors] storage buffers); another pool is created
    when none has room. *)
let allocate device t layout ~descriptors =
  let pool =
    match List.find_opt (has_room ~descriptors) t.pools with
    | Some pool -> pool
    | None ->
        let pool =
          create_pool
            device
            ~max_sets:pool_sets
            ~max_descriptors:(max pool_descriptors descriptors)
        in
        t.pools <- pool :: t.pools ;
        pool
  in
  let layout_ptr = allocate vk_descriptor_set_layout layout in
  let info = make vk_descriptor_set_allocate_info in
  setf
    info
    desc_set_alloc_sType
    (u32 vk_structure_type_descriptor_set_allocate_info) ;
  setf info desc_set_alloc_pNext null ;
  setf info desc_set_alloc_descriptorPool pool.handle ;
  setf info desc_set_alloc_descriptorSetCount (u32 1) ;
  setf info desc_set_alloc_pSetLayouts layout_ptr ;
  let set = allocate vk_descriptor_set vk_null_handle in
  check
    "vkAllocateDescriptorSets"
    (vkAllocateDescriptorSets device.Device.device (addr info) set) ;
  ignore (Sys.opaque_identity layout_ptr) ;
  ignore (Sys.opaque_identity info) ;
  pool.used_sets <- pool.used_sets + 1 ;
  pool.used_descriptors <- pool.used_descriptors + descriptors ;
  !@set

let reset device t =
  List.iter
    (fun pool ->
      check
        "vkResetDescriptorPool"
        (vkResetDescriptorPool device.Device.device pool.handle (u32 0)) ;
      pool.used_sets <- 0 ;
      pool.used_descriptors <- 0)
    t.pools

let destroy device t =
  List.iter
    (fun pool -> vkDestroyDescriptorPool device.Device.device pool.handle null)
    t.pools ;
  t.pools <- []
