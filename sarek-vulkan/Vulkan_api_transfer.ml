(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Ranged host<->device copies (specs/vulkan-async-api.md).

    Every copy completes before it returns. It first completes the device's
    pending launches ({!Vulkan_api_device.drain_pending}), so it never lands
    under a kernel still using the buffer and always sees what a kernel wrote; a
    copy issued before a launch is visible to it (the launch is recorded after
    the copy has finished).

    Three paths, by where the host bytes and the device bytes live:
    - the device buffer is host-mapped: one [memcpy] at the byte offset;
    - the host range lies inside an import ({!Vulkan_api_host}): one
      [vkCmdCopyBuffer] straight between the import and the device buffer;
    - otherwise: a staging buffer of exactly the range, one [memcpy] and one
      [vkCmdCopyBuffer]. *)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Host = Vulkan_api_host

let total_bytes (buf : _ Memory.buffer) = buf.Memory.size * buf.Memory.elem_size

let advance (p : unit ptr) bytes =
  ptr_of_raw_address
    (Nativeint.add (raw_address_of_ptr p) (Nativeint.of_int bytes))

let record_copy cmd_buf ~src ~src_offset ~dst ~dst_offset ~byte_size =
  let region = make vk_buffer_copy in
  setf region buffer_copy_srcOffset (Unsigned.UInt64.of_int src_offset) ;
  setf region buffer_copy_dstOffset (Unsigned.UInt64.of_int dst_offset) ;
  setf region buffer_copy_size (Unsigned.UInt64.of_int byte_size) ;
  vkCmdCopyBuffer cmd_buf src dst (u32 1) (addr region)

let with_staging device ~byte_size ~usage f =
  let staging_buf, staging_mem, staging_ptr =
    Memory.create_staging_buffer device byte_size usage
  in
  Fun.protect
    ~finally:(fun () ->
      vkDestroyBuffer device.Device.device staging_buf null ;
      vkFreeMemory device.Device.device staging_mem null)
    (fun () -> f staging_buf staging_ptr)

let check_range ~what buf ~off ~byte_size =
  Spoc_core.Host_async.check_range
    ~what
    ~total:(total_bytes buf)
    ~off
    ~byte_size

let host_to_device_staged (dst : _ Memory.buffer) ~src_ptr ~dst_offset
    ~byte_size =
  with_staging
    dst.Memory.device
    ~byte_size
    ~usage:vk_buffer_usage_transfer_src_bit
    (fun staging staging_ptr ->
      let _ = memcpy staging_ptr src_ptr (Unsigned.Size_t.of_int byte_size) in
      Memory.run_single_time_command dst.Memory.device (fun cmd_buf ->
          record_copy
            cmd_buf
            ~src:staging
            ~src_offset:0
            ~dst:dst.Memory.buffer
            ~dst_offset
            ~byte_size))

let device_to_host_staged (src : _ Memory.buffer) ~dst_ptr ~src_offset
    ~byte_size =
  with_staging
    src.Memory.device
    ~byte_size
    ~usage:vk_buffer_usage_transfer_dst_bit
    (fun staging staging_ptr ->
      Memory.run_single_time_command src.Memory.device (fun cmd_buf ->
          record_copy
            cmd_buf
            ~src:src.Memory.buffer
            ~src_offset
            ~dst:staging
            ~dst_offset:0
            ~byte_size) ;
      let _ = memcpy dst_ptr staging_ptr (Unsigned.Size_t.of_int byte_size) in
      ())

(** [byte_size] bytes from raw host address [src_ptr] to [dst_offset] of [dst].
*)
let host_to_device ~src_ptr ~(dst : _ Memory.buffer) ~dst_offset ~byte_size =
  check_range ~what:"Vulkan.Async.host_to_device" dst ~off:dst_offset ~byte_size ;
  if byte_size > 0 then begin
    let device = dst.Memory.device in
    !Device.drain_pending device ;
    let src_ptr_c = ptr_of_raw_address src_ptr in
    match dst.Memory.mapped_ptr with
    | Some mapped ->
        let _ =
          memcpy
            (advance mapped dst_offset)
            src_ptr_c
            (Unsigned.Size_t.of_int byte_size)
        in
        ()
    | None -> (
        match Host.find device ~addr:src_ptr ~byte_size with
        | Some (import, src_offset) ->
            Memory.run_single_time_command device (fun cmd_buf ->
                record_copy
                  cmd_buf
                  ~src:import
                  ~src_offset
                  ~dst:dst.Memory.buffer
                  ~dst_offset
                  ~byte_size)
        | None ->
            host_to_device_staged dst ~src_ptr:src_ptr_c ~dst_offset ~byte_size)
  end

(** [byte_size] bytes from [src_offset] of [src] to raw host address [dst_ptr].
*)
let device_to_host ~(src : _ Memory.buffer) ~src_offset ~dst_ptr ~byte_size =
  check_range ~what:"Vulkan.Async.device_to_host" src ~off:src_offset ~byte_size ;
  if byte_size > 0 then begin
    let device = src.Memory.device in
    !Device.drain_pending device ;
    let dst_ptr_c = ptr_of_raw_address dst_ptr in
    match src.Memory.mapped_ptr with
    | Some mapped ->
        let _ =
          memcpy
            dst_ptr_c
            (advance mapped src_offset)
            (Unsigned.Size_t.of_int byte_size)
        in
        ()
    | None -> (
        match Host.find device ~addr:dst_ptr ~byte_size with
        | Some (import, dst_offset) ->
            Memory.run_single_time_command device (fun cmd_buf ->
                record_copy
                  cmd_buf
                  ~src:src.Memory.buffer
                  ~src_offset
                  ~dst:import
                  ~dst_offset
                  ~byte_size)
        | None ->
            device_to_host_staged src ~dst_ptr:dst_ptr_c ~src_offset ~byte_size)
  end
