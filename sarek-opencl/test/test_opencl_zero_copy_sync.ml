(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * R-27 regression: host access to a CL_MEM_USE_HOST_PTR (zero-copy) buffer
 * must synchronise with the device.
 *
 * Zero-copy is only chosen for CPU OpenCL devices, and this machine has none,
 * so the test forces it on every OpenCL device through
 * [Opencl_plugin_base.For_testing.zero_copy_on_any_device]. On a GPU the
 * runtime may keep a private copy behind a USE_HOST_PTR buffer, which makes
 * the missing map/unmap deterministic instead of a data race: without it a
 * kernel's result never reaches the host array, and a host write never
 * reaches the kernel.
 *
 * The host-access calls below are the ones Transfer makes (device_to_host_ptr
 * before a host read, host_ptr_to_device after a host write). Skips when no
 * OpenCL device exists.
 ******************************************************************************)

open Sarek_opencl
module Backend = Opencl_plugin_base.Opencl

let n = 256

let write_iota_source =
  {|
__kernel void write_iota(__global float *out) {
    int i = get_local_id(0);
    out[i] = (float)(i + 1);
}
|}

let double_source =
  {|
__kernel void double_it(__global const float *in, __global float *out) {
    int i = get_local_id(0);
    out[i] = in[i] * 2.0f;
}
|}

let addr_of ba =
  Ctypes.raw_address_of_ptr (Spoc_core.Memory.bigarray_void_ptr ba)

let launch device ~name ~source args =
  let compiled = Backend.Kernel.compile_cached device ~name ~source in
  let dims = Spoc_framework.Framework_sig.dims_1d in
  Backend.Kernel.launch
    compiled
    ~args
    ~grid:(dims 1)
    ~block:(dims n)
    ~shared_mem:0
    ~stream:None

(* A float32 array of [n] elements whose address is a multiple of the device's
   CL_DEVICE_MEM_BASE_ADDR_ALIGN plus [skew_elems] elements: aligned when
   [skew_elems = 0], so USE_HOST_PTR is allowed. *)
let host_array ?(skew_elems = 0) ~align fill =
  let align = align in
  let slack = (align / 4) + skew_elems in
  let base =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (n + slack)
  in
  let rec first_aligned k =
    let addr = addr_of (Bigarray.Array1.sub base k 1) in
    if Nativeint.rem addr (Nativeint.of_int align) = 0n then k
    else first_aligned (k + 1)
  in
  let ba = Bigarray.Array1.sub base (first_aligned 0 + skew_elems) n in
  Bigarray.Array1.fill ba fill ;
  ba

let zero_copy_buffer device ba =
  match Backend.Memory.alloc_zero_copy device ba Bigarray.float32 with
  | Some buf -> buf
  | None -> Alcotest.fail "alloc_zero_copy refused despite the test hook"

(* A kernel writes a zero-copy buffer; the host reads the array through the
   device->host path with no explicit synchronize in between. *)
let check_host_read ~align device =
  let ba = host_array ~align (-1.0) in
  let buf = zero_copy_buffer device ba in
  let args = Backend.Kernel.create_args () in
  Backend.Kernel.set_arg_buffer args 0 buf ;
  launch device ~name:"write_iota" ~source:write_iota_source args ;
  Backend.Memory.device_to_host_ptr
    ~src:buf
    ~dst_ptr:(addr_of ba)
    ~byte_size:(n * 4) ;
  for i = 0 to n - 1 do
    Alcotest.(check (float 0.0))
      (Printf.sprintf "host sees kernel result at %d" i)
      (float_of_int (i + 1))
      ba.{i}
  done ;
  Backend.Memory.free buf

(* The host writes the array, publishes it with host_ptr_to_device, a kernel
   reads it through the zero-copy buffer. *)
let check_host_write ~align device =
  let ba = host_array ~align 0.0 in
  let buf = zero_copy_buffer device ba in
  for i = 0 to n - 1 do
    ba.{i} <- float_of_int (i + 1)
  done ;
  Backend.Memory.host_ptr_to_device
    ~src_ptr:(addr_of ba)
    ~byte_size:(n * 4)
    ~dst:buf ;
  let out = Backend.Memory.alloc device n Bigarray.float32 in
  let args = Backend.Kernel.create_args () in
  Backend.Kernel.set_arg_buffer args 0 buf ;
  Backend.Kernel.set_arg_buffer args 1 out ;
  launch device ~name:"double_it" ~source:double_source args ;
  let result = host_array ~align (-1.0) in
  Backend.Memory.device_to_host ~src:out ~dst:result ;
  for i = 0 to n - 1 do
    Alcotest.(check (float 0.0))
      (Printf.sprintf "kernel sees host write at %d" i)
      (2.0 *. float_of_int (i + 1))
      result.{i}
  done ;
  Backend.Memory.free out ;
  Backend.Memory.free buf

(* A host pointer that misses the alignment must not become USE_HOST_PTR: the
   runtime would silently keep a private copy. *)
let check_unaligned_refused ~align device =
  let ba = host_array ~skew_elems:1 ~align 0.0 in
  match Backend.Memory.alloc_zero_copy device ba Bigarray.float32 with
  | None -> ()
  | Some buf ->
      Backend.Memory.free buf ;
      Alcotest.fail "alloc_zero_copy accepted an unaligned host pointer"

let on_every_device check () =
  if not (Backend.is_available ()) then begin
    Printf.printf "[SKIP] No OpenCL device available\n%!" ;
    Alcotest.skip ()
  end
  else begin
    Backend.Device.init () ;
    Opencl_plugin_base.For_testing.zero_copy_on_any_device := true ;
    Fun.protect
      ~finally:(fun () ->
        Opencl_plugin_base.For_testing.zero_copy_on_any_device := false)
      (fun () ->
        for id = 0 to Backend.Device.count () - 1 do
          check
            ~align:
              (Opencl_api.Device.get id).Opencl_api.Device.mem_base_addr_align
            (Backend.Device.get id)
        done)
  end

let () =
  Alcotest.run
    "Opencl_zero_copy_sync"
    [
      ( "zero_copy_host_access",
        [
          Alcotest.test_case
            "host read sees the kernel's writes"
            `Quick
            (on_every_device check_host_read);
          Alcotest.test_case
            "kernel sees the host's writes"
            `Quick
            (on_every_device check_host_write);
          Alcotest.test_case
            "an unaligned host pointer is not aliased"
            `Quick
            (on_every_device check_unaligned_refused);
        ] );
    ]
