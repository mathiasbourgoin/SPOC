(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * R-27 through the user path: Vector, Transfer and Execute.run_vectors on an
 * OpenCL device whose buffers are CL_MEM_USE_HOST_PTR.
 *
 * Zero-copy is normally chosen for CPU devices only; the test forces it on
 * every OpenCL device with
 * [Opencl_plugin_base.For_testing.zero_copy_on_any_device] (test-only) and
 * allocates vectors until one meets the device's CL_DEVICE_MEM_BASE_ADDR_ALIGN.
 * It then checks that Vector.get sees a kernel's writes, and that a kernel
 * sees a Vector.set made after the first launch. Without the host-copy
 * forwarding in Transfer's zero-copy wrapper both fail on a runtime that
 * keeps a private copy (rusticl). Prints SKIP when there is no OpenCL device.
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

let () = Test_helpers.Benchmarks.init_backends ()

let iota =
  [%kernel
    fun (y : float32 vector) (n : int32) ->
      let i = global_thread_id in
      if i < n then y.(i) <- float (i + 1l)]

let dbl =
  [%kernel
    fun (x : float32 vector) (y : float32 vector) (n : int32) ->
      let i = global_thread_id in
      if i < n then y.(i) <- x.(i) *. 2.0]

let ir_of (_, k) =
  match k.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let n = 256

let addr v =
  Ctypes.raw_address_of_ptr
    (Spoc_core.Memory.borrow_bigarray_ptr (Vector.to_bigarray v))

let is_aligned align v = Nativeint.rem (addr v) (Nativeint.of_int align) = 0n

let rec aligned_vec align tries =
  let v = Vector.create Vector.float32 n in
  if is_aligned align v then v
  else if tries > 100_000 then failwith "no aligned vector found"
  else aligned_vec align (tries + 1)

let failures = ref 0

let check what got want =
  if got <> want then begin
    incr failures ;
    if !failures <= 10 then
      Printf.printf "  FAIL %s: got %g want %g\n%!" what got want
  end

let run (dev : Device.t) =
  let align =
    (Sarek_opencl.Opencl_api.Device.get dev.Device.backend_id)
      .Sarek_opencl.Opencl_api.Device.mem_base_addr_align
  in
  let launch ir args =
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args
      ~block:(Execute.dims1d 256)
      ~grid:(Execute.dims1d 1)
      ()
  in
  let y = aligned_vec align 0 in
  for i = 0 to n - 1 do
    Vector.set y i (-1.0)
  done ;
  launch (ir_of iota) Execute.[Vec y; Int n] ;
  for i = 0 to n - 1 do
    check (Printf.sprintf "read y[%d]" i) (Vector.get y i) (float (i + 1))
  done ;
  Vector.set y 0 100.0 ;
  let out = aligned_vec align 0 in
  launch (ir_of dbl) Execute.[Vec y; Vec out; Int n] ;
  check "kernel sees host set" (Vector.get out 0) 200.0 ;
  check "kernel sees y[1]" (Vector.get out 1) 4.0 ;
  Printf.printf "  %s (align %d): done\n%!" dev.Device.name align

let () =
  Sarek_opencl.Opencl_plugin_base.For_testing.zero_copy_on_any_device := true ;
  let devs =
    Device.init () |> Array.to_list
    |> List.filter (fun (d : Device.t) -> d.Device.framework = "OpenCL")
  in
  if devs = [] then
    print_endline "test_opencl_zero_copy_vector: SKIP (no OpenCL)"
  else begin
    List.iter run devs ;
    Printf.printf
      "test_opencl_zero_copy_vector: %s (%d failures)\n"
      (if !failures = 0 then "PASSED" else "FAILED")
      !failures ;
    if !failures <> 0 then exit 1
  end
