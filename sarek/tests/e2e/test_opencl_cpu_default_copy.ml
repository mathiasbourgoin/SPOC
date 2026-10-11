(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Default OpenCL CPU buffers copy, like every other device's.
 *
 * Host-pointer sharing (CL_MEM_USE_HOST_PTR) is an explicit opt-in
 * (Host_ptr_buffer). A plain vector on a CPU OpenCL device, even one whose
 * host array meets CL_DEVICE_MEM_BASE_ADDR_ALIGN, gets a buffer of its own:
 * a write to the host Bigarray that bypasses Vector is invisible to kernels
 * until the vector is uploaded again. A runtime that aliased the host array
 * (the Intel CPU runtime does when the alignment is met) would show it.
 *
 * Runs on CPU-type OpenCL devices only; prints SKIP when there is none.
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

let () = Test_helpers.Benchmarks.init_backends ()

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

(* A float32 array of [n] elements at a 4096-byte boundary: it meets every
   CL_DEVICE_MEM_BASE_ADDR_ALIGN seen on this machine. *)
let page_aligned_array () =
  let base =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (n + 1024)
  in
  let addr k =
    Ctypes.raw_address_of_ptr
      (Spoc_core.Memory.borrow_bigarray_ptr (Bigarray.Array1.sub base k 1))
  in
  let rec first_aligned k =
    if Nativeint.rem (addr k) 4096n = 0n then k else first_aligned (k + 1)
  in
  Bigarray.Array1.sub base (first_aligned 0) n

let run (dev : Device.t) =
  let launch x y =
    Execute.run_vectors
      ~device:dev
      ~ir:(ir_of dbl)
      ~args:Execute.[Vec x; Vec y; Int n]
      ~block:(Execute.dims1d 256)
      ~grid:(Execute.dims1d 1)
      ()
  in
  let ba = page_aligned_array () in
  Bigarray.Array1.fill ba 1.0 ;
  let x = Vector.of_bigarray Vector.Float32 ba in
  let y = Vector.create Vector.float32 n in
  launch x y ;
  let first = Vector.get y 0 in
  (* Bypasses Vector: no location change, so no upload before the launch. *)
  ba.{0} <- 21.0 ;
  launch x y ;
  let second = Vector.get y 0 in
  Printf.printf
    "  %s: first launch y[0]=%g, after raw host write y[0]=%g\n%!"
    dev.Device.name
    first
    second ;
  first = 2.0 && second = 2.0

let () =
  let cpus =
    Device.init ~frameworks:["OpenCL"] ()
    |> Array.to_list
    |> List.filter (fun (d : Device.t) -> d.capabilities.is_cpu)
  in
  if cpus = [] then
    print_endline "test_opencl_cpu_default_copy: SKIP (no CPU OpenCL device)"
  else begin
    let ok = List.for_all run cpus in
    Printf.printf
      "test_opencl_cpu_default_copy: %s\n"
      (if ok then "PASSED" else "FAILED (default buffer aliases host memory)") ;
    if not ok then exit 1
  end
