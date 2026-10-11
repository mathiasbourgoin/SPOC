(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Does a host-pointer buffer really avoid a copy on a CPU OpenCL runtime?
 *
 * Usage: bench_opencl_host_ptr_rss.exe (host_ptr|copy) GIB
 *
 * Fills a page-aligned float32 Bigarray of GIB GiB with 1.0, wraps it as a
 * vector (Host_ptr_buffer.of_bigarray, or a plain Vector.of_bigarray for the
 * copy path), launches a kernel that reads every element (65536 work items,
 * each summing one contiguous chunk) and prints the process VmRSS after each
 * step (and the peak, VmHWM). With a host-pointer buffer RSS should stay at about the array's size;
 * the copy path adds a second copy inside the OpenCL runtime.
 *
 * Runs on the first CPU-type OpenCL device. Not part of runtest: it allocates
 * GIB GiB (twice on the copy path).
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Hpb = Spoc_core.Host_ptr_buffer

let () = Test_helpers.Benchmarks.init_backends ()

let chunk_sum =
  [%kernel
    fun (x : float32 vector)
        (y : float32 vector)
        (chunk : int32)
        (items : int32) ->
      let i = global_thread_id in
      if i < items then begin
        let acc = mut 0.0 in
        for k = 0 to chunk - 1l do
          acc := acc +. x.((i * chunk) + k)
        done ;
        y.(i) <- acc
      end]

let ir_of (_, k) =
  match k.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

(* [field] of /proc/self/status, in KiB (VmRSS: resident now; VmHWM: peak). *)
let status_kib field =
  let ic = open_in "/proc/self/status" in
  let prefix = field ^ ":" in
  let rec scan () =
    match input_line ic with
    | line when String.starts_with ~prefix line ->
        Scanf.sscanf
          (String.sub
             line
             (String.length prefix)
             (String.length line - String.length prefix))
          " %d kB"
          Fun.id
    | _ -> scan ()
  in
  Fun.protect ~finally:(fun () -> close_in ic) scan

let report step t0 =
  let mib field = float_of_int (status_kib field) /. 1024. in
  Printf.printf
    "%-22s VmRSS %8.1f MiB  VmHWM %8.1f MiB  (+%.3f s)\n%!"
    step
    (mib "VmRSS")
    (mib "VmHWM")
    (Unix.gettimeofday () -. t0)

let page_aligned_array elems =
  let base =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (elems + 1024)
  in
  let addr k =
    Ctypes.raw_address_of_ptr
      (Spoc_core.Memory.borrow_bigarray_ptr (Bigarray.Array1.sub base k 1))
  in
  let rec first_aligned k =
    if Nativeint.rem (addr k) 4096n = 0n then k else first_aligned (k + 1)
  in
  Bigarray.Array1.sub base (first_aligned 0) elems

let wrap mode dev ba =
  match mode with
  | "host_ptr" -> (
      match Hpb.of_bigarray dev Vector.Float32 ba with
      | Ok v -> v
      | Error e -> failwith (Hpb.error_message e))
  | "copy" -> Vector.of_bigarray Vector.Float32 ba
  | other -> failwith ("mode must be host_ptr or copy, not " ^ other)

let run mode gib (dev : Device.t) =
  let items = 65536 in
  let elems = gib * 1024 * 1024 * 1024 / 4 in
  let chunk = elems / items in
  Printf.printf "mode=%s size=%d GiB device=%s\n%!" mode gib dev.Device.name ;
  let t0 = Unix.gettimeofday () in
  report "start" t0 ;
  let ba = page_aligned_array elems in
  Bigarray.Array1.fill ba 1.0 ;
  report "array filled" t0 ;
  let x = wrap mode dev ba in
  report "vector created" t0 ;
  let y = Vector.create Vector.float32 items in
  Execute.run_vectors
    ~device:dev
    ~ir:(ir_of chunk_sum)
    ~args:Execute.[Vec x; Vec y; Int chunk; Int items]
    ~block:(Execute.dims1d 256)
    ~grid:(Execute.dims1d (items / 256))
    () ;
  Device.synchronize dev ;
  report "after launch + sync" t0 ;
  let ok = ref true in
  for i = 0 to items - 1 do
    if Vector.get y i <> float_of_int chunk then ok := false
  done ;
  report "after read-back" t0 ;
  Printf.printf
    "host_link=%s result=%s\n%!"
    (match Hpb.host_link x dev with
    | Some Spoc_core.Memory.Host_ptr -> "Host_ptr"
    | Some Spoc_core.Memory.Separate -> "Separate"
    | Some Spoc_core.Memory.Host_storage -> "Host_storage"
    | None -> "none")
    (if !ok then "ok" else "WRONG") ;
  if not !ok then exit 1

let () =
  let mode, gib =
    match Sys.argv with
    | [|_; mode; gib|] -> (mode, int_of_string gib)
    | _ -> failwith "usage: bench_opencl_host_ptr_rss.exe (host_ptr|copy) GIB"
  in
  match
    Device.init ~frameworks:["OpenCL"] ()
    |> Array.to_list
    |> List.find_opt (fun (d : Device.t) -> d.capabilities.is_cpu)
  with
  | None -> print_endline "bench_opencl_host_ptr_rss: SKIP (no CPU OpenCL)"
  | Some dev -> run mode gib dev
