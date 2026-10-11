(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Host_ptr_buffer: the explicit host-pointer (CL_MEM_USE_HOST_PTR) path.
 *
 * On every CPU-type OpenCL device:
 * - a default vector's buffer is [Separate] (no implicit sharing);
 * - [of_bigarray] over a page-aligned array gives a [Host_ptr] buffer, and
 *   kernel results through it equal the copy path;
 * - the synchronisation contract: [publish_host_writes] makes raw host writes
 *   visible to the next launch, [acquire_device_writes] makes a kernel's
 *   writes visible in the Bigarray;
 * - refusals: misaligned or empty array, vector that already has a buffer on
 *   the device, non-CPU OpenCL device (checked without creating a context);
 * - the synchronisation contract again on the first non-CPU OpenCL device,
 *   forced with the test-only hook, where a missing map/unmap is visible.
 *
 * Prints SKIP when there is no CPU OpenCL device.
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Hpb = Spoc_core.Host_ptr_buffer
module Mem = Spoc_core.Memory

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

let failures = ref 0

let expect what cond =
  if not cond then begin
    incr failures ;
    Printf.printf "  FAIL %s\n%!" what
  end

(* [n] float32 at a 4096-byte boundary plus [skew] elements. *)
let page_aligned_array ?(skew = 0) () =
  let base =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (n + 1024 + skew)
  in
  let addr k =
    Ctypes.raw_address_of_ptr
      (Mem.borrow_bigarray_ptr (Bigarray.Array1.sub base k 1))
  in
  let rec first_aligned k =
    if Nativeint.rem (addr k) 4096n = 0n then k else first_aligned (k + 1)
  in
  Bigarray.Array1.sub base (first_aligned 0 + skew) n

let launch dev x y =
  Execute.run_vectors
    ~device:dev
    ~ir:(ir_of dbl)
    ~args:Execute.[Vec x; Vec y; Int n]
    ~block:(Execute.dims1d 256)
    ~grid:(Execute.dims1d 1)
    ()

let host_ptr_vector dev ba =
  match Hpb.of_bigarray dev Vector.Float32 ba with
  | Ok v -> v
  | Error e -> failwith ("of_bigarray refused: " ^ Hpb.error_message e)

let filled f =
  let ba = page_aligned_array () in
  for i = 0 to n - 1 do
    ba.{i} <- f i
  done ;
  ba

(* Same kernel, same input: host-pointer path and copy path agree. *)
let check_equals_copy_path dev =
  let input i = float_of_int (i * 7 mod 31) -. 3.5 in
  let x_hp = host_ptr_vector dev (filled input) in
  let x_cp = Vector.of_bigarray Vector.Float32 (filled input) in
  let y_hp = Vector.create Vector.float32 n in
  let y_cp = Vector.create Vector.float32 n in
  launch dev x_hp y_hp ;
  launch dev x_cp y_cp ;
  expect
    "host-ptr input is Host_ptr"
    (Hpb.host_link x_hp dev = Some Mem.Host_ptr) ;
  expect "default input is Separate" (Hpb.host_link x_cp dev = Some Mem.Separate) ;
  for i = 0 to n - 1 do
    expect
      (Printf.sprintf "result %d equals copy path" i)
      (Vector.get y_hp i = Vector.get y_cp i)
  done

(* Raw host writes reach a kernel after [publish_host_writes]; a kernel's
   writes reach the Bigarray after [acquire_device_writes]. *)
let check_sync_contract dev =
  let in_ba = filled (fun _ -> 1.0) in
  let out_ba = filled (fun _ -> -1.0) in
  let x = host_ptr_vector dev in_ba in
  let y = host_ptr_vector dev out_ba in
  launch dev x y ;
  Hpb.acquire_device_writes y dev ;
  expect "kernel write seen in Bigarray" (out_ba.{5} = 2.0) ;
  in_ba.{5} <- 21.0 ;
  Hpb.publish_host_writes x dev ;
  (* The host copy is now authoritative and published to [dev]. *)
  expect
    "publish made the host copy current"
    (match Vector.location x with
    | Vector.Both d -> d.Device.id = dev.Device.id
    | _ -> false) ;
  launch dev x y ;
  Hpb.acquire_device_writes y dev ;
  expect "published host write seen by kernel" (out_ba.{5} = 42.0) ;
  (* Reading the device copy back must not bring the old value back. *)
  Hpb.acquire_device_writes x dev ;
  expect "published host write survives a read-back" (in_ba.{5} = 21.0)

let is_misaligned = function
  | Hpb.Refused {reason = Spoc_framework.Framework_sig.Host_ptr_misaligned _; _}
    ->
      true
  | _ -> false

let check_refusals dev =
  (match Hpb.of_bigarray dev Vector.Float32 (page_aligned_array ~skew:1 ()) with
  | Error e -> expect "misaligned refusal" (is_misaligned e)
  | Ok _ -> expect "misaligned array accepted" false) ;
  let v = Vector.of_bigarray Vector.Float32 (filled (fun _ -> 0.0)) in
  launch dev v (Vector.create Vector.float32 n) ;
  match Hpb.attach v dev with
  | Error (Hpb.Buffer_exists _) -> ()
  | Error e -> expect ("wrong refusal: " ^ Hpb.error_message e) false
  | Ok () -> expect "attach over an existing buffer accepted" false

let check_empty_refused dev =
  let empty = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 0 in
  match Hpb.of_bigarray dev Vector.Float32 empty with
  | Error
      (Hpb.Refused {reason = Spoc_framework.Framework_sig.Host_ptr_empty; _}) ->
      ()
  | Error e -> expect ("empty: wrong refusal " ^ Hpb.error_message e) false
  | Ok _ -> expect "empty array accepted" false

let check_non_cpu_refused (dev : Device.t) =
  match Hpb.of_bigarray dev Vector.Float32 (filled (fun _ -> 0.0)) with
  | Error
      (Hpb.Refused
         {reason = Spoc_framework.Framework_sig.Host_ptr_not_cpu_device; _}) ->
      ()
  | Error e -> expect ("non-CPU: wrong refusal " ^ Hpb.error_message e) false
  | Ok _ -> expect ("non-CPU device accepted: " ^ dev.Device.name) false

let () =
  let devs = Device.init ~frameworks:["OpenCL"] () |> Array.to_list in
  let cpus, others =
    List.partition (fun (d : Device.t) -> d.capabilities.is_cpu) devs
  in
  if cpus = [] then
    print_endline "test_opencl_host_ptr_buffer: SKIP (no CPU OpenCL device)"
  else begin
    List.iter
      (fun dev ->
        check_equals_copy_path dev ;
        check_sync_contract dev ;
        check_refusals dev ;
        check_empty_refused dev ;
        Printf.printf "  %s: done\n%!" dev.Device.name)
      cpus ;
    List.iter check_non_cpu_refused others ;
    (* The Intel CPU runtime aliases the host memory even without a map, so
       on it a no-op publish/acquire would pass. A GPU runtime keeps a private
       copy behind CL_MEM_USE_HOST_PTR, which makes the map/unmap observable:
       force the opt-in on the first non-CPU OpenCL device (the test-only
       hook; device order puts a discrete GPU first on rusticl). *)
    (match others with
    | gpu :: _ ->
        Sarek_opencl.Opencl_plugin_base.For_testing.host_ptr_on_any_device :=
          true ;
        Fun.protect
          ~finally:(fun () ->
            Sarek_opencl.Opencl_plugin_base.For_testing.host_ptr_on_any_device :=
              false)
          (fun () -> check_sync_contract gpu) ;
        Printf.printf "  %s: sync contract done\n%!" gpu.Device.name
    | [] ->
        print_endline "  no non-CPU OpenCL device: sync contract on CPU only") ;
    Printf.printf
      "test_opencl_host_ptr_buffer: %s (%d failures)\n"
      (if !failures = 0 then "PASSED" else "FAILED")
      !failures ;
    if !failures <> 0 then exit 1
  end
