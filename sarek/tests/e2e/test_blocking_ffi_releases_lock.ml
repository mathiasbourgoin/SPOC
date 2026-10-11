(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A blocking device wait must not stop the other domains.
 *
 * A minor collection is a stop-the-world over every domain. A domain that sits
 * in a foreign call holding its runtime lock cannot take part, so every other
 * domain stops at its next minor collection until the call returns. The
 * blocking driver calls (device/stream/event syncs, synchronous copies, JIT
 * compiles) therefore release the lock.
 *
 * On each GPU device (HIP, CUDA, OpenCL, Vulkan; CPU devices are skipped): one
 * domain allocates in a loop while the main domain waits in [Device.synchronize]
 * on a single-thread spin kernel calibrated to about 200 ms. The test requires
 * at least [min_collections] minor collections to complete during the wait.
 * With the lock held the count is 0 to 2 (measured); unblocked, it is in the
 * hundreds. A device whose wait is shorter than
 * [min_wait_ms] cannot tell the two apart and fails as inconclusive.
 ******************************************************************************)

(* The [%kernel] expansion opens [Sarek.Kirc_types] whether it is used or not. *)
[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

let spin =
  [%kernel
    fun (out : int32 vector) (iters : int32) ->
      let open Std in
      let tid = global_thread_id in
      let acc = mut 0l in
      let i = mut 0l in
      while i < iters do
        acc := (acc * 3l) + i ;
        i := i + 1l
      done ;
      if tid = 0l then out.(0) <- acc]

let spin_ir =
  lazy
    (let _, kirc = spin in
     match kirc.Sarek.Kirc_types.body_ir with
     | Some ir -> ir
     | None -> failwith "spin: no IR")

let target_ms = 200.

let min_wait_ms = 80.

let min_collections = 10

let max_iters = Int32.to_int Int32.max_int

let launch dev out iters =
  Execute.run_vectors
    ~device:dev
    ~ir:(Lazy.force spin_ir)
    ~args:[Execute.Vec out; Execute.Int iters]
    ~block:(Execute.dims1d 1)
    ~grid:(Execute.dims1d 1)
    ()

let timed_run dev out iters =
  let t0 = Unix.gettimeofday () in
  launch dev out iters ;
  Device.synchronize dev ;
  (Unix.gettimeofday () -. t0) *. 1000.

(* Spin iterations for a kernel of about [target_ms]: double from a small count
   until one run takes 20 ms, then scale linearly. *)
let calibrate dev out =
  ignore (timed_run dev out 1000) ;
  let rec grow iters =
    let ms = timed_run dev out iters in
    if ms >= 20. || iters >= max_iters / 2 then (iters, ms) else grow (iters * 2)
  in
  let iters, ms = grow (1 lsl 16) in
  let scaled = float_of_int iters *. target_ms /. Float.max ms 1e-3 in
  int_of_float (Float.min scaled (float_of_int max_iters))

let minor_collections () = (Gc.quick_stat ()).Gc.minor_collections

(* Allocates until [stop]; [progress] counts thousands of iterations. *)
let allocator stop progress () =
  let n = ref 0 in
  while not (Atomic.get stop) do
    ignore (Sys.opaque_identity (Array.make 16 !n)) ;
    incr n ;
    if !n land 1023 = 0 then Atomic.incr progress
  done

let rec wait_started progress =
  if Atomic.get progress = 0 then begin
    Domain.cpu_relax () ;
    wait_started progress
  end

(* [true] iff the other domain completed [min_collections] minor collections
   while this one waited in [Device.synchronize]. *)
let check_device (dev : Device.t) =
  let out = Vector.create Vector.int32 1 in
  let iters = calibrate dev out in
  let stop = Atomic.make false and progress = Atomic.make 0 in
  let other = Domain.spawn (allocator stop progress) in
  wait_started progress ;
  launch dev out iters ;
  let c0 = minor_collections () and p0 = Atomic.get progress in
  let t0 = Unix.gettimeofday () in
  Device.synchronize dev ;
  let wait_ms = (Unix.gettimeofday () -. t0) *. 1000. in
  let collections = minor_collections () - c0 in
  let kiters = Atomic.get progress - p0 in
  Atomic.set stop true ;
  Domain.join other ;
  Printf.printf
    "  [%s] %s: wait %.0f ms (spin %d), %d minor GCs and %dk allocating \
     iterations on the other domain during the wait\n\
     %!"
    dev.framework
    dev.name
    wait_ms
    iters
    collections
    kiters ;
  if wait_ms < min_wait_ms then begin
    Printf.printf "    FAIL inconclusive: wait under %.0f ms\n%!" min_wait_ms ;
    false
  end
  else if collections < min_collections then begin
    Printf.printf
      "    FAIL the other domain stalled (fewer than %d minor GCs)\n%!"
      min_collections ;
    false
  end
  else begin
    Printf.printf "    ok   the other domain kept collecting\n%!" ;
    true
  end

let is_gpu (d : Device.t) =
  (not d.capabilities.is_cpu)
  &&
  match d.framework with
  | "HIP" | "OpenCL" | "Vulkan" | "CUDA/PTX" | "CUDA/C" -> true
  | _ -> false

let () =
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init () ;
  let devices = Device.init () |> Array.to_list |> List.filter is_gpu in
  Printf.printf
    "Blocking FFI releases the runtime lock (%d GPU devices)\n%!"
    (List.length devices) ;
  let failures =
    List.fold_left
      (fun acc d -> if check_device d then acc else acc + 1)
      0
      devices
  in
  exit (if failures = 0 then 0 else 1)
