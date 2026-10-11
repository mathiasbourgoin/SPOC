(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Stress for specs/transfer-pitfalls.md C-9: interleave 300 staged writes
 * with 300 launches on the default stream, each launch copying the staged
 * vector into its own slice of an output pre-filled with -5. A slice still
 * holding -5 is a launch that never wrote; any other wrong value is a write
 * that was lost or reordered. Every HIP device, SPOC_STRESS_RUNS runs each
 * (default 1). Exits 1 on any bad slice, on every device. Hand-run only, as
 * the #427 regression check: make stress-427. Not in runtest: never loop it,
 * raise the run count or run it concurrently on the iGPU (heavy iGPU load
 * hard-reset the machine on 2026-10-11).
 *
 * Other switches ("1" to enable): SPOC_STRESS_GC_PRESSURE (drop vectors,
 * random Gc.full_major / Gc.compact mid-loop; or one part only: device, host,
 * gc, sleep), SPOC_STRESS_SWITCH (a small
 * launch on another HIP device every iteration), SPOC_STRESS_VERIFY (read each
 * slice back right after its launch and 20 ms later), SPOC_STRESS_GC (a
 * Gc.full_major before the loop).
 *
 * SPOC_STRESS_MODE selects how the vector is written instead of
 * stage_to_device, all from pinned memory on the default stream:
 *   pinned        a fresh pinned buffer per write
 *   pinned_event  ... plus an event recorded after each copy
 *   pinned_wait   ... plus a wait on the event recorded 64 writes earlier
 *   pinned_sub    sub-ranges of one pinned allocation, never reused
 *   pinned_ring   64 sub-ranges of one allocation reused in turn, each reuse
 *                 after a wait on the event recorded after its last copy
 *                 (stage_to_device's discipline, written by the caller)
 * Issue #427: on the 7950X iGPU, results a kernel wrote into hipMalloc memory
 * before a host idle gap (a pause, GC, a sleep) were lost; every mode failed,
 * not only the staged ones. Integrated HIP devices now allocate uncached
 * (Hip_api.Memory); SAREK_HIP_IGPU_UNCACHED=0 restores the loss for a
 * before/after run.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Std = Sarek_stdlib.Std
open Sarek

let snapshot =
  [%kernel
    fun (out : int32 vector) (x : int32 vector) (n : int32) (base : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then out.(base + tid) <- x.(tid)]

let snapshot_ir =
  let _, kirc = snapshot in
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "snapshot: no IR"

let n = 4096

let iters = 300

let i32 len v =
  let ba = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout len in
  Bigarray.Array1.fill ba v ;
  ba

(* SPOC_STRESS_MODE=pinned: the control. Same loop, but each write is a
   copy_to_device on the default stream from its own pinned buffer (no
   staging, no ring, no events). *)
let mode = Sys.getenv_opt "SPOC_STRESS_MODE"

let control =
  mode = Some "pinned" || mode = Some "pinned_event"
  || mode = Some "pinned_wait" || mode = Some "pinned_sub"
  || mode = Some "pinned_ring"

let flag name = Sys.getenv_opt name = Some "1"

(* SPOC_STRESS_VERIFY=1: read the slice back right after its launch (the
   blocking read drains the default stream), and again after 20 ms. A slice
   still at the sentinel both times is a launch that never wrote (not a late
   write, not a write overwritten afterwards). *)
let verify_launch dev out it =
  if flag "SPOC_STRESS_VERIFY" then begin
    let first = i32 n 0l in
    Transfer.copy_to_host ~src:out ~src_offset:(it * n) first ;
    let want = Int32.of_int (it + 1000) in
    if Bigarray.Array1.get first 0 <> want then begin
      Unix.sleepf 0.02 ;
      let again = i32 n 0l in
      Transfer.copy_to_host ~src:out ~src_offset:(it * n) again ;
      Printf.printf
        "    verify: iteration %d on %s: %ld right after, %ld after 20 ms\n%!"
        it
        dev.Device.name
        (Bigarray.Array1.get first 0)
        (Bigarray.Array1.get again 0)
    end
  end

(* SPOC_STRESS_GC_PRESSURE=1: allocate and drop device vectors so finalisers
   (hipFree, ctypes frees) run mid-loop, with Gc.full_major / Gc.compact at
   random points (issue #427 hypotheses (a) and (b)). A single part can be
   selected instead: device, host, gc, or sleep (a 20 ms host pause at ~5% of
   the iterations: the trigger of the iGPU loss, see
   docs/measurements/hip-igpu-lost-launches-2026-10-11). "1" is all four. *)
let gc_pressure dev =
  match Sys.getenv_opt "SPOC_STRESS_GC_PRESSURE" with
  | None | Some "" | Some "0" -> ()
  | Some part ->
      let all = part = "1" in
      if all || part = "device" then
        ignore (Transfer.create_device_only dev Vector.int32 1024) ;
      if all || part = "host" then
        ignore (Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout 65536) ;
      let r = Random.int 40 in
      if (all || part = "sleep") && r < 2 then Unix.sleepf 0.02 ;
      if all || part = "gc" then
        if r = 0 then Gc.full_major () else if r = 1 then Gc.compact ()

(* SPOC_STRESS_SWITCH=1: after every launch, a small launch on another HIP
   device, so the current device flips every iteration (R-4/R-5/R-25). *)
let make_side_work dev =
  let others =
    Array.to_list (Device.init ())
    |> List.filter (fun d ->
        d.Device.framework = "HIP" && d.Device.id <> dev.Device.id)
  in
  match others with
  | other :: _ when flag "SPOC_STRESS_SWITCH" ->
      let o = Transfer.create_device_only other Vector.int32 64 in
      let a = Transfer.create_device_only other Vector.int32 64 in
      fun () ->
        Execute.run_vectors
          ~device:other
          ~ir:snapshot_ir
          ~args:[Execute.Vec o; Execute.Vec a; Execute.Int 64; Execute.Int 0]
          ~block:(Execute.dims1d 64)
          ~grid:(Execute.dims1d 1)
          ()
  | _ -> fun () -> ()

(* How the vector is written each iteration (see SPOC_STRESS_MODE above). *)
type writer = {
  pins : (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t array;
  big : (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t option;
  ev : Transfer.event Lazy.t;
  evs : Transfer.event array Lazy.t;
}

let make_writer dev =
  let ring = mode = Some "pinned_ring" in
  let big =
    if mode = Some "pinned_sub" || ring then
      Some
        (Transfer.alloc_pinned
           dev
           Bigarray.int32
           (n * if ring then 64 else iters))
    else None
  in
  let pins =
    match big with
    | Some b ->
        Array.init iters (fun it ->
            Bigarray.Array1.sub b ((if ring then it mod 64 else it) * n) n)
    | None when control ->
        Array.init iters (fun _ -> Transfer.alloc_pinned dev Bigarray.int32 n)
    | None -> [||]
  in
  {
    pins;
    big;
    ev = lazy (Transfer.create_event dev);
    evs = lazy (Array.init 64 (fun _ -> Transfer.create_event dev));
  }

let release_writer dev w =
  match w.big with
  | Some b ->
      Transfer.flush dev ;
      Transfer.free_pinned b
  | None -> Array.iter Transfer.free_pinned w.pins

(* Write [it + 1000] into [x] the way the selected mode does. *)
let write_vector dev w ~src ~x it =
  let src = if control then w.pins.(it) else src in
  if mode = Some "pinned_ring" && it >= 64 then
    Transfer.synchronize_event (Lazy.force w.evs).(it mod 64) ;
  (* Element by element on purpose: a slow host lets the queue drain, the
     timing under which the C-9 failure showed. *)
  for i = 0 to n - 1 do
    Bigarray.Array1.set src i (Int32.of_int (it + 1000))
  done ;
  if not control then Transfer.stage_to_device src ~dst:x ~dst_offset:0
  else begin
    let stream = Transfer.default_stream dev in
    Transfer.copy_to_device ~stream src ~dst:x ~dst_offset:0 ;
    if mode = Some "pinned_event" then
      Transfer.record_event (Lazy.force w.ev) stream ;
    if mode = Some "pinned_wait" || mode = Some "pinned_ring" then begin
      let e = (Lazy.force w.evs).(it mod 64) in
      if it >= 64 then Transfer.synchronize_event e ;
      Transfer.record_event e stream
    end
  end

(* Read [out] back and count the slices that do not hold their iteration. *)
let count_bad_slices out =
  let all = i32 (n * iters) 0l in
  Transfer.copy_to_host ~src:out ~src_offset:0 all ;
  let slice_ok it =
    let want = Int32.of_int (it + 1000) in
    let rec go i =
      i >= n || (Bigarray.Array1.get all ((it * n) + i) = want && go (i + 1))
    in
    go 0
  in
  let bad = ref 0 in
  for it = 0 to iters - 1 do
    if not (slice_ok it) then begin
      incr bad ;
      Printf.printf
        "    iteration %d: slice starts with %ld\n"
        it
        (Bigarray.Array1.get all (it * n))
    end
  done ;
  !bad

let launch_snapshot dev ~out ~x it =
  Execute.run_vectors
    ~device:dev
    ~ir:snapshot_ir
    ~args:[Execute.Vec out; Execute.Vec x; Execute.Int n; Execute.Int (it * n)]
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d (n / 64))
    ()

let run dev =
  (* SPOC_STRESS_GC=1: run pending finalisers before the loop (issue #427
     lead: finaliser hipFree during the loop). *)
  if flag "SPOC_STRESS_GC" then Gc.full_major () ;
  let side_work = make_side_work dev in
  let out = Transfer.create_device_only dev Vector.int32 (n * iters) in
  Transfer.copy_to_device (i32 (n * iters) (-5l)) ~dst:out ~dst_offset:0 ;
  let x = Transfer.create_device_only dev Vector.int32 n in
  Transfer.copy_to_device (i32 n (-1l)) ~dst:x ~dst_offset:0 ;
  let src = i32 n 0l in
  let w = make_writer dev in
  for it = 0 to iters - 1 do
    gc_pressure dev ;
    write_vector dev w ~src ~x it ;
    launch_snapshot dev ~out ~x it ;
    verify_launch dev out it ;
    side_work ()
  done ;
  Transfer.flush dev ;
  let bad = count_bad_slices out in
  Transfer.free_all_buffers out ;
  Transfer.free_all_buffers x ;
  release_writer dev w ;
  bad

let () = Sarek_hip.Hip_plugin.register ()

let () =
  let runs =
    match Option.bind (Sys.getenv_opt "SPOC_STRESS_RUNS") int_of_string_opt with
    | Some r -> r
    | None -> 1
  in
  let hip =
    Array.to_list (Device.init ())
    |> List.filter (fun d -> d.Device.framework = "HIP")
  in
  if runs < 1 then begin
    prerr_endline "stress_hip_staged_launches: SPOC_STRESS_RUNS must be >= 1" ;
    exit 2
  end ;
  (* Hand-run only: no device is a failure, not a pass. *)
  if hip = [] then begin
    prerr_endline "stress_hip_staged_launches: no HIP device, nothing checked" ;
    exit 2
  end
  else begin
    let failed = ref 0 in
    List.iter
      (fun dev ->
        let bad_runs = ref 0 in
        for _ = 1 to runs do
          if run dev > 0 then incr bad_runs
        done ;
        Printf.printf
          "%s: %d of %d runs with a bad slice\n%!"
          dev.Device.name
          !bad_runs
          runs ;
        failed := !failed + !bad_runs)
      hip ;
    if !failed > 0 then exit 1
  end
