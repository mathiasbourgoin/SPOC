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
 * (default 20). Exits 1 on any bad slice. Manual reproducer, in no alias.
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
 * On the 7950X iGPU (HIP, ROCm 7.2, 2026-10-10) the default mode and
 * pinned_ring lost launches; the other modes did not (20 runs each).
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

let run dev =
  (* SPOC_STRESS_GC=1: run pending finalisers before the loop (issue #427
     lead: finaliser hipFree during the loop). *)
  if Sys.getenv_opt "SPOC_STRESS_GC" = Some "1" then Gc.full_major () ;
  let out = Transfer.create_device_only dev Vector.int32 (n * iters) in
  Transfer.copy_to_device (i32 (n * iters) (-5l)) ~dst:out ~dst_offset:0 ;
  let x = Transfer.create_device_only dev Vector.int32 n in
  Transfer.copy_to_device (i32 n (-1l)) ~dst:x ~dst_offset:0 ;
  let src = i32 n 0l in
  let ev = lazy (Transfer.create_event dev) in
  let evs = lazy (Array.init 64 (fun _ -> Transfer.create_event dev)) in
  let big =
    lazy
      (Transfer.alloc_pinned
         dev
         Bigarray.int32
         (n * if mode = Some "pinned_ring" then 64 else iters))
  in
  let pins =
    if mode = Some "pinned_sub" || mode = Some "pinned_ring" then
      Array.init iters (fun it ->
          let k = if mode = Some "pinned_ring" then it mod 64 else it in
          Bigarray.Array1.sub (Lazy.force big) (k * n) n)
    else if control then
      Array.init iters (fun _ -> Transfer.alloc_pinned dev Bigarray.int32 n)
    else [||]
  in
  for it = 0 to iters - 1 do
    let src = if control then pins.(it) else src in
    if mode = Some "pinned_ring" && it >= 64 then
      Transfer.synchronize_event (Lazy.force evs).(it mod 64) ;
    (* Element by element on purpose: a slow host lets the queue drain, the
       timing under which the C-9 failure showed. *)
    for i = 0 to n - 1 do
      Bigarray.Array1.set src i (Int32.of_int (it + 1000))
    done ;
    if control then begin
      Transfer.copy_to_device
        ~stream:(Transfer.default_stream dev)
        src
        ~dst:x
        ~dst_offset:0 ;
      if mode = Some "pinned_event" then
        Transfer.record_event (Lazy.force ev) (Transfer.default_stream dev) ;
      if mode = Some "pinned_wait" || mode = Some "pinned_ring" then begin
        let e = (Lazy.force evs).(it mod 64) in
        if it >= 64 then Transfer.synchronize_event e ;
        Transfer.record_event e (Transfer.default_stream dev)
      end
    end
    else Transfer.stage_to_device src ~dst:x ~dst_offset:0 ;
    Execute.run_vectors
      ~device:dev
      ~ir:snapshot_ir
      ~args:
        [Execute.Vec out; Execute.Vec x; Execute.Int n; Execute.Int (it * n)]
      ~block:(Execute.dims1d 64)
      ~grid:(Execute.dims1d (n / 64))
      ()
  done ;
  Transfer.flush dev ;
  let all = i32 (n * iters) 0l in
  Transfer.copy_to_host ~src:out ~src_offset:0 all ;
  let bad = ref 0 in
  for it = 0 to iters - 1 do
    let ok = ref true in
    for i = 0 to n - 1 do
      if Bigarray.Array1.get all ((it * n) + i) <> Int32.of_int (it + 1000) then
        ok := false
    done ;
    if not !ok then begin
      incr bad ;
      Printf.printf
        "    iteration %d: slice starts with %ld\n"
        it
        (Bigarray.Array1.get all (it * n))
    end
  done ;
  Transfer.free_all_buffers out ;
  Transfer.free_all_buffers x ;
  if Lazy.is_val big then begin
    Transfer.flush dev ;
    Transfer.free_pinned (Lazy.force big)
  end
  else Array.iter Transfer.free_pinned pins ;
  !bad

let () = Sarek_hip.Hip_plugin.register ()

let () =
  let runs =
    match Option.bind (Sys.getenv_opt "SPOC_STRESS_RUNS") int_of_string_opt with
    | Some r -> r
    | None -> 20
  in
  let hip =
    Array.to_list (Device.init ())
    |> List.filter (fun d -> d.Device.framework = "HIP")
  in
  if hip = [] then
    print_endline "stress_hip_staged_launches: [SKIP] no HIP device"
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
