(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Profiling events on device B must be created and recorded with B selected,
 * even while device A is current: HIP events and the NULL (default) stream
 * belong to the CURRENT device. Needs two HIP devices; self-skips otherwise.
 ******************************************************************************)

module Device = Spoc_core.Device
module Profiling = Spoc_core.Profiling
module H = Sarek_hip.Hip_api

let () = Sarek_hip.Hip_plugin.register ()

let fail fmt =
  Printf.ksprintf
    (fun s ->
      Printf.printf "test_hip_profiling_selects_device FAILED: %s\n" s ;
      exit 1)
    fmt

let select (d : Device.t) = H.Device.set_current (H.Device.get d.backend_id)

let expect_driver_on what (d : Device.t) =
  let cur = H.Device.driver_current () in
  if cur <> d.backend_id then
    fail "%s: driver has device %d selected, expected %d" what cur d.backend_id

let run_timer a b =
  select a ;
  let timer = Profiling.create_timer b in
  expect_driver_on "create_timer" b ;
  select a ;
  let (module T : Profiling.EVENT_PAIR) = timer in
  T.start () ;
  expect_driver_on "start" b ;
  select a ;
  T.stop () ;
  expect_driver_on "stop" b ;
  select a ;
  let ms = T.elapsed_ms () in
  if not (ms >= 0.0) then fail "elapsed_ms = %g" ms ;
  T.destroy ()

let run_event a b =
  select a ;
  let ev = Profiling.create_event b in
  expect_driver_on "create_event" b ;
  select a ;
  Profiling.record_event ev ;
  expect_driver_on "record_event" b ;
  select a ;
  Profiling.synchronize_event ev ;
  expect_driver_on "synchronize_event" b ;
  select a ;
  Profiling.destroy_event ev ;
  expect_driver_on "destroy_event" b

let () =
  if Sarek_hip.Hip_shared.is_disabled () || not (H.is_available ()) then begin
    print_endline "test_hip_profiling_selects_device SKIPPED (no HIP)" ;
    exit 0
  end ;
  match Array.to_list (Device.by_framework "HIP") with
  | a :: b :: _ ->
      run_timer a b ;
      run_event a b ;
      print_endline "test_hip_profiling_selects_device PASSED"
  | l ->
      print_endline
        (Printf.sprintf
           "test_hip_profiling_selects_device SKIPPED (%d HIP device(s) of %d \
            total)"
           (List.length l)
           (Spoc_core.Device.count ()))
