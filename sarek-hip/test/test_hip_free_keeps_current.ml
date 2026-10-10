(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Freeing a buffer of device A while device B is current must leave B
 * current. [Hip_api.Memory.free] runs from GC finalisers, i.e. in the middle
 * of whatever launch is in progress; leaving A selected made a launch on B
 * fail with hipErrorInvalidDevice (cheap-repeated-launches). Needs two HIP
 * devices; self-skips otherwise.
 ******************************************************************************)

module H = Sarek_hip.Hip_api

let () =
  if Sarek_hip.Hip_shared.is_disabled () || not (H.is_available ()) then begin
    print_endline "test_hip_free_keeps_current SKIPPED (no HIP)" ;
    exit 0
  end ;
  H.Device.init () ;
  if H.Device.count () < 2 then begin
    print_endline "test_hip_free_keeps_current SKIPPED (fewer than 2 devices)" ;
    exit 0
  end ;
  let a = H.Device.get 0 and b = H.Device.get 1 in
  let buf = H.Memory.alloc a 1024 Bigarray.float32 in
  H.Device.set_current b ;
  H.Memory.free buf ;
  let cur = H.Device.current_id () in
  if cur <> b.H.Device.id then begin
    Printf.printf
      "test_hip_free_keeps_current FAILED: current device %d after freeing a \
       buffer of device %d with device %d selected\n"
      cur
      a.H.Device.id
      b.H.Device.id ;
    exit 1
  end ;
  (* The tracker is not enough: the DRIVER must have [b] selected too, or the
     next module launch on [b] fails exactly as the bug did. *)
  let drv = H.Device.driver_current () in
  if drv <> b.H.Device.id then begin
    Printf.printf
      "test_hip_free_keeps_current FAILED: the driver has device %d selected \
       after the free (tracker says %d)\n"
      drv
      cur ;
    exit 1
  end ;
  print_endline "test_hip_free_keeps_current PASSED"
