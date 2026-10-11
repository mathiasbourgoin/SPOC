(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Freeing a buffer of device A while device B is current must leave B
 * current. [Cuda_api.Memory.free] runs from GC finalisers, i.e. in the middle
 * of whatever launch is in progress; leaving A's context selected made a
 * launch on B run with the wrong context (the HIP twin of this test is
 * test_hip_free_keeps_current). Needs two CUDA devices; self-skips otherwise.
 ******************************************************************************)

module C = Sarek_cuda.Cuda_api

let fail fmt =
  Printf.ksprintf
    (fun s ->
      Printf.printf "test_cuda_free_keeps_current FAILED: %s\n" s ;
      exit 1)
    fmt

let () =
  if Sarek_cuda.Cuda_shared.is_disabled () || not (C.is_driver_available ())
  then begin
    print_endline "test_cuda_free_keeps_current SKIPPED (no CUDA)" ;
    exit 0
  end ;
  C.Device.init () ;
  if C.Device.count () < 2 then begin
    print_endline "test_cuda_free_keeps_current SKIPPED (fewer than 2 devices)" ;
    exit 0
  end ;
  let a = C.Device.get 0 and b = C.Device.get 1 in
  let buf = C.Memory.alloc a 1024 Bigarray.float32 in
  C.Device.set_current b ;
  C.Memory.free buf ;
  let cur = C.Device.current_id () in
  if cur <> b.C.Device.id then
    fail
      "current device %d after freeing a buffer of device %d with device %d \
       selected"
      cur
      a.C.Device.id
      b.C.Device.id ;
  (* The tracker is not enough: the DRIVER must have [b]'s context current. *)
  let drv = C.Device.driver_current () in
  if drv <> b.C.Device.id then
    fail
      "the driver has device %d selected after the free (tracker says %d)"
      drv
      cur ;
  print_endline "test_cuda_free_keeps_current PASSED"
