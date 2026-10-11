(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Memory.host_to_device / device_to_host / device_to_device must size a copy
   from the bigarray's own bytes and refuse a copy that does not fit. ['a
   buffer] only fixes the OCaml element type, so a float32 bigarray type-checks
   against a float64 buffer. Runs on the Native CPU device. *)

open Spoc_core

let () = ignore (Lazy.force Sarek_native.Native_plugin.registered_backend)

let dev =
  match
    Array.to_list (Device.init ~frameworks:["Native"] ())
    |> List.filter (fun d -> d.Device.framework = "Native")
  with
  | d :: _ -> d
  | [] -> failwith "No Native device found"

let failures = ref 0

let expect_refused name f =
  match f () with
  | () ->
      incr failures ;
      Printf.printf "FAIL: %s was not refused\n" name
  | exception Invalid_argument _ -> Printf.printf "PASS: %s refused\n" name

let expect_ok name f =
  match f () with
  | () -> Printf.printf "PASS: %s\n" name
  | exception e ->
      incr failures ;
      Printf.printf "FAIL: %s raised %s\n" name (Printexc.to_string e)

let f32 n = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n

let f64 n = Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout n

let () =
  let buf64 = Memory.alloc dev 4 Bigarray.float64 in
  let buf32 = Memory.alloc dev 4 Bigarray.float32 in
  let small32 = Memory.alloc dev 2 Bigarray.float32 in
  (* R-2: both buffers are [float buffer], so a float32 bigarray type-checks
     against the float64 buffer. *)
  expect_refused "host_to_device float32 -> float64 buffer" (fun () ->
      Memory.host_to_device ~src:(f32 4) ~dst:buf64) ;
  expect_refused "device_to_host float64 buffer -> float32 bigarray" (fun () ->
      Memory.device_to_host ~src:buf64 ~dst:(f32 4)) ;
  expect_refused "host_to_device bigarray larger than buffer" (fun () ->
      Memory.host_to_device ~src:(f32 8) ~dst:buf32) ;
  expect_refused "device_to_host buffer smaller than bigarray" (fun () ->
      Memory.device_to_host ~src:buf32 ~dst:(f32 8)) ;
  expect_ok "host_to_device exact fit" (fun () ->
      Memory.host_to_device ~src:(f32 4) ~dst:buf32) ;
  expect_ok "host_to_device smaller bigarray" (fun () ->
      Memory.host_to_device ~src:(f32 2) ~dst:buf32) ;
  expect_ok "device_to_host exact fit f64" (fun () ->
      Memory.device_to_host ~src:buf64 ~dst:(f64 4)) ;
  (* R-3 *)
  expect_refused "device_to_device source larger than destination" (fun () ->
      Memory.device_to_device ~src:buf32 ~dst:small32) ;
  expect_ok "device_to_device fits" (fun () ->
      Memory.device_to_device ~src:small32 ~dst:buf32) ;
  if !failures > 0 then exit 1 ;
  print_endline "All memory sizing tests passed"
