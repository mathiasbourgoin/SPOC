(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * [Advanced.Pinned]: explicit, opt-in page-locked host memory on top of
 * [Transfer.alloc_pinned] / [register_host]. Runs on the CPU backends, where
 * "page-locked" is ordinary host memory but the bookkeeping (is_pinned,
 * lifetime, refusals) is the same as on a GPU.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Pinned = Spoc_core.Advanced.Pinned

let failures = ref 0

let check name cond =
  if cond then Printf.printf "    ok   %s\n%!" name
  else begin
    incr failures ;
    Printf.printf "    FAIL %s\n%!" name
  end

let refuses_invalid f =
  match f () with
  | _ -> false
  | exception Invalid_argument _ -> true
  | exception e ->
      Printf.printf "      (unexpected %s)\n" (Printexc.to_string e) ;
      false

let i32_ba n f =
  let ba = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
  for i = 0 to n - 1 do
    Bigarray.Array1.set ba i (f i)
  done ;
  ba

let test_alloc_roundtrip dev =
  check "is_available" (Pinned.is_available dev) ;
  check "has_advanced_features" (Spoc_core.Advanced.has_advanced_features dev) ;
  let n = 256 in
  let p = Pinned.alloc dev Bigarray.int32 n in
  let ba = Pinned.bigarray p in
  check "length" (Bigarray.Array1.dim ba = n) ;
  check "is_pinned" (Transfer.is_pinned ba) ;
  check "live" (Pinned.is_live p) ;
  check "device" (Pinned.device p == dev) ;
  for i = 0 to n - 1 do
    Bigarray.Array1.set ba i (Int32.of_int (i * 3))
  done ;
  let dst = Transfer.create_device_only dev Vector.int32 n in
  Transfer.copy_to_device ba ~dst ~dst_offset:0 ;
  let back = i32_ba n (fun _ -> -1l) in
  Transfer.copy_to_host ~src:dst ~src_offset:0 back ;
  check
    "round trip through the pinned buffer"
    (let ok = ref true in
     for i = 0 to n - 1 do
       if Bigarray.Array1.get back i <> Int32.of_int (i * 3) then ok := false
     done ;
     !ok) ;
  Transfer.free_all_buffers dst ;
  Pinned.release p ;
  check "not live after release" (not (Pinned.is_live p)) ;
  check "not pinned after release" (not (Transfer.is_pinned ba)) ;
  check
    "bigarray refused after release"
    (refuses_invalid (fun () -> Pinned.bigarray p)) ;
  Pinned.release p ;
  check "release is idempotent" (not (Pinned.is_live p))

let test_alloc_refusals dev =
  check
    "length 0 refused"
    (refuses_invalid (fun () -> Pinned.alloc dev Bigarray.int32 0)) ;
  check
    "float16 refused"
    (refuses_invalid (fun () -> Pinned.alloc dev Bigarray.float16 4))

let test_register dev =
  let ba = i32_ba 1024 Int32.of_int in
  let p = Pinned.register dev ba in
  check "registered is_pinned" (Transfer.is_pinned ba) ;
  check "registered handle shares the array" (Pinned.bigarray p == ba) ;
  check
    "second registration refused"
    (refuses_invalid (fun () -> Pinned.register dev ba)) ;
  check "first handle still live" (Pinned.is_live p) ;
  Pinned.release p ;
  check "unregistered" (not (Transfer.is_pinned ba)) ;
  check "array still usable" (Bigarray.Array1.get ba 7 = 7l)

let test_with_alloc dev =
  let seen = ref None in
  let r =
    Pinned.with_alloc dev Bigarray.int32 16 (fun p ->
        seen := Some p ;
        Bigarray.Array1.set (Pinned.bigarray p) 0 42l ;
        Bigarray.Array1.get (Pinned.bigarray p) 0)
  in
  check "with_alloc returns the result" (r = 42l) ;
  (match !seen with
  | Some p -> check "with_alloc releases" (not (Pinned.is_live p))
  | None -> check "with_alloc ran" false) ;
  let seen = ref None in
  (match
     Pinned.with_alloc dev Bigarray.int32 16 (fun p ->
         seen := Some p ;
         failwith "boom")
   with
  | _ -> check "exception propagates" false
  | exception Failure m -> check "exception propagates" (m = "boom")) ;
  match !seen with
  | Some p -> check "with_alloc releases on exception" (not (Pinned.is_live p))
  | None -> check "with_alloc ran (exception)" false

let () =
  Test_helpers.Benchmarks.init_backends () ;
  let devices = Device.init () in
  let cpu =
    Array.to_list devices
    |> List.filter (fun (d : Device.t) ->
        d.framework = "Native" || d.framework = "Interpreter")
  in
  check "a CPU device is present" (cpu <> []) ;
  List.iter
    (fun (d : Device.t) ->
      Printf.printf "  [%s]\n%!" d.name ;
      test_alloc_roundtrip d ;
      test_alloc_refusals d ;
      test_register d ;
      test_with_alloc d)
    cpu ;
  if !failures > 0 then begin
    Printf.printf "%d check(s) failed\n" !failures ;
    exit 1
  end
