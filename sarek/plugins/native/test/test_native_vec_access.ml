(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Element accessors handed to native kernels: values, bounds behaviour, and
    the allocation per read. Every [v.(i)] of a native kernel is one accessor
    call, and on a multi-domain pool every minor collection stops all domains,
    so the words allocated per read set how far Native scales. *)

module Vector = Spoc_core.Vector

let n = 4096

let f32_vector () =
  let v = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set v i (float_of_int (i * 37 mod 101) /. 8.0)
  done ;
  v

let i32_vector () =
  let v = Vector.create Vector.int32 n in
  for i = 0 to n - 1 do
    Vector.set v i (Int32.of_int ((i * 7919) - 1_000_000))
  done ;
  v

(* The closures a kernel calls; [NV] hides the element type, so they are
   copied out of it. *)
type acc = {
  get_f32 : int -> float;
  set_f32 : int -> float -> unit;
  get_i32 : int -> int32;
}

let accessors v =
  match
    Sarek_native.Native_vec_access.native_vec_of_exec_vector
      (match Sarek.Execute.exec_arg_of_vector v with
      | Spoc_framework.Framework_sig.EA_Vec m -> m
      | _ -> Alcotest.fail "exec_arg_of_vector: not a vector")
  with
  | Sarek_ir_types.NA_Vec (Sarek_ir_types.NV ops) ->
      {get_f32 = ops.get_f32; set_f32 = ops.set_f32; get_i32 = ops.get_i32}
  | _ -> Alcotest.fail "native_vec_of_exec_vector: not NA_Vec"

let test_f32_values () =
  let v = f32_vector () in
  let acc = accessors v in
  for i = 0 to n - 1 do
    Alcotest.(check (float 0.0)) "get_f32" (Vector.get v i) (acc.get_f32 i)
  done ;
  acc.set_f32 5 2.5 ;
  Alcotest.(check (float 0.0)) "set_f32" 2.5 (Vector.get v 5)

let test_i32_values () =
  let v = i32_vector () in
  let acc = accessors v in
  for i = 0 to n - 1 do
    Alcotest.(check int32) "get_i32" (Vector.get v i) (acc.get_i32 i)
  done ;
  Alcotest.(check (float 0.0))
    "get_f32 of an int32 vector converts"
    (Int32.to_float (Vector.get v 9))
    (acc.get_f32 9)

let raises_invalid_arg name f =
  match f () with
  | _ -> Alcotest.failf "%s: no exception" name
  | exception Invalid_argument msg ->
      if not (String.starts_with ~prefix:"Vector.get: index" msg) then
        Alcotest.failf "%s: unexpected message %S" name msg

let test_bounds () =
  let f = accessors (f32_vector ()) and i = accessors (i32_vector ()) in
  raises_invalid_arg "get_f32 n" (fun () -> f.get_f32 n) ;
  raises_invalid_arg "get_f32 -1" (fun () -> f.get_f32 (-1)) ;
  raises_invalid_arg "get_i32 n" (fun () -> i.get_i32 n) ;
  raises_invalid_arg "get_i32 -1" (fun () -> i.get_i32 (-1))

(* Words allocated per read on this domain, over [reps] passes. *)
let words_per_read (read : int -> 'a) =
  let reps = 25 in
  let before = Gc.minor_words () in
  for _ = 1 to reps do
    for i = 0 to n - 1 do
      ignore (Sys.opaque_identity (read i))
    done
  done ;
  (Gc.minor_words () -. before) /. float_of_int (reps * n)

(* A boxed float is 2 words and a boxed int32 3: what the closure returning the
   element costs. The typed_value round trip allocates 9 (float32) and 10 (int32)
   words per read. *)
let test_f32_allocation () =
  let acc = accessors (f32_vector ()) in
  let w = words_per_read acc.get_f32 in
  if w > 2.0 then Alcotest.failf "get_f32: %.1f words per read (max 2)" w

let test_i32_allocation () =
  let acc = accessors (i32_vector ()) in
  let w = words_per_read acc.get_i32 in
  if w > 3.0 then Alcotest.failf "get_i32: %.1f words per read (max 3)" w

let native_arg v =
  match Sarek.Execute.exec_arg_of_vector v with
  | Spoc_framework.Framework_sig.EA_Vec m ->
      Sarek_native.Native_vec_access.native_vec_of_exec_vector m
  | _ -> Alcotest.fail "exec_arg_of_vector: not a vector"

module Fast = Sarek.Sarek_native_fast

(* The record the PPX-generated kernels read: the vector's own storage and
   length for a matching float32/int32 vector, nothing for any other. *)
let test_fast_record () =
  let f = f32_vector () and i = i32_vector () in
  let ff = Fast.f32_of_native_arg (native_arg f) in
  Alcotest.(check int) "f32 length" n ff.Fast.f32_len ;
  Bigarray.Array1.set ff.Fast.f32_ba 5 2.5 ;
  Alcotest.(check (float 0.0)) "f32 shares storage" 2.5 (Vector.get f 5) ;
  let fi = Fast.i32_of_native_arg (native_arg i) in
  Alcotest.(check int) "i32 length" n fi.Fast.i32_len ;
  Alcotest.(check int)
    "int32 vector has no f32 record"
    0
    (Fast.f32_of_native_arg (native_arg i)).Fast.f32_len ;
  Alcotest.(check int)
    "float32 vector has no i32 record"
    0
    (Fast.i32_of_native_arg (native_arg f)).Fast.i32_len

(* A store marks the device copy stale, the transition [Vector.set] makes. *)
let test_mark_written () =
  Sarek_native.Native_plugin.init () ;
  let dev = (Spoc_core.Device.init ~frameworks:["Native"] ()).(0) in
  let v = f32_vector () in
  v.Vector.location <- Vector.Both dev ;
  Fast.mark_written v ;
  (match v.Vector.location with
  | Vector.Stale_GPU _ -> ()
  | _ -> Alcotest.fail "Both not turned into Stale_GPU") ;
  v.Vector.location <- Vector.CPU ;
  Fast.mark_written v ;
  match v.Vector.location with
  | Vector.CPU -> ()
  | _ -> Alcotest.fail "CPU location changed"

let () =
  Alcotest.run
    "native_vec_access"
    [
      ( "fast",
        [
          Alcotest.test_case "record" `Quick test_fast_record;
          Alcotest.test_case "mark written" `Quick test_mark_written;
        ] );
      ( "values",
        [
          Alcotest.test_case "float32" `Quick test_f32_values;
          Alcotest.test_case "int32" `Quick test_i32_values;
          Alcotest.test_case "bounds" `Quick test_bounds;
        ] );
      ( "allocation",
        [
          Alcotest.test_case "get_f32" `Quick test_f32_allocation;
          Alcotest.test_case "get_i32" `Quick test_i32_allocation;
        ] );
    ]
