(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Byte-for-byte non-regression of the Native direct float32/int32 vector
   access (Sarek.Sarek_native_fast, Sarek_native_gen_expr.gen_fast_get/set).

   Every kernel below runs on the Native device; the digest of each output
   vector is compared with the digest recorded with the implementation that
   called the accessor object for every element (parent commit 82fdd6b8). Any
   difference in a value, a rounding, an integer wrap-around or a bound is a
   failure. The kernels cover reductions, a matrix-vector product, signed
   integers with overflow, bit operations on the unsigned reading of an int32,
   gathers and in-place updates, and the out-of-range errors. *)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_native.Native_plugin.init ()

let dev = (Device.init ~frameworks:["Native"] ()).(0)

let n = 4096

let k = 64

(* y.(i) <- a * x.(i) + y.(i): float32 rounding at the store. *)
let axpy =
  [%kernel
    fun (x : float32 vector) (y : float32 vector) (a : float32) ->
      let open Std in
      let i = global_thread_id in
      if i < 4096l then y.(i) <- (a *. x.(i)) +. y.(i)]

(* One thread per row: float accumulation over a row of a float32 matrix. *)
let rowdot =
  [%kernel
    fun (m : float32 vector) (x : float32 vector) (y : float32 vector) ->
      let open Std in
      let r = global_thread_id in
      if r < 4096l then begin
        let acc = mut 0.0 in
        for c = 0 to 63 do
          acc := acc +. (m.((r * 64l) + c) *. x.(c))
        done ;
        y.(r) <- acc
      end]

(* Signed int32 arithmetic with wrap-around, division and modulo. *)
let intarith =
  [%kernel
    fun (a : int32 vector) (b : int32 vector) (out : int32 vector) ->
      let open Std in
      let i = global_thread_id in
      if i < 4096l then
        out.(i) <-
          (a.(i) * 65537l) + (b.(i) / 7l) + (a.(i) mod 13l) - 2147483000l]

(* Bit operations on the unsigned reading of an int32. *)
let bits =
  [%kernel
    fun (a : int32 vector) (out : int32 vector) ->
      let open Std in
      let i = global_thread_id in
      if i < 4096l then
        out.(i) <- (a.(i) lsr 3l) lxor (a.(i) lsl 5l) lor (a.(i) land 255l)]

(* Gather through an int32 index vector, with an int32 to float32 mix. *)
let gather =
  [%kernel
    fun (idx : int32 vector) (x : float32 vector) (y : float32 vector) ->
      let open Std in
      let i = global_thread_id in
      if i < 4096l then y.(i) <- x.(idx.(i)) +. 0.5]

(* In-place update of one vector read and written by the same launch. *)
let inplace =
  [%kernel
    fun (v : float32 vector) (w : int32 vector) ->
      let open Std in
      let i = global_thread_id in
      if i < 4096l then begin
        v.(i) <- v.(i) *. v.(i) ;
        w.(i) <- w.(i) + w.(i)
      end]

(* Reads one past the end. *)
let oob_read =
  [%kernel
    fun (x : float32 vector) (y : float32 vector) ->
      let open Std in
      let i = global_thread_id in
      if i < 4096l then y.(i) <- x.(i + 1l)]

(* Writes one past the end. *)
let oob_write =
  [%kernel
    fun (x : int32 vector) ->
      let open Std in
      let i = global_thread_id in
      if i < 4096l then x.(i + 1l) <- 7l]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no ir"

let launch kern args =
  Execute.run_vectors
    ~device:dev
    ~ir:(ir_of kern)
    ~args
    ~block:(Execute.dims1d 256)
    ~grid:(Execute.dims1d (n / 256))
    () ;
  Transfer.flush dev

let f32s seed =
  let v = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set v i (float_of_int ((i * seed mod 1013) - 500) /. 7.0)
  done ;
  v

let i32s seed =
  let v = Vector.create Vector.int32 n in
  for i = 0 to n - 1 do
    Vector.set v i (Int32.of_int ((i * seed) - 1_000_003))
  done ;
  v

let zeros_f32 len = Vector.create Vector.float32 len

let zeros_i32 len = Vector.create Vector.int32 len

(* Digest of every element's exact bits, in order. *)
let digest_f32 v =
  let b = Buffer.create 1024 in
  for i = 0 to Vector.length v - 1 do
    Buffer.add_int32_le b (Int32.bits_of_float (Vector.get v i))
  done ;
  Digest.to_hex (Digest.string (Buffer.contents b))

let digest_i32 v =
  let b = Buffer.create 1024 in
  for i = 0 to Vector.length v - 1 do
    Buffer.add_int32_le b (Vector.get v i)
  done ;
  Digest.to_hex (Digest.string (Buffer.contents b))

let error_of f =
  match f () with
  | () -> "no exception"
  | exception Invalid_argument m -> "Invalid_argument: " ^ m
  | exception e -> "other: " ^ Printexc.to_string e

(* The first line of an error message: which element tripped is a race between
   work-items, the message shape and exception are what must not change. *)
let error_shape s =
  match String.index_opt s '0' with
  | None -> s
  | Some _ ->
      String.concat
        "#"
        (String.split_on_char ' ' s
        |> List.map (fun w ->
            if String.length w > 0 && w.[0] >= '0' && w.[0] <= '9' then "N"
            else w))

let results () =
  let x = f32s 37 and y = f32s 101 in
  launch axpy Execute.[Vec x; Vec y; Float32 1.3] ;
  let axpy_d = digest_f32 y in
  let m = zeros_f32 (n * k) and xv = f32s 53 and yr = zeros_f32 n in
  for i = 0 to (n * k) - 1 do
    Vector.set m i (float_of_int ((i * 31 mod 257) - 128) /. 16.0)
  done ;
  launch rowdot Execute.[Vec m; Vec xv; Vec yr] ;
  let rowdot_d = digest_f32 yr in
  let a = i32s 104729 and b = i32s 7919 and o = zeros_i32 n in
  launch intarith Execute.[Vec a; Vec b; Vec o] ;
  let intarith_d = digest_i32 o in
  let o2 = zeros_i32 n in
  launch bits Execute.[Vec a; Vec o2] ;
  let bits_d = digest_i32 o2 in
  let idx = zeros_i32 n in
  for i = 0 to n - 1 do
    Vector.set idx i (Int32.of_int (i * 2654435 mod n))
  done ;
  let g = zeros_f32 n in
  launch gather Execute.[Vec idx; Vec x; Vec g] ;
  let gather_d = digest_f32 g in
  let v = f32s 71 and w = i32s 9973 in
  launch inplace Execute.[Vec v; Vec w] ;
  let inplace_d = digest_f32 v ^ digest_i32 w in
  let xr = f32s 3 and yo = zeros_f32 n in
  let read_err =
    error_shape (error_of (fun () -> launch oob_read Execute.[Vec xr; Vec yo]))
  in
  let xw = zeros_i32 n in
  let write_err =
    error_shape (error_of (fun () -> launch oob_write Execute.[Vec xw]))
  in
  [
    ("axpy", axpy_d);
    ("rowdot", rowdot_d);
    ("intarith", intarith_d);
    ("bits", bits_d);
    ("gather", gather_d);
    ("inplace", inplace_d);
    ("oob_read", read_err);
    ("oob_write", write_err);
  ]

(* Recorded with the parent commit; see the header. *)
let expected =
  [
    ("axpy", "b29d913fc151ff8bcfe0b950f810da98");
    ("rowdot", "221dd397d544592b56e6c70394018fff");
    ("intarith", "13595e4e7c187acb1d6daeddc04cba08");
    ("bits", "e78aa7a39b24853ebd8e8a436daf5226");
    ("gather", "60097523e9c5993373d0e0901721eb5f");
    ( "inplace",
      "2ed1c68db1933389063acfdf2f4d4e541034f6e63b48b135fcbf66702c509160" );
    ("oob_read", "Invalid_argument:#Vector.get:#index#N#out#of#bounds#[0,#N");
    ("oob_write", "Invalid_argument:#Vector.set:#index#N#out#of#bounds#[0,#N");
  ]

let () =
  let got = results () in
  if Array.length Sys.argv > 1 && Sys.argv.(1) = "--print" then
    List.iter (fun (name, d) -> Printf.printf "    (%S, %S);\n" name d) got
  else begin
    let ok = ref true in
    List.iter
      (fun (name, d) ->
        match List.assoc_opt name expected with
        | Some e when String.equal e d -> ()
        | Some e ->
            Printf.printf "MISMATCH %s: expected %s, got %s\n%!" name e d ;
            ok := false
        | None ->
            Printf.printf "NO REFERENCE for %s (got %s)\n%!" name d ;
            ok := false)
      got ;
    if !ok then print_endline "test_native_typed_access_exact: PASSED"
    else begin
      print_endline "test_native_typed_access_exact: FAILED" ;
      exit 1
    end
  end
