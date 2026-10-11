(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A GEMM indexed with int64 arithmetic is not rewritten (compiler review
 * C-11). On every device that would otherwise rewrite the int32 [k_2d],
 * Sarek_matrix.run on the int64-indexed variant must return [Original] with
 * an int64 reason, and the launch must still compute the product (small
 * sizes: the address never nears 2^31, the point is the decision).
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
open Sarek_ir_types
open Matrix_rewrite_support
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Every leaf of an array index cast to int64: [int64 row * int64 k + int64 i]
   (the same widening as sarek/tests/matrix_pattern). *)
let rec widen = function
  | EBinop (op, a, b) -> EBinop (op, widen a, widen b)
  | leaf -> ECast (TInt64, leaf)

let rec map_index f = function
  | EArrayRead (arr, idx) -> EArrayRead (arr, f (map_index f idx))
  | EBinop (op, a, b) -> EBinop (op, map_index f a, map_index f b)
  | EUnop (op, a) -> EUnop (op, map_index f a)
  | ECast (t, a) -> ECast (t, map_index f a)
  | e -> e

let rec map_stmt f = function
  | SAssign (lv, e) -> SAssign (lv, map_index f e)
  | SSeq ss -> SSeq (List.map (map_stmt f) ss)
  | SIf (c, t, e) -> SIf (c, map_stmt f t, Option.map (map_stmt f) e)
  | SFor (v, lo, hi, dir, body) -> SFor (v, lo, hi, dir, map_stmt f body)
  | SLet (v, e, body) -> SLet (v, e, map_stmt f body)
  | SLetMut (v, e, body) -> SLetMut (v, e, map_stmt f body)
  | SBlock s -> SBlock (map_stmt f s)
  | SPragma (p, s) -> SPragma (p, map_stmt f s)
  | s -> s

let int64_kernel =
  let k = ir_of k_2d in
  {k with kern_body = map_stmt widen k.kern_body}

let mentions_int64 text =
  let text = String.lowercase_ascii text and affix = "int64" in
  let last = String.length text - String.length affix in
  let rec from i =
    i <= last
    && (String.sub text i (String.length affix) = affix || from (i + 1))
  in
  from 0

let run_case dev =
  let m = 17 and n = 33 and k = 5 in
  let a = Array.init (m * k) (fun _ -> random_int32 ()) in
  let b = Array.init (k * n) (fun _ -> random_int32 ()) in
  let cv = vec (Array.make (m * n) sentinel) in
  let i x = Execute.Int32 (Int32.of_int x) in
  let args =
    [Execute.Vec (vec a); Execute.Vec (vec b); Execute.Vec cv; i m; i n; i k]
  in
  let block, grid = geo_2d ~m ~n ~nb:1 in
  let outcome = launch ~rewrite:true dev int64_kernel ~args ~block ~grid in
  Transfer.flush dev ;
  let what = Printf.sprintf "int64 index on %s" dev.Device.name in
  (match outcome with
  | Sarek_matrix.Original why when mentions_int64 why ->
      ok "%s: original (%s)" what why
  | Sarek_matrix.Original why ->
      fail "%s: declined, but not for int64: %s" what why
  | Sarek_matrix.Rewritten _ ->
      fail
        "%s: rewritten (%s); int32 addressing would overflow"
        what
        (Sarek_matrix.outcome_text outcome)) ;
  let expected =
    reference
      ~layout:{a_rm = true; b_rm = true}
      ~m
      ~n
      ~k
      ~nb:1
      ~init:(fun _ -> 0l)
      ~epi:Fun.id
      a
      b
      (Array.make (m * n) sentinel)
  in
  ignore (compare_arrays (what ^ " vs host") expected (Vector.to_array cv))

let () =
  Random.init 20261011 ;
  Array.iter (fun dev -> if offers_int dev then run_case dev) (Device.init ()) ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
