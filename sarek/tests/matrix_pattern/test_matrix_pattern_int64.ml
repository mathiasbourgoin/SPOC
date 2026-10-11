(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A GEMM indexed with int64 arithmetic is never recognised as a matrix
 * product (compiler review C-11). The affine form is int-sized and the
 * rewrite emits int32 addresses, so [a.(int64 row * int64 k + int64 i)] would
 * overflow its address past 2^31 elements. The corpus kernel is the int32
 * [k_2d] with every array index widened to int64, as the review's probe did.
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

open Sarek
open Sarek_ir_types
module P = Sarek_matrix_pattern
module Rw = Sarek_matrix_rewrite
open Matrix_pattern_check

let k_2d =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = thread_idx_y + (block_dim_y * block_idx_y) in
      let col = thread_idx_x + (block_dim_x * block_idx_x) in
      if row < m && col < n then begin
        let sum = mut 0l in
        for i = 0 to k - 1l do
          sum := sum + (a.((row * k) + i) * b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

(* Every leaf of an index expression cast to int64: [int64 row * int64 k +
   int64 i]. *)
let rec widen = function
  | EBinop (op, a, b) -> EBinop (op, widen a, widen b)
  | leaf -> ECast (TInt64, leaf)

let rec map_index f = function
  | EArrayRead (arr, idx) -> EArrayRead (arr, f (map_index f idx))
  | EBinop (op, a, b) -> EBinop (op, map_index f a, map_index f b)
  | EUnop (op, a) -> EUnop (op, map_index f a)
  | ECast (t, a) -> ECast (t, map_index f a)
  | e -> e

(* [f] widens the indices of array reads, [g] those of array stores. *)
let rec map_stmt f g = function
  | SAssign (LArrayElem (arr, idx), e) ->
      SAssign (LArrayElem (arr, g idx), map_index f e)
  | SAssign (lv, e) -> SAssign (lv, map_index f e)
  | SSeq ss -> SSeq (List.map (map_stmt f g) ss)
  | SIf (c, t, e) -> SIf (c, map_stmt f g t, Option.map (map_stmt f g) e)
  | SFor (v, lo, hi, dir, body) -> SFor (v, lo, hi, dir, map_stmt f g body)
  | SLet (v, e, body) -> SLet (v, e, map_stmt f g body)
  | SLetMut (v, e, body) -> SLetMut (v, e, map_stmt f g body)
  | SBlock s -> SBlock (map_stmt f g s)
  | SPragma (p, s) -> SPragma (p, map_stmt f g s)
  | s -> s

let with_body ?(store = Fun.id) kernel f =
  {kernel with kern_body = map_stmt f store kernel.kern_body}

let int64_kernel = with_body (ir_of k_2d) widen

(* The same, with an int64 constant offset instead of casts. *)
let int64_offset_kernel =
  with_body (ir_of k_2d) (fun idx ->
      EBinop (Add, idx, EConst (CInt64 4_294_967_296L)))

(* Only the output index is 64-bit. *)
let int64_output_kernel = with_body ~store:widen (ir_of k_2d) Fun.id

let mentions_int64 text =
  Astring.String.is_infix ~affix:"int64" (String.lowercase_ascii text)

let rewritten_text kernel g =
  match Rw.kernel kernel g Rw.Int32_byte_planes with
  | Ok rewritten -> Format.asprintf "%a" Sarek_ir_pp.pp_kernel rewritten
  | Error why -> "rewrite declined: " ^ why

(* The int64-indexed kernel is declined with an int64 reason. Before the fix
   it was recognised, and [Rw.kernel] produced int32 addressing. *)
let check_declined name kernel =
  match P.analyze kernel with
  | [P.Gemm g] ->
      Alcotest.failf
        "%s: recognised as a GEMM; the rewrite would be:\n%s"
        name
        (rewritten_text kernel g)
  | findings ->
      let texts =
        List.filter_map
          (function
            | P.Near_miss nm -> Some (P.reason_text nm.nm_reason) | _ -> None)
          findings
      in
      Alcotest.(check bool)
        (name ^ ": a near miss mentions int64: " ^ String.concat "; " texts)
        true
        (List.exists mentions_int64 texts)

let test_cast_indices () = check_declined "int64 casts" int64_kernel

let test_constant_offset () =
  check_declined "int64 constant offset" int64_offset_kernel

let test_output_index () = check_declined "int64 output" int64_output_kernel

(* FR-C4: the int32 original keeps being recognised and rewritten. *)
let test_int32_unchanged () =
  let g = only_gemm "int32" k_2d in
  match Rw.kernel (ir_of k_2d) g Rw.Int32_byte_planes with
  | Ok _ -> ()
  | Error why -> Alcotest.failf "int32 rewrite declined: %s" why

(* FR-C3: the advice has no GEMM to claim matrix units for. *)
let test_no_advice () =
  Alcotest.(check bool)
    "no GEMM finding for the int64 kernel"
    false
    (List.exists
       (function P.Gemm _ -> true | P.Near_miss _ -> false)
       (P.analyze int64_kernel))

let () =
  Alcotest.run
    "matrix_pattern_int64"
    [
      ( "int64 index",
        [
          Alcotest.test_case "cast indices" `Quick test_cast_indices;
          Alcotest.test_case "constant offset" `Quick test_constant_offset;
          Alcotest.test_case "output index" `Quick test_output_index;
          Alcotest.test_case "int32 unchanged" `Quick test_int32_unchanged;
          Alcotest.test_case "no advice" `Quick test_no_advice;
        ] );
    ]
