(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The opt-ins of the AUTOMATIC float matrix rewrite
 * (specs/matrix-pattern-dynamic.md, "Opt-ins"). The opt-ins are one-way and
 * process-wide, so each combination runs in its own process:
 *
 *   test_matrix_optin.exe relaxed-only   accept_relaxed_accumulation only
 *   test_matrix_optin.exe new-only       accept_automatic_float_rewrite only
 *   test_matrix_optin.exe both           both
 *
 * On a HIP gfx11 device, for an f16 GEMM kernel launched in mode Auto by
 * Execute.run_vectors: rewritten only with both. Sarek_matrix.run (explicit)
 * needs the accuracy opt-in only. Without such a device: SKIP.
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
open Matrix_rewrite_support
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer
module Spec = Sarek_execute.Matrix_specialise

let k_f16 =
  [%kernel
    fun (a : float16 vector)
        (b : float16 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum :=
            sum
            +. float32_of_float16 a.((row * k) + i)
               *. float32_of_float16 b.((i * n) + col)
        done ;
        c.((row * n) + col) <- sum
      end]

let size = 64

let args () =
  let i = Execute.Int32 (Int32.of_int size) in
  [
    Execute.Vec (vec_f16_of (Array.make (size * size) 0.5));
    Execute.Vec (vec_f16_of (Array.make (size * size) 0.25));
    Execute.Vec (vec_f32_zero (size * size));
    i;
    i;
    i;
  ]

let contains ~needle s =
  let n = String.length needle in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = needle || at (i + 1))
  in
  at 0

(* The automatic path: not explicit. *)
let automatic dev =
  let block, grid = geo_2d ~m:size ~n:size ~nb:1 in
  let o =
    Execute.run_vectors_outcome
      ~device:dev
      ~ir:(ir_of k_f16)
      ~args:(args ())
      ~block
      ~grid
      ()
  in
  Transfer.flush dev ;
  o

let explicit dev =
  let block, grid = geo_2d ~m:size ~n:size ~nb:1 in
  let o =
    Sarek_matrix.run
      ~device:dev
      ~ir:(ir_of k_f16)
      ~args:(args ())
      ~block
      ~grid
      ()
  in
  Transfer.flush dev ;
  o

let expect_original what ~naming = function
  | Spec.Original why when contains ~needle:naming why ->
      ok "%s: original, names %s (%s)" what naming why
  | Spec.Original why ->
      fail "%s: original, but the reason omits %s: %s" what naming why
  | Spec.Specialised _ -> fail "%s: rewritten" what

let expect_rewritten what = function
  | Spec.Specialised _ -> ok "%s: rewritten" what
  | Spec.Original why -> fail "%s: not rewritten (%s)" what why

let check_explicit what dev ~rewritten =
  match (explicit dev, rewritten) with
  | Sarek_matrix.Rewritten _, true -> ok "%s: rewritten" what
  | Sarek_matrix.Original why, false -> ok "%s: original (%s)" what why
  | Sarek_matrix.Rewritten _, false -> fail "%s: rewritten" what
  | Sarek_matrix.Original why, true -> fail "%s: not rewritten (%s)" what why

let run_case combo dev =
  let name = dev.Device.name in
  match combo with
  | "relaxed-only" ->
      expect_original
        (Printf.sprintf "automatic, relaxed only, on %s" name)
        ~naming:"accept_automatic_float_rewrite"
        (automatic dev) ;
      check_explicit
        (Printf.sprintf "Sarek_matrix.run, relaxed only, on %s" name)
        dev
        ~rewritten:true
  | "new-only" ->
      expect_original
        (Printf.sprintf "automatic, new opt-in only, on %s" name)
        ~naming:"accept_relaxed_accumulation"
        (automatic dev) ;
      check_explicit
        (Printf.sprintf "Sarek_matrix.run, new opt-in only, on %s" name)
        dev
        ~rewritten:false
  | _ ->
      expect_rewritten
        (Printf.sprintf "automatic, both opt-ins, on %s" name)
        (automatic dev)

let () =
  let combo = if Array.length Sys.argv > 1 then Sys.argv.(1) else "" in
  if not (List.mem combo ["relaxed-only"; "new-only"; "both"]) then (
    prerr_endline "usage: test_matrix_optin (relaxed-only|new-only|both)" ;
    exit 2) ;
  (* The environment may say SAREK_MATRIX_REWRITE=off: the mode is not under test. *)
  Spec.set_mode Spec.Auto ;
  if combo <> "new-only" then
    Sarek_coopmat.accept_relaxed_accumulation ~reason:"test_matrix_optin" ;
  if combo <> "relaxed-only" then
    Sarek_coopmat.accept_automatic_float_rewrite ~reason:"test_matrix_optin" ;
  match
    List.filter
      (fun d -> d.Device.framework = "HIP" && offers_int d)
      (Array.to_list (Device.init ()))
  with
  | [] -> print_endline "[SKIP] no gfx11 HIP device"
  | devices ->
      List.iter (run_case combo) devices ;
      if !failures > 0 then (
        Printf.printf "%d failure(s)\n" !failures ;
        exit 1)
      else print_endline "PASS"
