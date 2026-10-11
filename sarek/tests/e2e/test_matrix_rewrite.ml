(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Matrix-product rewrite, int32 differential (specs/matrix-pattern-rewrite.md).
 *
 * For every kernel and shape: the ORIGINAL kernel and Sarek_matrix.run (the
 * rewrite) are launched with the same arguments and geometry on the same
 * device, from the same initial C (a sentinel, so an output the original
 * leaves alone must stay alone), and compared element by element; both are
 * also compared with a host reference. Bit-identical on the interpreter and
 * on every gfx11 HIP device; every other device must decline (Original).
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
open Matrix_rewrite_support
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let run_case dev ~expect_rewrite case (m, n, k) =
  let nb = if case.batched then 3 else 1 in
  let alpha = 7l in
  let a = Array.init (nb * m * k) (fun _ -> random_int32 ()) in
  let b = Array.init (nb * k * n) (fun _ -> random_int32 ()) in
  let c0 =
    Array.init
      (nb * m * n)
      (fun i ->
        if case.batched then random_int32 ()
        else Int32.add sentinel (Int32.of_int i))
  in
  let block, grid = case.geometry ~m ~n ~nb in
  let go ~rewrite =
    let cv = vec c0 in
    let i x = Execute.Int32 (Int32.of_int x) in
    let args =
      [Execute.Vec (vec a); Execute.Vec (vec b); Execute.Vec cv; i m; i n; i k]
      @ if case.batched then [i nb; Execute.Int32 alpha] else []
    in
    let outcome = launch ~rewrite dev case.kernel ~args ~block ~grid in
    Transfer.flush dev ;
    (Vector.to_array cv, outcome)
  in
  let original, _ = go ~rewrite:false in
  let rewritten, outcome = go ~rewrite:true in
  let what =
    Printf.sprintf "%s %dx%dx%d on %s" case.name m n k dev.Device.name
  in
  let rewrote =
    match outcome with
    | Sarek_matrix.Rewritten _ -> true
    | Sarek_matrix.Original _ -> false
  in
  if rewrote <> expect_rewrite then
    fail
      "%s: expected %s, got %s"
      what
      (if expect_rewrite then "a rewrite" else "the original")
      (Sarek_matrix.outcome_text outcome) ;
  let same =
    compare_arrays (what ^ " rewrite vs original") original rewritten
  in
  let host_ok =
    (not case.full)
    ||
    let init x = if case.batched then x else 0l in
    let epi x = if case.batched then Int32.add (Int32.mul alpha x) 3l else x in
    compare_arrays
      (what ^ " original vs host")
      (reference ~layout:case.layout ~m ~n ~k ~nb ~init ~epi a b c0)
      original
  in
  if same && host_ok && rewrote = expect_rewrite then
    ok "%s: %s, bit-identical" what (Sarek_matrix.outcome_text outcome)

let shapes = [(1, 1, 1); (17, 33, 5); (64, 64, 64); (100, 37, 129)]

(* A kernel that does more than the product is never rewritten: the rewrite
   would drop the other write. *)
let k_extra_write =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (d : int32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0l in
      for i = 0 to k - 1l do
        sum := sum + (a.((row * k) + i) * b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum ;
      d.(col) <- 1l]

let check_extra_write dev =
  let v len = Execute.Vec (vec (Array.make len 1l)) in
  let i x = Execute.Int32 (Int32.of_int x) in
  match
    Sarek_matrix.run
      ~device:dev
      ~ir:(ir_of k_extra_write)
      ~args:[v 256; v 256; v 256; v 16; i 16; i 16]
      ~block:(Execute.dims3d 16 16 1)
      ~grid:(Execute.dims3d 1 1 1)
      ()
  with
  | Sarek_matrix.Original why ->
      ok "extra write on %s: original (%s)" dev.Device.name why
  | Sarek_matrix.Rewritten _ ->
      fail
        "extra write on %s: rewritten, the other write would be lost"
        dev.Device.name

let () =
  Random.init 20261011 ;
  let devices = Array.to_list (Device.init ()) in
  List.iter
    (fun dev ->
      let expect_rewrite = offers_int dev in
      check_extra_write dev ;
      Printf.printf
        "device: %s [%s], rewrite expected: %b\n%!"
        dev.Device.name
        dev.Device.framework
        expect_rewrite ;
      let shapes =
        if dev.Device.framework = "Interpreter" then
          [(1, 1, 1); (17, 33, 5); (40, 24, 20)]
        else shapes
      in
      List.iter
        (fun case -> List.iter (run_case dev ~expect_rewrite case) shapes)
        cases)
    devices ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
