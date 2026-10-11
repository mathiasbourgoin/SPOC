(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The rewrite templates are parameterised by the subgroup width
 * (specs/matrix-pattern-rewrite.md, "Subgroup width"). On the interpreter, the
 * oracle, the int32 byte-plane rewrite is laid out for 32 and for 64 lanes
 * (four waves: blocks of 128 and 256 threads) and must equal the original
 * kernel and the host reference bit for bit, for every case and shape of the
 * rewrite corpus. The launch the rewrite chose is read back from the report,
 * so a width that was ignored shows as a wrong block size.
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@alert "-test_only"]

[@@@warning "-33"]

open Sarek
open Matrix_rewrite_support
module Spec = Sarek_execute.Matrix_specialise
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let shapes = [(1, 1, 1); (17, 33, 5); (64, 64, 64); (40, 24, 20); (33, 65, 17)]

let last_block = ref None

let () =
  Spec.set_reporter
    (Some
       (fun r ->
         match r.Spec.outcome with
         | Spec.Specialised s -> last_block := Some s.Spec.block
         | Spec.Original _ -> last_block := None))

let run_case dev ~subgroup case (m, n, k) =
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
    last_block := None ;
    let outcome = launch ~rewrite dev case.kernel ~args ~block ~grid in
    Transfer.flush dev ;
    (Vector.to_array cv, outcome)
  in
  let original, _ = go ~rewrite:false in
  let rewritten, outcome = go ~rewrite:true in
  let what = Printf.sprintf "%s %dx%dx%d, %d lanes" case.name m n k subgroup in
  (match (outcome, !last_block) with
  | Sarek_matrix.Rewritten _, Some (bx, 1, 1) when bx = 4 * subgroup -> ()
  | Sarek_matrix.Rewritten _, other ->
      fail
        "%s: block %s, expected %d threads"
        what
        (match other with
        | Some (x, y, z) -> Printf.sprintf "%dx%dx%d" x y z
        | None -> "unreported")
        (4 * subgroup)
  | Sarek_matrix.Original why, _ -> fail "%s: not rewritten (%s)" what why) ;
  let same =
    compare_arrays (what ^ " rewrite vs original") original rewritten
  in
  let init x = if case.batched then x else 0l in
  let epi x = if case.batched then Int32.add (Int32.mul alpha x) 3l else x in
  let host_ok =
    (not case.full)
    || compare_arrays
         (what ^ " original vs host")
         (reference ~layout:case.layout ~m ~n ~k ~nb ~init ~epi a b c0)
         original
  in
  if same && host_ok then ok "%s: bit-identical" what

let () =
  Random.init 20261012 ;
  let interpreters =
    List.filter
      (fun d -> d.Device.framework = "Interpreter")
      (Array.to_list (Device.init ()))
  in
  (* The sequential interpreter only: the parallel one adds nothing here. *)
  let dev = List.hd interpreters in
  List.iter
    (fun subgroup ->
      Spec.subgroup_override_for_tests (Some subgroup) ;
      Spec.clear () ;
      List.iter
        (fun case -> List.iter (run_case dev ~subgroup case) shapes)
        cases)
    [32; 64] ;
  Spec.subgroup_override_for_tests None ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
