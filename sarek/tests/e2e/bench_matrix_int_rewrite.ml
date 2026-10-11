(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The int32 matrix-product rewrite (specs/matrix-pattern-rewrite.md) on every
   device that rewrites it: the naive DSL GEMM launched as written (A) and
   through Sarek_matrix.run (B), alternated A,B,A,B,... for [pairs] pairs;
   wall time per launch of a batch of [batch] back-to-back launches closed by a
   flush. Median and min-max per launch, the 1-minute load average before and
   after each device. The first pair is checked for bit-identical results.

   Usage: bench_matrix_int_rewrite.exe [size] [pairs] [batch] *)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
open Matrix_rewrite_support
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let arg i default =
  if Array.length Sys.argv > i then int_of_string Sys.argv.(i) else default

let size = arg 1 512

let pairs = arg 2 7

let batch = arg 3 5

let loadavg () =
  try
    In_channel.with_open_text "/proc/loadavg" (fun ic ->
        List.hd (String.split_on_char ' ' (input_line ic)))
  with _ -> "?"

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a ;
  let n = Array.length a in
  if n mod 2 = 1 then a.(n / 2) else (a.((n / 2) - 1) +. a.(n / 2)) /. 2.0

let stats label xs =
  Printf.printf
    "  %-9s median %8.3f ms  min-max %8.3f - %8.3f ms  (n=%d)\n%!"
    label
    (median xs *. 1e3)
    (List.fold_left min infinity xs *. 1e3)
    (List.fold_left max neg_infinity xs *. 1e3)
    (List.length xs)

let bench dev =
  let m = size and n = size and k = size in
  let a = Array.init (m * k) (fun _ -> random_int32 ()) in
  let b = Array.init (k * n) (fun _ -> random_int32 ()) in
  let av = vec a and bv = vec b and cv = vec (Array.make (m * n) 0l) in
  let i x = Execute.Int32 (Int32.of_int x) in
  let args = [Execute.Vec av; Execute.Vec bv; Execute.Vec cv; i m; i n; i k] in
  let block, grid = geo_2d ~m ~n ~nb:1 in
  let ir = ir_of k_2d in
  let once ~rewrite =
    let outcome = launch ~rewrite dev ir ~args ~block ~grid in
    Transfer.flush dev ;
    outcome
  in
  let timed ~rewrite =
    let t0 = Unix.gettimeofday () in
    for _ = 1 to batch do
      ignore (once ~rewrite)
    done ;
    (Unix.gettimeofday () -. t0) /. float_of_int batch
  in
  ignore (once ~rewrite:false) ;
  let reference = Vector.to_array cv in
  let outcome = once ~rewrite:true in
  (match outcome with
  | Sarek_matrix.Rewritten _ ->
      if Vector.to_array cv <> reference then fail "results differ"
  | Sarek_matrix.Original why -> fail "not rewritten: %s" why) ;
  Printf.printf
    "%s [%s] %dx%dx%d int32: %s\n  load before %s\n%!"
    dev.Device.name
    dev.Device.framework
    m
    n
    k
    (Sarek_matrix.outcome_text outcome)
    (loadavg ()) ;
  let off = ref [] and on = ref [] in
  for _ = 1 to pairs do
    off := timed ~rewrite:false :: !off ;
    on := timed ~rewrite:true :: !on
  done ;
  stats "original" !off ;
  stats "rewritten" !on ;
  Printf.printf
    "  speed-up (median) %.2fx; load after %s\n%!"
    (median !off /. median !on)
    (loadavg ())

let () =
  Random.init 20261012 ;
  Array.iter
    (fun dev ->
      if dev.Device.framework <> "Interpreter" && offers_int dev then bench dev)
    (Device.init ()) ;
  if !failures > 0 then exit 1
