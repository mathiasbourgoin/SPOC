(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* f16 x f16 -> f32 GEMM through SCoopmat on Metal (simdgroup_matrix 8x8x8,
   Sarek_gemm.Coopmat ~frag:8) against the scalar f32 tiled
   Sarek_gemm.sgemm_tiled_kernel, square sizes (specs/coopmat-metal-simdgroup.md).

   Method: INTERLEAVED. For each size, [rounds] rounds; in each round every
   variant is timed once, in an order that alternates by round (A B C ... then
   ... C B A), so a slow drift of the machine hits all variants alike. One timed
   sample is a batch of [batch] back-to-back launches followed by one
   [Transfer.flush] (wall clock, Unix.gettimeofday), reported per launch.
   Reported: median and min-max over the rounds, in microseconds and GFLOP/s
   (2*n^3 flops). The machine is shared, so run it with the load recorded
   alongside (uptime / ps) and read the spread as part of the result.
   Information only: no threshold.

   Usage: bench_metal_coopmat_gemm.exe [rounds] [batch] [size ...] *)

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Gemm = Sarek_gemm.Coopmat

let () = Sarek_metal.Metal_plugin.init ()

let arg i default =
  if Array.length Sys.argv > i then int_of_string Sys.argv.(i) else default

let rounds = arg 1 7

let batch = arg 2 5

let sizes =
  if Array.length Sys.argv > 3 then
    List.map
      int_of_string
      (List.tl (List.tl (List.tl (Array.to_list Sys.argv))))
  else [512; 1024; 2048]

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a ;
  let n = Array.length a in
  if n mod 2 = 1 then a.(n / 2) else (a.((n / 2) - 1) +. a.(n / 2)) /. 2.0

let f16_vec len =
  let v = Vector.create Vector.float16 len in
  for i = 0 to len - 1 do
    Vector.set v i (float_of_int ((i * 7919 mod 17) - 8) /. 16.0)
  done ;
  v

let f32_vec len =
  let v = Vector.create Vector.float32 len in
  for i = 0 to len - 1 do
    Vector.set v i (float_of_int ((i * 104729 mod 13) - 6) /. 16.0)
  done ;
  v

(** One timed sample: microseconds per launch over a drained batch. *)
let sample dev launch =
  let t0 = Unix.gettimeofday () in
  for _ = 1 to batch do
    launch ()
  done ;
  Transfer.flush dev ;
  (Unix.gettimeofday () -. t0) *. 1e6 /. float_of_int batch

type variant = {name : string; launch : unit -> unit}

let coopmat_variant dev ~n tiling =
  let a = f16_vec (n * n) and b = f16_vec (n * n) in
  let c = f32_vec (n * n) and d = f32_vec (n * n) in
  {
    name =
      Printf.sprintf
        "coopmat f16->f32 %dx%d/w%d"
        tiling.Gemm.frags_m
        tiling.frags_n
        tiling.waves;
    launch =
      (fun () ->
        Gemm.run ~device:dev ~frag:8 ~tiling ~a ~b ~c ~d ~m:n ~n ~k:n ());
  }

let scalar_ir =
  match (snd Sarek_gemm.sgemm_tiled_kernel).Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "Sarek_gemm.sgemm_tiled_kernel has no IR"

let scalar_variant dev ~n =
  let a = f32_vec (n * n) and b = f32_vec (n * n) and c = f32_vec (n * n) in
  let i x = Execute.Int32 (Int32.of_int x) in
  {
    name = "scalar f32 Sarek_gemm 16x16";
    launch =
      (fun () ->
        Execute.run_vectors
          ~device:dev
          ~ir:scalar_ir
          ~args:
            Execute.
              [Vec a; Vec b; Vec c; i n; i n; i n; Float32 1.0; Float32 0.0]
          ~block:(Sarek_gemm.Host.block ())
          ~grid:(Sarek_gemm.Host.grid ~m:n ~n)
          ());
  }

let tilings =
  Gemm.
    [
      {frags_m = 1; frags_n = 1; waves = 4};
      {frags_m = 2; frags_n = 2; waves = 4};
      {frags_m = 2; frags_n = 2; waves = 2};
      {frags_m = 4; frags_n = 4; waves = 1};
    ]

let run_size dev n =
  let variants =
    List.map (coopmat_variant dev ~n) tilings @ [scalar_variant dev ~n]
  in
  List.iter
    (fun v ->
      v.launch () ;
      Transfer.flush dev)
    variants ;
  let results = List.map (fun v -> (v, ref [])) variants in
  for r = 1 to rounds do
    let order = if r mod 2 = 1 then results else List.rev results in
    List.iter (fun (v, acc) -> acc := sample dev v.launch :: !acc) order
  done ;
  List.iter
    (fun (v, acc) ->
      let med = median !acc in
      let gflops us = 2.0 *. (float n ** 3.0) /. (us *. 1e-6) /. 1e9 in
      Printf.printf
        "%-30s n=%-5d median %10.1f us (%8.1f GFLOP/s)  min %10.1f max %10.1f \
         us  (%d rounds x batch %d)\n\
         %!"
        v.name
        n
        med
        (gflops med)
        (List.fold_left min infinity !acc)
        (List.fold_left max 0.0 !acc)
        rounds
        batch)
    results

let () =
  match
    List.filter
      (fun d ->
        d.Device.framework = "Metal"
        && Sarek_capability.permits
             (Sarek_coopmat.verdict
                ~support:
                  d.Device.capabilities.Spoc_framework.Framework_sig.coopmat
                (Gemm.config_for ~frag:8)))
      (Array.to_list (Device.init ()))
  with
  | [] -> print_endline "[SKIP] no Metal device offering simdgroup_matrix"
  | dev :: _ ->
      Sarek_coopmat.accept_relaxed_accumulation
        ~reason:"benchmark: throughput, not accuracy, is measured here" ;
      Printf.printf
        "device: %s; rounds=%d batch=%d\n%!"
        dev.Device.name
        rounds
        batch ;
      List.iter (run_size dev) sizes
