(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The launch-time matrix rewrite on Metal, OFF against ON
   (specs/matrix-pattern-rewrite.md, "Metal"): a GEMM-shaped f16 x f16 -> f32
   kernel written as a plain nest, launched as written (Execute.run_vectors_plain)
   and rewritten to 8x8x8 simdgroup_matrix fragments (Sarek_matrix.run).

   Method: INTERLEAVED PAIRS. For each size, [rounds] rounds; in each round the
   two variants are timed once each, in an order that alternates by round (off,
   on then on, off), so a slow drift of the machine hits both alike. One timed
   sample is a batch of [batch] back-to-back launches followed by one
   [Transfer.flush] (wall clock), reported per launch. Reported: median and
   min-max over the rounds, in microseconds and GFLOP/s (2*n^3 flops), and the
   ratio of the medians. Record uptime and ps next to the output: the machine is
   shared. Information only: no threshold.

   Usage: bench_metal_matrix_rewrite.exe [rounds] [batch] [size ...] *)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_metal.Metal_plugin.init ()

(* Test setup: delete this line once Metal's unexecuted-lowering opt-in is gone. *)
let () =
  Sarek_metal.Sarek_ir_metal.accept_unexecuted_lowering
    ~reason:"benchmark of the lowering under evaluation"

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

let kernel_ir =
  match
    (snd
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
           end])
      .Kirc_types.body_ir
  with
  | Some ir -> ir
  | None -> failwith "the benchmark kernel has no IR"

let f16_vec len =
  let v = Vector.create Vector.float16 len in
  for i = 0 to len - 1 do
    Vector.set v i (float_of_int ((i * 7919 mod 17) - 8) /. 16.0)
  done ;
  v

let sample dev launch =
  let t0 = Unix.gettimeofday () in
  for _ = 1 to batch do
    launch ()
  done ;
  Transfer.flush dev ;
  (Unix.gettimeofday () -. t0) *. 1e6 /. float_of_int batch

let run_size dev n =
  let a = f16_vec (n * n) and b = f16_vec (n * n) in
  let c = Vector.create Vector.float32 (n * n) in
  let i x = Execute.Int32 (Int32.of_int x) in
  let args = Execute.[Vec a; Vec b; Vec c; i n; i n; i n] in
  let block = Execute.dims3d 16 16 1 in
  let grid = Execute.dims3d ((n + 15) / 16) ((n + 15) / 16) 1 in
  let off () =
    Execute.run_vectors_plain ~device:dev ~ir:kernel_ir ~args ~block ~grid ()
  in
  let on () =
    match Sarek_matrix.run ~device:dev ~ir:kernel_ir ~args ~block ~grid () with
    | Sarek_matrix.Rewritten _ -> ()
    | Sarek_matrix.Original why ->
        failwith ("the rewrite was not applied: " ^ why)
  in
  List.iter
    (fun launch ->
      launch () ;
      Transfer.flush dev)
    [off; on] ;
  let off_us = ref [] and on_us = ref [] in
  for r = 1 to rounds do
    if r mod 2 = 1 then (
      off_us := sample dev off :: !off_us ;
      on_us := sample dev on :: !on_us)
    else (
      on_us := sample dev on :: !on_us ;
      off_us := sample dev off :: !off_us)
  done ;
  let gflops us = 2.0 *. (float n ** 3.0) /. (us *. 1e-6) /. 1e9 in
  let report name xs =
    Printf.printf
      "%-22s n=%-5d median %10.1f us (%8.1f GFLOP/s)  min %10.1f max %10.1f us\n\
       %!"
      name
      n
      (median xs)
      (gflops (median xs))
      (List.fold_left min infinity xs)
      (List.fold_left max 0.0 xs)
  in
  report "rewrite off" !off_us ;
  report "rewrite on (8x8x8)" !on_us ;
  Printf.printf
    "n=%-5d on/off speed-up %.2fx (ratio of medians; %d pairs, batch %d)\n%!"
    n
    (median !off_us /. median !on_us)
    rounds
    batch

let () =
  match
    List.filter
      (fun d -> d.Device.framework = "Metal")
      (Array.to_list (Device.init ()))
  with
  | [] -> print_endline "[SKIP] no Metal device"
  | dev :: _ ->
      Sarek_coopmat.accept_relaxed_accumulation
        ~reason:"benchmark: throughput, not accuracy, is measured here" ;
      Printf.printf
        "device: %s; pairs=%d batch=%d\n%!"
        dev.Device.name
        rounds
        batch ;
      List.iter (run_size dev) sizes
