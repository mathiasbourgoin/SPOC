(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Benchmark: 2D Jacobi / 5-Point Stencil
 *
 * Priority: P1 (Sprint 3)
 * Pattern: 5-point stencil computation (up, down, left, right, center)
 * Category: Scientific Computing / Stencil / PDE Solver
 *
 * Description:
 * Implements 2D Jacobi iteration for heat diffusion / Laplace equation.
 * Each cell is updated as the average of its 4 neighbors (5-point stencil).
 * Classic stencil pattern for PDE solvers and image processing.
 *
 * Performance Notes:
 * - Memory bandwidth bound: 5 reads, 1 write per cell
 * - Grid size N×N, processes N² cells
 * - Can be optimized with shared memory tiling
 * - Arithmetic intensity: 4 adds + 1 div = 5 FLOPs, 6 memory ops
 * - Performance: cells/sec, GB/s
 *
 * Hardware Tested:
 * - Intel Arc GPU (OpenCL + Vulkan)
 * - Intel Core CPU (OpenCL)
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Std = Sarek_stdlib.Std
open Benchmark_backends

(** Pure OCaml baseline: 2D Jacobi stencil *)
let cpu_stencil_2d input output width height =
  (* Interior points only, boundaries unchanged *)
  for y = 1 to height - 2 do
    for x = 1 to width - 2 do
      let idx = (y * width) + x in
      let up = input.(((y - 1) * width) + x) in
      let down = input.(((y + 1) * width) + x) in
      let left = input.((y * width) + x - 1) in
      let right = input.((y * width) + x + 1) in
      let center = input.(idx) in
      output.(idx) <- (up +. down +. left +. right +. center) /. 5.0
    done
  done

(** Sarek kernel: 2D 5-point stencil *)
let stencil_2d_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let open Std in
      let x = thread_idx_x + (block_idx_x * block_dim_x) in
      let y = thread_idx_y + (block_idx_y * block_dim_y) in
      if x > 0 && x < width - 1 && y > 0 && y < height - 1 then
        let idx = (y * width) + x in
        let up = input.(((y - 1) * width) + x) in
        let down = input.(((y + 1) * width) + x) in
        let left = input.((y * width) + x - 1) in
        let right = input.((y * width) + x + 1) in
        let center = input.(idx) in
        output.(idx) <- (up +. down +. left +. right +. center) /. 5.0]

(** Run 2D stencil benchmark on specified device *)
let run_stencil_2d_benchmark device backend_name size =
  (* Use square grid *)
  let width = size in
  let height = size in
  let n = width * height in

  let _, kirc = stencil_2d_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  (* Create test data: sin wave pattern *)
  let input_data =
    Array.init n (fun i ->
        let x = i mod width in
        let y = i / width in
        sin (float_of_int x *. 0.1) *. cos (float_of_int y *. 0.1))
  in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Vector.set input i input_data.(i) ;
    Vector.set output i 0.0
  done ;

  (* Launch configuration: 2D grid *)
  let block_x = 16 in
  let block_y = 16 in
  let grid_x = (width + block_x - 1) / block_x in
  let grid_y = (height + block_y - 1) / block_y in
  let block = Sarek.Execute.dims2d block_x block_y in
  let grid = Sarek.Execute.dims2d grid_x grid_y in

  (* Warmup run *)
  Sarek.Execute.run_vectors
    ~device
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int32 (Int32.of_int width);
        Sarek.Execute.Int32 (Int32.of_int height);
      ]
    ~block
    ~grid
    () ;
  Transfer.flush device ;

  (* Timed runs *)
  let num_runs = 100 in
  let times = ref [] in

  for _ = 1 to num_runs do
    let t0 = Unix.gettimeofday () in
    Sarek.Execute.run_vectors
      ~device
      ~ir
      ~args:
        [
          Sarek.Execute.Vec input;
          Sarek.Execute.Vec output;
          Sarek.Execute.Int32 (Int32.of_int width);
          Sarek.Execute.Int32 (Int32.of_int height);
        ]
      ~block
      ~grid
      () ;
    Transfer.flush device ;
    let t1 = Unix.gettimeofday () in
    times := (t1 -. t0) :: !times
  done ;

  (* Compute statistics *)
  let sorted_times = List.sort compare !times in
  let avg_time =
    List.fold_left ( +. ) 0.0 sorted_times /. float_of_int num_runs
  in
  let min_time = List.hd sorted_times in
  let median_time = List.nth sorted_times (num_runs / 2) in

  (* Verify correctness *)
  let gpu_result = Vector.to_array output in
  let cpu_result = Array.copy input_data in
  cpu_stencil_2d input_data cpu_result width height ;

  let errors = ref 0 in
  let tolerance = 0.0001 in
  for y = 1 to height - 2 do
    for x = 1 to width - 2 do
      let idx = (y * width) + x in
      let diff = abs_float (gpu_result.(idx) -. cpu_result.(idx)) in
      if diff > tolerance then begin
        if !errors < 5 then
          Printf.printf
            "  ERROR at (%d,%d): GPU=%.6f CPU=%.6f diff=%.6f\n"
            x
            y
            gpu_result.(idx)
            cpu_result.(idx)
            diff ;
        incr errors
      end
    done
  done ;

  (* Report results *)
  Printf.printf "\n=== 2D Jacobi / 5-Point Stencil Benchmark ===\n" ;
  Printf.printf "Backend: %s\n" backend_name ;
  Printf.printf "Grid size: %d×%d (%d cells)\n" width height n ;
  Printf.printf
    "Interior points: %d×%d (%d cells)\n"
    (width - 2)
    (height - 2)
    ((width - 2) * (height - 2)) ;
  Printf.printf "Block size: %d×%d\n" block_x block_y ;
  Printf.printf "Grid size: %d×%d blocks\n" grid_x grid_y ;
  Printf.printf "\nTiming (%d runs):\n" num_runs ;
  Printf.printf "  Min:    %.3f ms\n" (min_time *. 1000.0) ;
  Printf.printf "  Median: %.3f ms\n" (median_time *. 1000.0) ;
  Printf.printf "  Mean:   %.3f ms\n" (avg_time *. 1000.0) ;

  let throughput_mcells = float_of_int n /. (median_time *. 1e6) in
  Printf.printf "  Throughput: %.2f M cells/s\n" throughput_mcells ;

  (* Bandwidth: 5 reads + 1 write = 6 floats per cell = 24 bytes *)
  let bandwidth_gb = float_of_int n *. 24.0 /. (median_time *. 1e9) in
  Printf.printf "  Bandwidth: %.2f GB/s\n" bandwidth_gb ;

  Printf.printf "\nVerification: " ;
  if !errors = 0 then
    Printf.printf
      "PASS ✓ (all %d interior cells correct)\n"
      ((width - 2) * (height - 2))
  else Printf.printf "FAIL ✗ (%d errors)\n" !errors ;

  Printf.printf
    "Sample output (center): %.6f\n"
    gpu_result.((height / 2 * width) + (width / 2)) ;
  Printf.printf "============================================\n" ;

  !errors = 0

(** Main benchmark runner *)
let () =
  Printf.printf "2D Jacobi / 5-Point Stencil Benchmark\n" ;
  Printf.printf "Heat diffusion / Laplace equation solver\n\n" ;

  let size =
    if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 512
  in

  (* Initialize backends *)
  Backend_loader.init () ;

  (* Initialize SPOC and discover devices *)
  let devices = Device.init () in

  if Array.length devices = 0 then begin
    Printf.printf "No GPU devices found. Cannot run benchmark.\n" ;
    exit 1
  end ;

  (* Run on first available device *)
  let device = devices.(0) in
  let backend_name = device.Device.framework in

  Printf.printf "Using device: %s (%s)\n\n" device.Device.name backend_name ;

  let success = run_stencil_2d_benchmark device backend_name size in
  exit (if success then 0 else 1)
