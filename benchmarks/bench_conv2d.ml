(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Benchmark: 2D Convolution
 *
 * Priority: P1 (Sprint 3)
 * Pattern: 2D image filtering with 3×3 kernel
 * Category: Image Processing / Stencil
 *
 * Description:
 * Implements 2D convolution for image filtering using 3×3 kernel.
 * Common operation in image processing, computer vision, and CNNs.
 * Tests memory access patterns and arithmetic intensity.
 *
 * Performance Notes:
 * - Memory: 9 reads + 1 write per pixel for 3×3 kernel
 * - Arithmetic: 9 multiplies + 8 adds = 17 FLOPs per pixel
 * - Can be optimized with shared memory tiling
 * - Separable kernels can reduce to 2*(3 reads + 3 muls + 2 adds)
 * - Performance: pixels/sec, GB/s
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

(** Pure OCaml baseline: 2D convolution with 3×3 kernel *)
let cpu_conv2d input output width height kernel =
  (* Interior points only, boundaries set to 0 *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let idx = (y * width) + x in
      if x = 0 || x = width - 1 || y = 0 || y = height - 1 then
        output.(idx) <- 0.0
      else
        let sum = ref 0.0 in
        for ky = 0 to 2 do
          for kx = 0 to 2 do
            let px = x + kx - 1 in
            let py = y + ky - 1 in
            let pidx = (py * width) + px in
            let kidx = (ky * 3) + kx in
            sum := !sum +. (input.(pidx) *. kernel.(kidx))
          done
        done ;
        output.(idx) <- !sum
    done
  done

(** Sarek kernel: 2D convolution with hardcoded 3×3 box blur *)
let conv2d_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let open Std in
      let tid = global_thread_id in
      let n = width * height in
      if tid < n then
        let x = tid mod width in
        let y = tid / width in
        if x = 0 || x = width - 1 || y = 0 || y = height - 1 then
          output.(tid) <- 0.0
        else
          (* Unroll 3×3 box blur (all coefficients = 1/9) *)
          let p00 = input.(((y - 1) * width) + x - 1) in
          let p01 = input.(((y - 1) * width) + x) in
          let p02 = input.(((y - 1) * width) + x + 1) in
          let p10 = input.((y * width) + x - 1) in
          let p11 = input.((y * width) + x) in
          let p12 = input.((y * width) + x + 1) in
          let p20 = input.(((y + 1) * width) + x - 1) in
          let p21 = input.(((y + 1) * width) + x) in
          let p22 = input.(((y + 1) * width) + x + 1) in
          let sum =
            p00 +. p01 +. p02 +. p10 +. p11 +. p12 +. p20 +. p21 +. p22
          in
          output.(tid) <- sum /. 9.0]

(** Run 2D convolution benchmark on specified device *)
let run_conv2d_benchmark device backend_name size =
  (* Use square image *)
  let width = size in
  let height = size in
  let n = width * height in

  let _, kirc = conv2d_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  (* Create test image: gradient pattern *)
  let input_data =
    Array.init n (fun i ->
        let x = i mod width in
        let y = i / width in
        float_of_int ((x + y) mod 256) /. 255.0)
  in

  (* Box blur kernel (normalized) *)
  let kernel_data = Array.make 9 (1.0 /. 9.0) in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Vector.set input i input_data.(i) ;
    Vector.set output i 0.0
  done ;

  (* Launch configuration: 1D grid *)
  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

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
  let cpu_result = Array.make n 0.0 in
  cpu_conv2d input_data cpu_result width height kernel_data ;

  let errors = ref 0 in
  let tolerance = 0.001 in
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
  Printf.printf "\n=== 2D Convolution Benchmark ===\n" ;
  Printf.printf "Backend: %s\n" backend_name ;
  Printf.printf "Image size: %d×%d (%d pixels)\n" width height n ;
  Printf.printf "Kernel size: 3×3 (box blur)\n" ;
  Printf.printf
    "Interior pixels: %d×%d (%d pixels)\n"
    (width - 2)
    (height - 2)
    ((width - 2) * (height - 2)) ;
  Printf.printf "Block size: %d\n" block_size ;
  Printf.printf "Grid size: %d blocks\n" grid_size ;
  Printf.printf "\nTiming (%d runs):\n" num_runs ;
  Printf.printf "  Min:    %.3f ms\n" (min_time *. 1000.0) ;
  Printf.printf "  Median: %.3f ms\n" (median_time *. 1000.0) ;
  Printf.printf "  Mean:   %.3f ms\n" (avg_time *. 1000.0) ;

  let throughput_mpixels = float_of_int n /. (median_time *. 1e6) in
  Printf.printf "  Throughput: %.2f M pixels/s\n" throughput_mpixels ;

  (* Bandwidth: 9 reads + 1 write = 10 floats per pixel = 40 bytes *)
  let bandwidth_gb = float_of_int n *. 40.0 /. (median_time *. 1e9) in
  Printf.printf "  Bandwidth: %.2f GB/s\n" bandwidth_gb ;

  (* FLOPs: 9 muls + 8 adds = 17 ops per pixel *)
  let flops = float_of_int n *. 17.0 in
  let gflops = flops /. (median_time *. 1e9) in
  Printf.printf "  Performance: %.2f GFLOPS\n" gflops ;

  Printf.printf "\nVerification: " ;
  if !errors = 0 then
    Printf.printf
      "PASS ✓ (all %d interior pixels correct)\n"
      ((width - 2) * (height - 2))
  else Printf.printf "FAIL ✗ (%d errors)\n" !errors ;

  Printf.printf
    "Sample output (center): %.6f\n"
    gpu_result.((height / 2 * width) + (width / 2)) ;
  Printf.printf "================================\n" ;

  !errors = 0

(** Main benchmark runner *)
let () =
  Printf.printf "2D Convolution Benchmark\n" ;
  Printf.printf "3×3 box blur filter\n\n" ;

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

  let success = run_conv2d_benchmark device backend_name size in
  exit (if success then 0 else 1)
