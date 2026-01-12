(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Benchmark: Radix Sort
 *
 * Priority: P1 (Sprint 2)
 * Pattern: Histogram-based sorting with multiple passes
 * Category: Sorting algorithm
 *
 * Description:
 * Implements radix sort using 4-bit digits (16 bins per pass). Radix sort
 * processes integers digit-by-digit from least to most significant, using
 * counting sort for each digit. This implementation requires 8 passes for
 * 32-bit integers (32 bits / 4 bits per pass).
 *
 * Performance Notes:
 * - Complexity: O(k*n) where k is number of passes
 * - Uses histogram and prefix sum as building blocks
 * - Stable sort: preserves relative order of equal elements
 * - Memory: requires temporary buffer for ping-pong between passes
 * - Good for uniformly distributed integer data
 *
 * Hardware Tested:
 * - Intel Arc GPU (OpenCL + Vulkan)
 * - Intel Core CPU (OpenCL)
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Std = Sarek_stdlib.Std
module Gpu = Sarek_stdlib.Gpu
open Benchmark_common
open Benchmark_runner

(** Pure OCaml baseline: radix sort *)
let cpu_radix_sort input n =
  let bits_per_pass = 4 in
  let num_bins = 1 lsl bits_per_pass in
  let num_passes = 32 / bits_per_pass in
  let mask = Int32.of_int (num_bins - 1) in

  let src = Array.copy input in
  let dst = Array.make n 0l in

  for pass = 0 to num_passes - 1 do
    let shift = pass * bits_per_pass in

    (* Count occurrences of each digit *)
    let counts = Array.make num_bins 0 in
    for i = 0 to n - 1 do
      let digit =
        Int32.to_int
          (Int32.logand (Int32.shift_right_logical src.(i) shift) mask)
      in
      counts.(digit) <- counts.(digit) + 1
    done ;

    (* Compute prefix sum for starting positions *)
    let positions = Array.make num_bins 0 in
    for i = 1 to num_bins - 1 do
      positions.(i) <- positions.(i - 1) + counts.(i - 1)
    done ;

    (* Distribute elements to their positions *)
    for i = 0 to n - 1 do
      let digit =
        Int32.to_int
          (Int32.logand (Int32.shift_right_logical src.(i) shift) mask)
      in
      dst.(positions.(digit)) <- src.(i) ;
      positions.(digit) <- positions.(digit) + 1
    done ;

    (* Swap buffers *)
    Array.blit dst 0 src 0 n
  done ;

  src

(** Sarek kernel: Histogram for one radix pass *)
let radix_histogram_kernel =
  [%kernel
    fun (input : int32 vector)
        (histogram : int32 vector)
        (n : int32)
        (shift : int32)
        (mask : int32) ->
      let open Std in
      let open Gpu in
      let%shared (local_hist : int32) = 16l in
      let tid = thread_idx_x in
      let gid = global_thread_id in
      let num_bins = 16l in
      (* Initialize local histogram *)
      let%superstep init = if tid < num_bins then local_hist.(tid) <- 0l in
      (* Count in local histogram *)
      let%superstep[@divergent] count =
        if gid < n then begin
          let value = input.(gid) in
          let digit = (value lsr shift) land mask in
          let _old = atomic_add_int32 local_hist digit 1l in
          ()
        end
      in
      (* Merge to global histogram *)
      let%superstep[@divergent] merge =
        if tid < num_bins then begin
          let _old = atomic_add_global_int32 histogram tid local_hist.(tid) in
          ()
        end
      in
      ()]
[@@warning "-33"]

(** Sarek kernel: Scatter elements based on prefix sum *)
let radix_scatter_kernel =
  [%kernel
    fun (input : int32 vector)
        (output : int32 vector)
        (counters : int32 vector)
        (n : int32)
        (shift : int32)
        (mask : int32) ->
      let open Std in
      let open Gpu in
      let gid = global_thread_id in
      if gid < n then begin
        let value = input.(gid) in
        let digit = (value lsr shift) land mask in
        (* Atomically get and increment counter for this digit *)
        let pos = atomic_add_global_int32 counters digit 1l in
        output.(pos) <- value
      end]
[@@warning "-33"]

(** Compute prefix sum on CPU (helper) *)
let cpu_prefix_sum arr n =
  let result = Array.make n 0l in
  if n > 0 then begin
    result.(0) <- 0l ;
    for i = 1 to n - 1 do
      result.(i) <- Int32.add result.(i - 1) arr.(i - 1)
    done
  end ;
  result

(** Run radix sort benchmark on specified device *)
let run_radix_sort_benchmark ~device ~size ~config =
  let n = size in
  let bits_per_pass = 4 in
  let num_bins = 1 lsl bits_per_pass in
  let num_passes = 32 / bits_per_pass in
  let mask = Int32.of_int (num_bins - 1) in

  let _, hist_kirc = radix_histogram_kernel in
  let hist_ir =
    match hist_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Histogram kernel has no IR"
  in

  let _, scatter_kirc = radix_scatter_kernel in
  let scatter_ir =
    match scatter_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Scatter kernel has no IR"
  in

  (* Launch configuration *)
  let block_size = 256 in
  let num_blocks = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d num_blocks in

  (* Prepare host data for verification *)
  Random.init 42 ;
  let input_arr = Array.init n (fun _ -> Int32.of_int (Random.int 100000)) in
  Printf.printf
    "DEBUG input_arr[0..4]: %ld %ld %ld %ld %ld\n"
    input_arr.(0)
    input_arr.(1)
    input_arr.(2)
    input_arr.(3)
    input_arr.(4) ;
  let cpu_result = cpu_radix_sort input_arr n in
  Printf.printf
    "DEBUG cpu_result[0..4] after sort: %ld %ld %ld %ld %ld\n"
    cpu_result.(0)
    cpu_result.(1)
    cpu_result.(2)
    cpu_result.(3)
    cpu_result.(4) ;

  (* init: Create fresh vectors from host data *)
  let init () =
    let input = Vector.create Vector.int32 n in
    let output = Vector.create Vector.int32 n in
    Array.iteri (fun i x -> Vector.set input i x) input_arr ;
    (* Create reusable histogram and counters vectors *)
    let histogram = Vector.create Vector.int32 num_bins in
    let counters = Vector.create Vector.int32 num_bins in
    (input, output, histogram, counters)
  in

  (* compute: Run all radix passes *)
  let compute (input, output, histogram, counters) =
    let current_input = ref input in
    let current_output = ref output in

    for pass = 0 to num_passes - 1 do
      let shift = Int32.of_int (pass * bits_per_pass) in

      (* Reset histogram for this pass *)
      for i = 0 to num_bins - 1 do
        Vector.set histogram i 0l
      done ;

      (* Compute histogram *)
      Sarek.Execute.run_vectors
        ~device
        ~ir:hist_ir
        ~args:
          [
            Sarek.Execute.Vec !current_input;
            Sarek.Execute.Vec histogram;
            Sarek.Execute.Int32 (Int32.of_int n);
            Sarek.Execute.Int32 shift;
            Sarek.Execute.Int32 mask;
          ]
        ~block
        ~grid
        () ;
      Device.synchronize device ;

      (* Compute prefix sum on CPU - histogram must be transferred back *)
      let hist_arr = Vector.to_array histogram in
      let prefix_arr = cpu_prefix_sum hist_arr num_bins in
      for i = 0 to num_bins - 1 do
        Vector.set counters i prefix_arr.(i)
      done ;

      (* Scatter elements *)
      Sarek.Execute.run_vectors
        ~device
        ~ir:scatter_ir
        ~args:
          [
            Sarek.Execute.Vec !current_input;
            Sarek.Execute.Vec !current_output;
            Sarek.Execute.Vec counters;
            Sarek.Execute.Int32 (Int32.of_int n);
            Sarek.Execute.Int32 shift;
            Sarek.Execute.Int32 mask;
          ]
        ~block
        ~grid
        () ;
      Device.synchronize device ;

      (* Swap buffers for next pass *)
      if pass < num_passes - 1 then begin
        let temp = !current_input in
        current_input := !current_output ;
        current_output := temp
      end
    done
    (* After all passes without final swap, result is in current_output *)
  in

  (* verify: Check correctness *)
  let verify (input, output, _histogram, _counters) =
    (* Ensure all GPU work is complete and data is back on host *)
    Transfer.flush device ;
    Device.synchronize device ;

    (* Read BOTH buffers to see which has the sorted result *)
    let input_arr = Vector.to_array input in
    let output_arr = Vector.to_array output in

    (* Debug output *)
    Printf.printf
      "DEBUG input[0..4]:  %ld %ld %ld %ld %ld\n"
      input_arr.(0)
      input_arr.(1)
      input_arr.(2)
      input_arr.(3)
      input_arr.(4) ;
    Printf.printf
      "DEBUG output[0..4]: %ld %ld %ld %ld %ld\n"
      output_arr.(0)
      output_arr.(1)
      output_arr.(2)
      output_arr.(3)
      output_arr.(4) ;
    Printf.printf
      "DEBUG cpu[0..4]:    %ld %ld %ld %ld %ld\n"
      cpu_result.(0)
      cpu_result.(1)
      cpu_result.(2)
      cpu_result.(3)
      cpu_result.(4) ;
    Printf.printf
      "DEBUG input[n-5..]: %ld %ld %ld %ld %ld\n"
      input_arr.(n - 5)
      input_arr.(n - 4)
      input_arr.(n - 3)
      input_arr.(n - 2)
      input_arr.(n - 1) ;
    Printf.printf
      "DEBUG output[n-5..]: %ld %ld %ld %ld %ld\n"
      output_arr.(n - 5)
      output_arr.(n - 4)
      output_arr.(n - 3)
      output_arr.(n - 2)
      output_arr.(n - 1) ;
    Printf.printf
      "DEBUG cpu[n-5..]:   %ld %ld %ld %ld %ld\n"
      cpu_result.(n - 5)
      cpu_result.(n - 4)
      cpu_result.(n - 3)
      cpu_result.(n - 2)
      cpu_result.(n - 1) ;

    (* Use whichever buffer looks sorted - check last element *)
    let gpu_result =
      if
        output_arr.(n - 1) >= output_arr.(0)
        && input_arr.(n - 1) < input_arr.(0)
      then output_arr
      else input_arr
    in

    let errors = ref 0 in
    for i = 0 to n - 1 do
      if gpu_result.(i) <> cpu_result.(i) then begin
        if !errors < 5 then
          Printf.printf
            "  ERROR at %d: GPU=%ld CPU=%ld\n"
            i
            gpu_result.(i)
            cpu_result.(i) ;
        incr errors
      end
    done ;

    (* Check if sorted *)
    let is_sorted = ref true in
    for i = 1 to n - 1 do
      if gpu_result.(i) < gpu_result.(i - 1) then is_sorted := false
    done ;

    !errors = 0 && !is_sorted
  in

  (* Run benchmark using Common infrastructure *)
  let times, verified =
    Common.benchmark_gpu
      ~dev:device
      ~warmup:config.warmup
      ~iterations:config.iterations
      ~init
      ~compute
      ~verify
  in

  let times_array = times in
  let median_ms = Common.median times_array in

  (* Report results *)
  Printf.printf
    "  %s: size=%d, median=%.3f ms, verified=%s\n"
    device.Device.name
    n
    median_ms
    (if verified then "✓" else "✗") ;

  let throughput_melems = float_of_int n /. (median_ms /. 1000.0 *. 1e6) in

  Output.
    {
      device_id = device.Device.id;
      device_name = device.Device.name;
      framework = device.Device.framework;
      iterations = times_array;
      mean_ms = Common.mean times_array;
      stddev_ms = Common.stddev times_array;
      median_ms;
      min_ms = Common.min times_array;
      max_ms = Common.max times_array;
      throughput = Some throughput_melems;
      verified = Some verified;
    }

(** Main benchmark runner *)
let () =
  let config =
    Benchmark_runner.parse_args
      ~benchmark_name:"radix_sort"
      ~default_sizes:[1_000_000; 10_000_000; 50_000_000]
      ()
  in
  Benchmark_runner.run_benchmark
    ~benchmark_name:"radix_sort"
    ~config
    ~run_fn:run_radix_sort_benchmark
