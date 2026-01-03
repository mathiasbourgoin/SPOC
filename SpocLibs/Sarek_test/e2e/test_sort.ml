(******************************************************************************
 * E2E test for Sarek PPX - Sorting algorithms
 *
 * Tests bitonic sort and odd-even merge sort - parallel sorting algorithms
 * that work well on GPUs due to their regular communication patterns.
 * V2 runtime only.
 ******************************************************************************)

(* Module aliases *)
module V2_Device = Spoc_core.Device
module V2_Vector = Spoc_core.Vector
module V2_Transfer = Spoc_core.Transfer
module Std = Sarek_stdlib.Std

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

let cfg = Test_helpers.default_config ()

(* ========== Pure OCaml baselines ========== *)

let ocaml_sort arr _n =
  let a = Array.copy arr in
  Array.sort Int32.compare a ;
  a

(* ========== Shared test data ========== *)

let input_bitonic_global = ref [||]

let expected_bitonic_global = ref [||]

let sort_size_global = ref 0

let input_odd_even = ref [||]

let expected_odd_even = ref [||]

let sort_size_odd_even = ref 0

let init_bitonic_global_data () =
  let log2n = int_of_float (log (float_of_int cfg.size) /. log 2.0) in
  let n = 1 lsl log2n in
  sort_size_global := n ;
  Random.init 42 ;
  let inp = Array.init n (fun _ -> Int32.of_int (Random.int 10000)) in
  input_bitonic_global := inp ;
  let t0 = Unix.gettimeofday () in
  expected_bitonic_global := ocaml_sort inp n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_odd_even_data () =
  let n = min 512 cfg.size in
  sort_size_odd_even := n ;
  Random.init 42 ;
  let inp = Array.init n (fun _ -> Int32.of_int (Random.int 10000)) in
  input_odd_even := inp ;
  let t0 = Unix.gettimeofday () in
  expected_odd_even := ocaml_sort inp n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Sarek kernels ========== *)

(** Bitonic sort step - one comparison/swap pass *)
let bitonic_sort_step_kernel =
  [%kernel
    fun (data : int32 vector) (j : int32) (k : int32) (n : int32) ->
      let open Std in
      let i = global_thread_id in
      if i < n then begin
        let ij = i lxor j in
        if ij > i then begin
          let di = data.(i) in
          let dij = data.(ij) in
          let ascending = i land k = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            data.(i) <- dij ;
            data.(ij) <- di
          end
        end
      end]

(** Odd-even transposition sort step *)
let odd_even_step_kernel =
  [%kernel
    fun (data : int32 vector) (phase : int32) (n : int32) ->
      let open Std in
      let i = global_thread_id in
      let idx = (2l * i) + phase in
      if idx + 1l < n then begin
        let a = data.(idx) in
        let b = data.(idx + 1l) in
        if a > b then begin
          data.(idx) <- b ;
          data.(idx + 1l) <- a
        end
      end]

(* ========== V2 test runners ========== *)

(** Run global bitonic sort on V2 *)
let run_bitonic_sort_global_v2 (dev : V2_Device.t) =
  let n = !sort_size_global in
  let inp = !input_bitonic_global in
  let exp = !expected_bitonic_global in
  let _, kirc = bitonic_sort_step_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "No V2 IR"
  in

  let data = V2_Vector.create V2_Vector.int32 n in

  for i = 0 to n - 1 do
    V2_Vector.set data i inp.(i)
  done ;

  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

  let t0 = Unix.gettimeofday () in
  let k = ref 2 in
  while !k <= n do
    let j = ref (!k / 2) in
    while !j > 0 do
      Sarek.Execute.run_vectors
        ~device:dev
        ~ir
        ~args:
          [
            Sarek.Execute.Vec data;
            Sarek.Execute.Int32 (Int32.of_int !j);
            Sarek.Execute.Int32 (Int32.of_int !k);
            Sarek.Execute.Int32 (Int32.of_int n);
          ]
        ~block
        ~grid
        () ;
      V2_Transfer.flush dev ;
      j := !j / 2
    done ;
    k := !k * 2
  done ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = V2_Vector.to_array data in
      let errors = ref 0 in
      for i = 0 to n - 2 do
        if result.(i) > result.(i + 1) then incr errors
      done ;
      for i = 0 to min 10 (n - 1) do
        if result.(i) <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run odd-even transposition sort on V2 *)
let run_odd_even_sort_v2 (dev : V2_Device.t) =
  let n = !sort_size_odd_even in
  let inp = !input_odd_even in
  let exp = !expected_odd_even in
  let _, kirc = odd_even_step_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_v2 with
    | Some ir -> ir
    | None -> failwith "No V2 IR"
  in

  let data = V2_Vector.create V2_Vector.int32 n in

  for i = 0 to n - 1 do
    V2_Vector.set data i inp.(i)
  done ;

  let block_size = 256 in
  let grid_size = ((n / 2) + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d (max 1 grid_size) in

  let t0 = Unix.gettimeofday () in
  for phase = 0 to n - 1 do
    let phase_mod = Int32.of_int (phase mod 2) in
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        [
          Sarek.Execute.Vec data;
          Sarek.Execute.Int32 phase_mod;
          Sarek.Execute.Int32 (Int32.of_int n);
        ]
      ~block
      ~grid
      () ;
    V2_Transfer.flush dev
  done ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = V2_Vector.to_array data in
      let errors = ref 0 in
      for i = 0 to n - 2 do
        if result.(i) > result.(i + 1) then incr errors
      done ;
      for i = 0 to min 10 (n - 1) do
        if result.(i) <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(* ========== Main ========== *)

let () =
  let c = Test_helpers.parse_args "test_sort" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  print_endline "=== Sorting Tests (V2) ===" ;
  Printf.printf "Size: %d elements\n\n" cfg.size ;

  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length v2_devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices v2_devs ;
  Printf.printf "\nFound %d V2 device(s)\n\n" (Array.length v2_devs) ;

  (* Initialize test data *)
  ignore (init_bitonic_global_data ()) ;
  ignore (init_odd_even_data ()) ;

  let all_ok = ref true in

  (* Benchmark bitonic global *)
  print_endline "=== Bitonic Sort (global) ===" ;
  Array.iter
    (fun v2_dev ->
      let name = v2_dev.V2_Device.name in
      let framework = v2_dev.V2_Device.framework in
      let v2_time, v2_ok = run_bitonic_sort_global_v2 v2_dev in
      Printf.printf
        "  %s (%s): %.4f ms, %s\n%!"
        name
        framework
        v2_time
        (if v2_ok then "OK" else "FAIL") ;
      if not v2_ok then all_ok := false)
    v2_devs ;

  (* Benchmark odd-even *)
  print_endline "\n=== Odd-Even Sort ===" ;
  Array.iter
    (fun v2_dev ->
      let name = v2_dev.V2_Device.name in
      let framework = v2_dev.V2_Device.framework in
      let v2_time, v2_ok = run_odd_even_sort_v2 v2_dev in
      Printf.printf
        "  %s (%s): %.4f ms, %s\n%!"
        name
        framework
        v2_time
        (if v2_ok then "OK" else "FAIL") ;
      if not v2_ok then all_ok := false)
    v2_devs ;

  if !all_ok then print_endline "\n=== All sort tests PASSED ==="
  else begin
    print_endline "\n=== Some sort tests FAILED ===" ;
    exit 1
  end
