(******************************************************************************
 * E2E test for Sarek PPX - Sorting algorithms with V2 comparison
 *
 * Tests bitonic sort and odd-even merge sort - parallel sorting algorithms
 * that work well on GPUs due to their regular communication patterns.
 *
 * V2 comparison: global bitonic step and odd-even step only.
 * Block-level bitonic (shared memory + supersteps) runs SPOC-only.
 ******************************************************************************)

(* Module aliases *)
module Spoc_Vector = Spoc.Vector
module Spoc_Devices = Spoc.Devices
module Spoc_Mem = Spoc.Mem
module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
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

let input_bitonic_block = ref [||]

let expected_bitonic_block = ref [||]

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

let init_bitonic_block_data () =
  let n = 16 in
  Random.init 42 ;
  let inp = Array.init n (fun _ -> Int32.of_int (Random.int 10000)) in
  input_bitonic_block := inp ;
  let t0 = Unix.gettimeofday () in
  expected_bitonic_block := ocaml_sort inp n ;
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

(** Bitonic sort step - one comparison/swap pass (V2 compatible) *)
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

(* NOTE: Not V2 compatible - uses shared memory and supersteps *)

(** Block-level bitonic sort using shared memory with supersteps *)
let bitonic_sort_block_kernel =
  [%kernel
    fun (data : int32 vector) (n : int32) ->
      let%shared (shared : int32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      (* Load to shared memory *)
      let%superstep load =
        if gid < n then shared.(tid) <- data.(gid)
        else shared.(tid) <- 2147483647l
      in
      (* Bitonic sort stage k=2 *)
      let%superstep[@divergent] sort_k2 =
        let ij = tid lxor 1l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 2l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=4, j=2 *)
      let%superstep[@divergent] sort_k4_j2 =
        let ij = tid lxor 2l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 4l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=4, j=1 *)
      let%superstep[@divergent] sort_k4_j1 =
        let ij = tid lxor 1l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 4l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=8, j=4 *)
      let%superstep[@divergent] sort_k8_j4 =
        let ij = tid lxor 4l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 8l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=8, j=2 *)
      let%superstep[@divergent] sort_k8_j2 =
        let ij = tid lxor 2l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 8l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=8, j=1 *)
      let%superstep[@divergent] sort_k8_j1 =
        let ij = tid lxor 1l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 8l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=16, j=8 *)
      let%superstep[@divergent] sort_k16_j8 =
        let ij = tid lxor 8l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 16l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=16, j=4 *)
      let%superstep[@divergent] sort_k16_j4 =
        let ij = tid lxor 4l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 16l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=16, j=2 *)
      let%superstep[@divergent] sort_k16_j2 =
        let ij = tid lxor 2l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 16l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=16, j=1 *)
      let%superstep[@divergent] sort_k16_j1 =
        let ij = tid lxor 1l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 16l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Write back *)
      if gid < n then data.(gid) <- shared.(tid)]

(** Odd-even transposition sort step (V2 compatible) *)
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

(* ========== SPOC test runners ========== *)

(** Run global bitonic sort test *)
let run_bitonic_sort_global_spoc dev =
  let n = !sort_size_global in
  let inp = !input_bitonic_global in
  let exp = !expected_bitonic_global in

  let data = Spoc_Vector.create Spoc_Vector.int32 n in

  for i = 0 to n - 1 do
    Spoc_Mem.set data i inp.(i)
  done ;

  ignore (Sarek.Kirc.gen bitonic_sort_step_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Spoc.Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Spoc.Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  let k = ref 2 in
  while !k <= n do
    let j = ref (!k / 2) in
    while !j > 0 do
      Sarek.Kirc.run
        bitonic_sort_step_kernel
        (data, !j, !k, n)
        (block, grid)
        0
        dev ;
      Spoc_Devices.flush dev () ;
      j := !j / 2
    done ;
    k := !k * 2
  done ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Spoc_Mem.to_cpu data () ;
      Spoc_Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 2 do
        if Spoc_Mem.get data i > Spoc_Mem.get data (i + 1) then incr errors
      done ;
      for i = 0 to min 10 (n - 1) do
        if Spoc_Mem.get data i <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run block-level bitonic sort test - sorts 16 elements *)
let run_bitonic_sort_block_spoc dev =
  let n = 16 in
  let inp = !input_bitonic_block in
  let exp = !expected_bitonic_block in

  let data = Spoc_Vector.create Spoc_Vector.int32 n in

  for i = 0 to n - 1 do
    Spoc_Mem.set data i inp.(i)
  done ;

  ignore (Sarek.Kirc.gen bitonic_sort_block_kernel dev) ;
  let block = {Spoc.Kernel.blockX = n; blockY = 1; blockZ = 1} in
  let grid = {Spoc.Kernel.gridX = 1; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run bitonic_sort_block_kernel (data, n) (block, grid) 0 dev ;
  Spoc_Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Spoc_Mem.to_cpu data () ;
      Spoc_Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 2 do
        if Spoc_Mem.get data i > Spoc_Mem.get data (i + 1) then begin
          if !errors < 10 then
            Printf.printf
              "  Out of order at %d: %ld > %ld\n"
              i
              (Spoc_Mem.get data i)
              (Spoc_Mem.get data (i + 1)) ;
          incr errors
        end
      done ;
      for i = 0 to n - 1 do
        if Spoc_Mem.get data i <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run odd-even transposition sort test *)
let run_odd_even_sort_spoc dev =
  let n = !sort_size_odd_even in
  let inp = !input_odd_even in
  let exp = !expected_odd_even in

  let data = Spoc_Vector.create Spoc_Vector.int32 n in

  for i = 0 to n - 1 do
    Spoc_Mem.set data i inp.(i)
  done ;

  ignore (Sarek.Kirc.gen odd_even_step_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = ((n / 2) + block_size - 1) / block_size in
  let block = {Spoc.Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Spoc.Kernel.gridX = max 1 blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  for phase = 0 to n - 1 do
    let phase_mod = phase mod 2 in
    Sarek.Kirc.run odd_even_step_kernel (data, phase_mod, n) (block, grid) 0 dev ;
    Spoc_Devices.flush dev ()
  done ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Spoc_Mem.to_cpu data () ;
      Spoc_Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 2 do
        if Spoc_Mem.get data i > Spoc_Mem.get data (i + 1) then incr errors
      done ;
      for i = 0 to min 10 (n - 1) do
        if Spoc_Mem.get data i <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(* ========== V2 test runners ========== *)

(** Run global bitonic sort on V2 *)
let run_bitonic_sort_global_v2 (dev : V2_Device.t) =
  let n = !sort_size_global in
  let inp = !input_bitonic_global in
  let exp = !expected_bitonic_global in
  let _, kirc = bitonic_sort_step_kernel in
  let ir =
    match kirc.Sarek.Kirc.body_v2 with
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
    match kirc.Sarek.Kirc.body_v2 with
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

  print_endline "=== Sorting Tests (SPOC + V2 Comparison) ===" ;
  Printf.printf "Size: %d elements\n\n" cfg.size ;

  let spoc_devs = Spoc_Devices.init () in
  if Array.length spoc_devs = 0 then begin
    print_endline "No GPU devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices spoc_devs ;

  let v2_devs = V2_Device.init ~frameworks:["CUDA"; "OpenCL"] () in
  Printf.printf "\nFound %d V2 device(s)\n\n" (Array.length v2_devs) ;

  (* Initialize test data *)
  ignore (init_bitonic_global_data ()) ;
  ignore (init_bitonic_block_data ()) ;
  ignore (init_odd_even_data ()) ;

  if cfg.benchmark_all then begin
    (* Benchmark bitonic global *)
    print_endline "=== Bitonic Sort (global) ===" ;
    print_endline (String.make 80 '-') ;
    Printf.printf
      "%-35s %10s %10s %8s %8s\n"
      "Device"
      "SPOC(ms)"
      "V2(ms)"
      "SPOC"
      "V2" ;
    print_endline (String.make 80 '-') ;

    let all_ok = ref true in

    Array.iter
      (fun v2_dev ->
        let name = v2_dev.V2_Device.name in
        let framework = v2_dev.V2_Device.framework in

        let spoc_dev_opt =
          Array.find_opt
            (fun d -> d.Spoc_Devices.general_info.Spoc_Devices.name = name)
            spoc_devs
        in

        let spoc_time, spoc_ok =
          match spoc_dev_opt with
          | Some spoc_dev ->
              let time, ok = run_bitonic_sort_global_spoc spoc_dev in
              (Printf.sprintf "%.4f" time, if ok then "OK" else "FAIL")
          | None -> ("-", "SKIP")
        in

        let v2_time, v2_ok = run_bitonic_sort_global_v2 v2_dev in
        let v2_status = if v2_ok then "OK" else "FAIL" in

        if not v2_ok then all_ok := false ;
        if spoc_ok = "FAIL" then all_ok := false ;

        Printf.printf
          "%-35s %10s %10.4f %8s %8s\n"
          (Printf.sprintf "%s (%s)" name framework)
          spoc_time
          v2_time
          spoc_ok
          v2_status)
      v2_devs ;

    print_endline (String.make 80 '-') ;

    (* Benchmark bitonic block (SPOC only) *)
    print_endline "\n=== Bitonic Sort (block-level, SPOC only) ===" ;
    print_endline (String.make 60 '-') ;
    Printf.printf "%-35s %10s %8s\n" "Device" "SPOC(ms)" "Status" ;
    print_endline (String.make 60 '-') ;

    Array.iter
      (fun spoc_dev ->
        let name = spoc_dev.Spoc_Devices.general_info.Spoc_Devices.name in
        let time, ok = run_bitonic_sort_block_spoc spoc_dev in
        Printf.printf
          "%-35s %10.4f %8s\n"
          name
          time
          (if ok then "OK" else "FAIL") ;
        if not ok then all_ok := false)
      spoc_devs ;

    print_endline (String.make 60 '-') ;

    (* Benchmark odd-even *)
    print_endline "\n=== Odd-Even Sort ===" ;
    print_endline (String.make 80 '-') ;
    Printf.printf
      "%-35s %10s %10s %8s %8s\n"
      "Device"
      "SPOC(ms)"
      "V2(ms)"
      "SPOC"
      "V2" ;
    print_endline (String.make 80 '-') ;

    Array.iter
      (fun v2_dev ->
        let name = v2_dev.V2_Device.name in
        let framework = v2_dev.V2_Device.framework in

        let spoc_dev_opt =
          Array.find_opt
            (fun d -> d.Spoc_Devices.general_info.Spoc_Devices.name = name)
            spoc_devs
        in

        let spoc_time, spoc_ok =
          match spoc_dev_opt with
          | Some spoc_dev ->
              let time, ok = run_odd_even_sort_spoc spoc_dev in
              (Printf.sprintf "%.4f" time, if ok then "OK" else "FAIL")
          | None -> ("-", "SKIP")
        in

        let v2_time, v2_ok = run_odd_even_sort_v2 v2_dev in
        let v2_status = if v2_ok then "OK" else "FAIL" in

        if not v2_ok then all_ok := false ;
        if spoc_ok = "FAIL" then all_ok := false ;

        Printf.printf
          "%-35s %10s %10.4f %8s %8s\n"
          (Printf.sprintf "%s (%s)" name framework)
          spoc_time
          v2_time
          spoc_ok
          v2_status)
      v2_devs ;

    print_endline (String.make 80 '-') ;

    if !all_ok then print_endline "\n=== All sort tests PASSED ==="
    else begin
      print_endline "\n=== Some sort tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    let dev = Test_helpers.get_device cfg spoc_devs in
    let dev_name = dev.Spoc_Devices.general_info.Spoc_Devices.name in
    Printf.printf "Using device: %s\n%!" dev_name ;

    (* Bitonic global *)
    Printf.printf "\n--- Bitonic Sort (global) ---\n%!" ;
    Printf.printf "Running SPOC path...\n%!" ;
    let spoc_time, spoc_ok = run_bitonic_sort_global_spoc dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      spoc_time
      (if spoc_ok then "PASSED" else "FAILED") ;

    let v2_dev_opt =
      Array.find_opt (fun d -> d.V2_Device.name = dev_name) v2_devs
    in
    (match v2_dev_opt with
    | Some v2_dev ->
        Printf.printf "Running V2 path...\n%!" ;
        let v2_time, v2_ok = run_bitonic_sort_global_v2 v2_dev in
        Printf.printf
          "  Time: %.4f ms, %s\n%!"
          v2_time
          (if v2_ok then "PASSED" else "FAILED")
    | None -> Printf.printf "No matching V2 device\n%!") ;

    (* Bitonic block *)
    Printf.printf "\n--- Bitonic Sort (block-level, SPOC only) ---\n%!" ;
    let time, ok = run_bitonic_sort_block_spoc dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time
      (if ok then "PASSED" else "FAILED") ;

    (* Odd-even *)
    Printf.printf "\n--- Odd-Even Sort ---\n%!" ;
    Printf.printf "Running SPOC path...\n%!" ;
    let spoc_time, spoc_ok = run_odd_even_sort_spoc dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      spoc_time
      (if spoc_ok then "PASSED" else "FAILED") ;

    (match v2_dev_opt with
    | Some v2_dev ->
        Printf.printf "Running V2 path...\n%!" ;
        let v2_time, v2_ok = run_odd_even_sort_v2 v2_dev in
        Printf.printf
          "  Time: %.4f ms, %s\n%!"
          v2_time
          (if v2_ok then "PASSED" else "FAILED")
    | None -> Printf.printf "No matching V2 device\n%!") ;

    print_endline "\nSort tests PASSED"
  end
