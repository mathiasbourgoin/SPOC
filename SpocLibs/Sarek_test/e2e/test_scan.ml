(******************************************************************************
 * E2E test for Sarek PPX - Prefix Sum (Scan) with V2 comparison
 *
 * Tests inclusive prefix sum operations with shared memory and supersteps.
 * Scan is a fundamental parallel primitive for many algorithms.
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

let ocaml_inclusive_scan input output n =
  if n > 0 then begin
    output.(0) <- input.(0) ;
    for i = 1 to n - 1 do
      output.(i) <- Int32.add output.(i - 1) input.(i)
    done
  end

(* ========== Shared test data ========== *)

let input_ones = ref [||]

let expected_ones = ref [||]

let input_varying = ref [||]

let expected_varying = ref [||]

let init_ones_data () =
  let n = min cfg.size 256 in
  let inp = Array.make n 1l in
  let out = Array.make n 0l in
  input_ones := inp ;
  expected_ones := out ;
  let t0 = Unix.gettimeofday () in
  ocaml_inclusive_scan inp out n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_varying_data () =
  let n = min cfg.size 256 in
  let inp = Array.init n (fun i -> Int32.of_int (i + 1)) in
  let out = Array.make n 0l in
  input_varying := inp ;
  expected_varying := out ;
  let t0 = Unix.gettimeofday () in
  ocaml_inclusive_scan inp out n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Sarek kernels ========== *)

(** Inclusive scan within a block using Hillis-Steele algorithm. *)
let inclusive_scan_kernel =
  [%kernel
    fun (input : int32 vector) (output : int32 vector) (n : int32) ->
      let%shared (temp : int32) = 512l in
      let%shared (temp2 : int32) = 512l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then temp.(tid) <- input.(gid) else temp.(tid) <- 0l
      in
      let%superstep step1 =
        let v = temp.(tid) in
        let add = if tid >= 1l then temp.(tid - 1l) else 0l in
        temp2.(tid) <- v + add
      in
      let%superstep step2 =
        let v = temp2.(tid) in
        let add = if tid >= 2l then temp2.(tid - 2l) else 0l in
        temp.(tid) <- v + add
      in
      let%superstep step4 =
        let v = temp.(tid) in
        let add = if tid >= 4l then temp.(tid - 4l) else 0l in
        temp2.(tid) <- v + add
      in
      let%superstep step8 =
        let v = temp2.(tid) in
        let add = if tid >= 8l then temp2.(tid - 8l) else 0l in
        temp.(tid) <- v + add
      in
      let%superstep step16 =
        let v = temp.(tid) in
        let add = if tid >= 16l then temp.(tid - 16l) else 0l in
        temp2.(tid) <- v + add
      in
      let%superstep step32 =
        let v = temp2.(tid) in
        let add = if tid >= 32l then temp2.(tid - 32l) else 0l in
        temp.(tid) <- v + add
      in
      let%superstep step64 =
        let v = temp.(tid) in
        let add = if tid >= 64l then temp.(tid - 64l) else 0l in
        temp2.(tid) <- v + add
      in
      let%superstep step128 =
        let v = temp2.(tid) in
        let add = if tid >= 128l then temp2.(tid - 128l) else 0l in
        temp.(tid) <- v + add
      in
      if gid < n then output.(gid) <- temp.(tid)]

(* ========== SPOC test runners ========== *)

let run_inclusive_scan_spoc dev inp exp =
  let n = min cfg.size 256 in

  let input = Spoc_Vector.create Spoc_Vector.int32 n in
  let output = Spoc_Vector.create Spoc_Vector.int32 n in

  for i = 0 to n - 1 do
    Spoc_Mem.set input i inp.(i) ;
    Spoc_Mem.set output i 0l
  done ;

  ignore (Sarek.Kirc.gen inclusive_scan_kernel dev) ;
  let block_size = min 256 (Test_helpers.get_block_size cfg dev) in
  let block = {Spoc.Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Spoc.Kernel.gridX = 1; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run inclusive_scan_kernel (input, output, n) (block, grid) 0 dev ;
  Spoc_Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Spoc_Mem.to_cpu output () ;
      Spoc_Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 1 do
        let got = Spoc_Mem.get output i in
        if got <> exp.(i) then begin
          if !errors < 10 then
            Printf.printf
              "  Mismatch at %d: expected %ld, got %ld\n"
              i
              exp.(i)
              got ;
          incr errors
        end
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(* ========== V2 test runners ========== *)

let run_inclusive_scan_v2 (dev : V2_Device.t) inp exp =
  let n = min cfg.size 256 in
  let _, kirc = inclusive_scan_kernel in
  let ir =
    match kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "No V2 IR"
  in

  let input = V2_Vector.create V2_Vector.int32 n in
  let output = V2_Vector.create V2_Vector.int32 n in

  for i = 0 to n - 1 do
    V2_Vector.set input i inp.(i) ;
    V2_Vector.set output i 0l
  done ;

  let block_size = 256 in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d 1 in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    ~block
    ~grid
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = V2_Vector.to_array output in
      let errors = ref 0 in
      for i = 0 to n - 1 do
        if result.(i) <> exp.(i) then begin
          if !errors < 10 then
            Printf.printf
              "  V2 Mismatch at %d: expected %ld, got %ld\n"
              i
              exp.(i)
              result.(i) ;
          incr errors
        end
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(* ========== Main ========== *)

let () =
  let c = Test_helpers.parse_args "test_scan" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  print_endline "=== Prefix Scan Tests (SPOC + V2 Comparison) ===" ;
  Printf.printf
    "Size: %d elements (max 256 for block-level scan)\n\n"
    (min cfg.size 256) ;

  let spoc_devs = Spoc_Devices.init () in
  if Array.length spoc_devs = 0 then begin
    print_endline "No GPU devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices spoc_devs ;

  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  Printf.printf "\nFound %d V2 device(s)\n\n" (Array.length v2_devs) ;

  if cfg.benchmark_all then begin
    let all_ok = ref true in

    (* Scan with ones *)
    ignore (init_ones_data ()) ;
    print_endline "=== Inclusive Scan (all ones) ===" ;
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
              let t, ok =
                run_inclusive_scan_spoc spoc_dev !input_ones !expected_ones
              in
              (Printf.sprintf "%.4f" t, if ok then "OK" else "FAIL")
          | None -> ("-", "SKIP")
        in
        let v2_time, v2_ok =
          run_inclusive_scan_v2 v2_dev !input_ones !expected_ones
        in
        if not v2_ok then all_ok := false ;
        if spoc_ok = "FAIL" then all_ok := false ;
        Printf.printf
          "%-35s %10s %10.4f %8s %8s\n"
          (Printf.sprintf "%s (%s)" name framework)
          spoc_time
          v2_time
          spoc_ok
          (if v2_ok then "OK" else "FAIL"))
      v2_devs ;
    print_endline (String.make 80 '-') ;

    (* Scan with varying values *)
    ignore (init_varying_data ()) ;
    print_endline "\n=== Inclusive Scan (varying values) ===" ;
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
              let t, ok =
                run_inclusive_scan_spoc
                  spoc_dev
                  !input_varying
                  !expected_varying
              in
              (Printf.sprintf "%.4f" t, if ok then "OK" else "FAIL")
          | None -> ("-", "SKIP")
        in
        let v2_time, v2_ok =
          run_inclusive_scan_v2 v2_dev !input_varying !expected_varying
        in
        if not v2_ok then all_ok := false ;
        if spoc_ok = "FAIL" then all_ok := false ;
        Printf.printf
          "%-35s %10s %10.4f %8s %8s\n"
          (Printf.sprintf "%s (%s)" name framework)
          spoc_time
          v2_time
          spoc_ok
          (if v2_ok then "OK" else "FAIL"))
      v2_devs ;
    print_endline (String.make 80 '-') ;

    if !all_ok then print_endline "\n=== All scan tests PASSED ==="
    else begin
      print_endline "\n=== Some scan tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    let dev = Test_helpers.get_device cfg spoc_devs in
    let dev_name = dev.Spoc_Devices.general_info.Spoc_Devices.name in
    Printf.printf "Using device: %s\n%!" dev_name ;
    let v2_dev_opt =
      Array.find_opt (fun d -> d.V2_Device.name = dev_name) v2_devs
    in

    ignore (init_ones_data ()) ;
    Printf.printf "\n--- Inclusive Scan (all ones) ---\n%!" ;
    let t, ok = run_inclusive_scan_spoc dev !input_ones !expected_ones in
    Printf.printf
      "  SPOC: %.4f ms, %s\n%!"
      t
      (if ok then "PASSED" else "FAILED") ;
    (match v2_dev_opt with
    | Some v2_dev ->
        let t, ok = run_inclusive_scan_v2 v2_dev !input_ones !expected_ones in
        Printf.printf
          "  V2: %.4f ms, %s\n%!"
          t
          (if ok then "PASSED" else "FAILED")
    | None -> ()) ;

    ignore (init_varying_data ()) ;
    Printf.printf "\n--- Inclusive Scan (varying values) ---\n%!" ;
    let t, ok = run_inclusive_scan_spoc dev !input_varying !expected_varying in
    Printf.printf
      "  SPOC: %.4f ms, %s\n%!"
      t
      (if ok then "PASSED" else "FAILED") ;
    (match v2_dev_opt with
    | Some v2_dev ->
        let t, ok =
          run_inclusive_scan_v2 v2_dev !input_varying !expected_varying
        in
        Printf.printf
          "  V2: %.4f ms, %s\n%!"
          t
          (if ok then "PASSED" else "FAILED")
    | None -> ()) ;

    print_endline "\nScan tests PASSED"
  end
