(******************************************************************************
 * E2E test for Sarek PPX - Prefix Sum (Scan)
 *
 * Tests inclusive prefix sum operations with shared memory and supersteps.
 * Scan is a fundamental parallel primitive for many algorithms.
 * GPU runtime only.
 ******************************************************************************)

(* Module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

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

(* ========== runtime test runners ========== *)

let run_inclusive_scan (dev : Device.t) inp exp =
  let n = min cfg.size 256 in
  let _, kirc = inclusive_scan_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let input = Vector.create Vector.int32 n in
  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Vector.set input i inp.(i) ;
    Vector.set output i 0l
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
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = Vector.to_array output in
      let errors = ref 0 in
      for i = 0 to n - 1 do
        if result.(i) <> exp.(i) then begin
          if !errors < 10 then
            Printf.printf
              "  runtime Mismatch at %d: expected %ld, got %ld\n"
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

  print_endline "=== Prefix Scan Tests (runtime) ===" ;
  Printf.printf
    "Size: %d elements (max 256 for block-level scan)\n\n"
    (min cfg.size 256) ;

  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices devs ;
  Printf.printf "\nFound %d runtime device(s)\n\n" (Array.length devs) ;

  let all_ok = ref true in

  (* Scan with ones *)
  ignore (init_ones_data ()) ;
  print_endline "=== Inclusive Scan (all ones) ===" ;
  Array.iter
    (fun dev ->
      let name = dev.Device.name in
      let framework = dev.Device.framework in
      let time, ok = run_inclusive_scan dev !input_ones !expected_ones in
      Printf.printf
        "  %s (%s): %.4f ms, %s\n%!"
        name
        framework
        time
        (if ok then "OK" else "FAIL") ;
      if not ok then all_ok := false)
    devs ;

  (* Scan with varying values *)
  ignore (init_varying_data ()) ;
  print_endline "\n=== Inclusive Scan (varying values) ===" ;
  Array.iter
    (fun dev ->
      let name = dev.Device.name in
      let framework = dev.Device.framework in
      let time, ok = run_inclusive_scan dev !input_varying !expected_varying in
      Printf.printf
        "  %s (%s): %.4f ms, %s\n%!"
        name
        framework
        time
        (if ok then "OK" else "FAIL") ;
      if not ok then all_ok := false)
    devs ;

  if !all_ok then print_endline "\n=== All scan tests PASSED ==="
  else begin
    print_endline "\n=== Some scan tests FAILED ===" ;
    exit 1
  end
