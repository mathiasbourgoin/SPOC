(******************************************************************************
 * E2E test for Sarek PPX - Mandelbrot set computation
 *
 * Tests iterative computation with complex arithmetic.
 * Mandelbrot is a classic GPU benchmark with high arithmetic intensity.
 *
 * GPU runtime only.
 ******************************************************************************)

module Std = Sarek_stdlib.Std

(* runtime module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init ()

let cfg = Test_helpers.default_config ()

let max_iter = ref 256

(* ========== Pure OCaml baselines ========== *)

let ocaml_mandelbrot output width height max_iter =
  for py = 0 to height - 1 do
    for px = 0 to width - 1 do
      let x0 = (4.0 *. float_of_int px /. float_of_int width) -. 2.5 in
      let y0 = (3.0 *. float_of_int py /. float_of_int height) -. 1.5 in
      let x = ref 0.0 in
      let y = ref 0.0 in
      let iter = ref 0 in
      while (!x *. !x) +. (!y *. !y) <= 4.0 && !iter < max_iter do
        let xtemp = (!x *. !x) -. (!y *. !y) +. x0 in
        y := (2.0 *. !x *. !y) +. y0 ;
        x := xtemp ;
        incr iter
      done ;
      output.((py * width) + px) <- Int32.of_int !iter
    done
  done

(* ========== Shared test data ========== *)

let expected_mandelbrot = ref [||]

let image_dim = ref 0

let init_mandelbrot_data () =
  let dim = int_of_float (sqrt (float_of_int cfg.size)) in
  image_dim := dim ;
  let n = dim * dim in
  let output = Array.make n 0l in
  expected_mandelbrot := output ;
  let t0 = Unix.gettimeofday () in
  ocaml_mandelbrot output dim dim !max_iter ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Mandelbrot kernel ========== *)

let mandelbrot_kernel =
  [%kernel
    fun (output : int32 vector) (width : int) (height : int) (max_iter : int) ->
      let open Std in
      let px = global_idx_x in
      let py = global_idx_y in
      if px < width && py < height then begin
        let x0 = (4.0 *. (float px /. float width)) -. 2.5 in
        let y0 = (3.0 *. (float py /. float height)) -. 1.5 in
        let x = mut 0.0 in
        let y = mut 0.0 in
        let iter = mut 0l in
        while (x *. x) +. (y *. y) <= 4.0 && iter < max_iter do
          let xtemp = (x *. x) -. (y *. y) +. x0 in
          y := (2.0 *. x *. y) +. y0 ;
          x := xtemp ;
          iter := iter + 1l
        done ;
        output.((py * width) + px) <- iter
      end]

let run_mandelbrot_test (dev : Device.t) =
  let _, kirc = mandelbrot_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Mandelbrot kernel has no IR"
  in

  let dim = !image_dim in
  let width = dim in
  let height = dim in
  let n = width * height in
  let exp = !expected_mandelbrot in

  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Vector.set output i 0l
  done ;

  let block_size = 16 in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = Sarek.Execute.dims2d block_size block_size in
  let grid = Sarek.Execute.dims2d blocks_x blocks_y in

  (* Warm up: run once to trigger JIT compilation (cached for subsequent runs) *)
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec output;
        Sarek.Execute.Int width;
        Sarek.Execute.Int height;
        Sarek.Execute.Int !max_iter;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;

  (* Reset output for timed run *)
  for i = 0 to n - 1 do
    Vector.set output i 0l
  done ;

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec output;
        Sarek.Execute.Int width;
        Sarek.Execute.Int height;
        Sarek.Execute.Int !max_iter;
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
      for i = 0 to min 1000 (n - 1) do
        if result.(i) <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(* ========== Tail-recursive Mandelbrot ========== *)

let mandelbrot_tailrec_kernel =
  [%kernel
    (* Tail-recursive iteration using module function *)
    let open Std in
    let rec iterate (x : float32) (y : float32) (x0 : float32) (y0 : float32)
        (iter : int32) (max_iter : int32) : int32 =
      if (x *. x) +. (y *. y) > 4.0 || iter >= max_iter then iter
      else
        let xtemp = (x *. x) -. (y *. y) +. x0 in
        let ynew = (2.0 *. x *. y) +. y0 in
        iterate xtemp ynew x0 y0 (iter + 1l) max_iter
    in

    fun (output : int32 vector) (width : int) (height : int) (max_iter : int) ->
      let px = global_idx_x in
      let py = global_idx_y in
      if px < width && py < height then begin
        let x0 = (4.0 *. (float px /. float width)) -. 2.5 in
        let y0 = (3.0 *. (float py /. float height)) -. 1.5 in
        let iter = iterate 0.0 0.0 x0 y0 0l max_iter in
        output.((py * width) + px) <- iter
      end]

let run_mandelbrot_tailrec_test (dev : Device.t) =
  let _, kirc = mandelbrot_tailrec_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Mandelbrot tailrec kernel has no IR"
  in

  let dim = !image_dim in
  let width = dim in
  let height = dim in
  let n = width * height in
  let exp = !expected_mandelbrot in

  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Vector.set output i 0l
  done ;

  let block_size = 16 in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = Sarek.Execute.dims2d block_size block_size in
  let grid = Sarek.Execute.dims2d blocks_x blocks_y in

  (* Warm up: run once to trigger JIT compilation *)
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec output;
        Sarek.Execute.Int width;
        Sarek.Execute.Int height;
        Sarek.Execute.Int !max_iter;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;

  (* Reset output for timed run *)
  for i = 0 to n - 1 do
    Vector.set output i 0l
  done ;

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec output;
        Sarek.Execute.Int width;
        Sarek.Execute.Int height;
        Sarek.Execute.Int !max_iter;
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
      for i = 0 to min 1000 (n - 1) do
        if result.(i) <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_mandelbrot" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  (* Initialize runtime devices *)
  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then begin
    print_endline "No GPU devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices devs ;

  let dim = int_of_float (sqrt (float_of_int cfg.size)) in
  Printf.printf "Image dimensions: %dx%d, max_iter=%d\n%!" dim dim !max_iter ;

  let baseline_ms, _ = init_mandelbrot_data () in
  Printf.printf "\nOCaml baseline (Mandelbrot): %.4f ms\n%!" baseline_ms ;

  if cfg.benchmark_all then begin
    Printf.printf "\n=== GPU Runtime Benchmarks ===\n%!" ;
    Array.iter
      (fun dev ->
        let dev_label =
          Printf.sprintf "%s (%s)" dev.Device.name dev.Device.framework
        in
        Printf.printf "\nV2 Mandelbrot on %s:\n%!" dev_label ;
        let time_ms, ok = run_mandelbrot_test dev in
        Printf.printf
          "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
          time_ms
          (baseline_ms /. time_ms)
          (if ok then "PASSED" else "FAILED") ;
        Printf.printf "\nV2 Mandelbrot (tailrec) on %s:\n%!" dev_label ;
        let tr_time_ms, tr_ok = run_mandelbrot_tailrec_test dev in
        Printf.printf
          "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
          tr_time_ms
          (baseline_ms /. tr_time_ms)
          (if tr_ok then "PASSED" else "FAILED"))
      devs
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Device.name ;

    (* Run runtime Mandelbrot *)
    Printf.printf "\nMandelbrot (runtime):\n%!" ;
    let time, ok = run_mandelbrot_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time
      (baseline_ms /. time)
      (if ok then "PASSED" else "FAILED") ;
    if not ok then begin
      print_endline "\nV2 Mandelbrot test FAILED" ;
      exit 1
    end ;

    (* runtime Tail-recursive test *)
    Printf.printf "\nMandelbrot (tail-recursive runtime):\n%!" ;
    let tr_time, tr_ok = run_mandelbrot_tailrec_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      tr_time
      (baseline_ms /. tr_time)
      (if tr_ok then "PASSED" else "FAILED") ;
    if not tr_ok then begin
      print_endline "\nV2 Mandelbrot tailrec test FAILED" ;
      exit 1
    end ;

    print_endline "\nMandelbrot tests PASSED"
  end
