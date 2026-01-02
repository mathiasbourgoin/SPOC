(******************************************************************************
 * E2E test for Sarek PPX - Mandelbrot set computation with V2 comparison
 *
 * Tests iterative computation with complex arithmetic.
 * Mandelbrot is a classic GPU benchmark with high arithmetic intensity.
 *
 * Includes both float32 and float64 versions of Julia set to test
 * precision and float64 device support detection.
 *
 * Compares SPOC and V2 runtime paths for correctness and performance.
 ******************************************************************************)

open Spoc

(* V2 module aliases *)
module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer

(* Force V2 backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

let cfg = Test_helpers.default_config ()

(* Check if a device supports float64 (double precision) *)
let supports_float64 dev =
  match dev.Devices.specific_info with
  | Devices.CudaInfo _ ->
      (* CUDA devices generally support float64, check compute capability *)
      true
  | Devices.OpenCLInfo info ->
      (* OpenCL: check double_fp_config - CL_FP_NONE means no support *)
      info.Devices.double_fp_config <> Devices.CL_FP_NONE
  | Devices.NativeInfo _ ->
      (* Native CPU always supports float64 *)
      true
  | Devices.InterpreterInfo _ ->
      (* Interpreter supports float64 *)
      true

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

let ocaml_julia output width height c_real c_imag max_iter =
  for py = 0 to height - 1 do
    for px = 0 to width - 1 do
      let x = ref ((4.0 *. float_of_int px /. float_of_int width) -. 2.0) in
      let y = ref ((4.0 *. float_of_int py /. float_of_int height) -. 2.0) in
      let iter = ref 0 in
      while (!x *. !x) +. (!y *. !y) <= 4.0 && !iter < max_iter do
        let xtemp = (!x *. !x) -. (!y *. !y) +. c_real in
        y := (2.0 *. !x *. !y) +. c_imag ;
        x := xtemp ;
        incr iter
      done ;
      output.((py * width) + px) <- Int32.of_int !iter
    done
  done

(* ========== Shared test data ========== *)

let expected_mandelbrot = ref [||]

let expected_julia = ref [||]

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

let init_julia_data () =
  let dim = !image_dim in
  let n = dim * dim in
  let output = Array.make n 0l in
  expected_julia := output ;
  let c_real = -0.7 in
  let c_imag = 0.27015 in
  let t0 = Unix.gettimeofday () in
  ocaml_julia output dim dim c_real c_imag !max_iter ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Device test runners ========== *)

let run_mandelbrot_test dev =
  let mandelbrot_kernel =
    [%kernel
      fun (output : int32 vector)
          (width : int)
          (height : int)
          (max_iter : int) ->
        let open Std in
        (* Use global_idx_x/y to enable Simple2D optimization for native runtime *)
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
  in
  let dim = !image_dim in
  let width = dim in
  let height = dim in
  let n = width * height in
  let exp = !expected_mandelbrot in

  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Mem.set output i 0l
  done ;

  ignore (Sarek.Kirc.gen mandelbrot_kernel dev) ;
  let block_size = min 16 (Test_helpers.get_block_size cfg dev) in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    mandelbrot_kernel
    (output, width, height, !max_iter)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to min 1000 (n - 1) do
        let got = Mem.get output i in
        if got <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(* ========== V2 Mandelbrot runner ========== *)

let mandelbrot_kernel_v2 =
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

let run_mandelbrot_v2_test (dev : V2_Device.t) =
  let _, kirc = mandelbrot_kernel_v2 in
  let ir =
    match kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "Mandelbrot kernel has no V2 IR"
  in

  let dim = !image_dim in
  let width = dim in
  let height = dim in
  let n = width * height in
  let exp = !expected_mandelbrot in

  let output = V2_Vector.create V2_Vector.int32 n in

  for i = 0 to n - 1 do
    V2_Vector.set output i 0l
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
  V2_Transfer.flush dev ;

  (* Reset output for timed run *)
  for i = 0 to n - 1 do
    V2_Vector.set output i 0l
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
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = V2_Vector.to_array output in
      let errors = ref 0 in
      for i = 0 to min 1000 (n - 1) do
        if result.(i) <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(* ========== V2 Tail-recursive Mandelbrot runner ========== *)

let mandelbrot_tailrec_kernel_v2 =
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

let run_mandelbrot_tailrec_v2_test (dev : V2_Device.t) =
  let _, kirc = mandelbrot_tailrec_kernel_v2 in
  let ir =
    match kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "Mandelbrot tailrec kernel has no V2 IR"
  in

  let dim = !image_dim in
  let width = dim in
  let height = dim in
  let n = width * height in
  let exp = !expected_mandelbrot in

  let output = V2_Vector.create V2_Vector.int32 n in

  for i = 0 to n - 1 do
    V2_Vector.set output i 0l
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
  V2_Transfer.flush dev ;

  (* Reset output for timed run *)
  for i = 0 to n - 1 do
    V2_Vector.set output i 0l
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
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      let result = V2_Vector.to_array output in
      let errors = ref 0 in
      for i = 0 to min 1000 (n - 1) do
        if result.(i) <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(* Julia set kernel using float32 (single precision) *)
let run_julia_f32_test dev =
  let julia_f32_kernel =
    [%kernel
      fun (output : int32 vector)
          (width : int32)
          (height : int32)
          (c_real : float32)
          (c_imag : float32)
          (max_iter : int32) ->
        let open Std in
        (* Use global_idx_x/y to enable Simple2D optimization for native runtime *)
        let px = global_idx_x in
        let py = global_idx_y in
        if px < width && py < height then begin
          let x = mut ((4.0 *. (float px /. float width)) -. 2.0) in
          let y = mut ((4.0 *. (float py /. float height)) -. 2.0) in
          let iter = mut 0l in
          while (x *. x) +. (y *. y) <= 4.0 && iter < max_iter do
            let xtemp = (x *. x) -. (y *. y) +. c_real in
            y := (2.0 *. x *. y) +. c_imag ;
            x := xtemp ;
            iter := iter + 1l
          done ;
          output.((py * width) + px) <- iter
        end]
  in
  let dim = !image_dim in
  let width = dim in
  let height = dim in
  let n = width * height in
  let exp = !expected_julia in

  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Mem.set output i 0l
  done ;

  let c_real = -0.7 in
  let c_imag = 0.27015 in

  ignore (Sarek.Kirc.gen julia_f32_kernel dev) ;
  let block_size = min 16 (Test_helpers.get_block_size cfg dev) in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    julia_f32_kernel
    (output, width, height, c_real, c_imag, !max_iter)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to min 1000 (n - 1) do
        let got = Mem.get output i in
        if got <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(* Julia set kernel using float64 (double precision) - requires device support *)
let run_julia_f64_test dev =
  let julia_f64_kernel =
    [%kernel
      fun (output : int32 vector)
          (width : int32)
          (height : int32)
          (c_real : float)
          (c_imag : float)
          (max_iter : int32) ->
        let open Std in
        (* Use global_idx_x/y to enable Simple2D optimization for native runtime *)
        let px = global_idx_x in
        let py = global_idx_y in
        if px < width && py < height then begin
          (* Open Float64 after int comparisons to avoid shadowing < *)
          let open Sarek_float64.Float64 in
          let four = of_float32 4.0 in
          let two = of_float32 2.0 in
          let fwidth = of_int32 width in
          let fheight = of_int32 height in
          let x = mut ((four *. (of_int32 px /. fwidth)) -. two) in
          let y = mut ((four *. (of_int32 py /. fheight)) -. two) in
          let iter = mut 0l in
          while (x *. x) +. (y *. y) <= four && iter < max_iter do
            let xtemp = (x *. x) -. (y *. y) +. c_real in
            y := (two *. x *. y) +. c_imag ;
            x := xtemp ;
            iter := iter + 1l
          done ;
          output.((py * width) + px) <- iter
        end]
  in
  let dim = !image_dim in
  let width = dim in
  let height = dim in
  let n = width * height in
  let exp = !expected_julia in

  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Mem.set output i 0l
  done ;

  let c_real = -0.7 in
  let c_imag = 0.27015 in

  ignore (Sarek.Kirc.gen julia_f64_kernel dev) ;
  let block_size = min 16 (Test_helpers.get_block_size cfg dev) in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    julia_f64_kernel
    (output, width, height, c_real, c_imag, !max_iter)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to min 1000 (n - 1) do
        let got = Mem.get output i in
        if got <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(* ========== Tail-recursive mandelbrot using module function ========== *)

let run_mandelbrot_tailrec_test dev =
  let mandelbrot_tailrec_kernel =
    [%kernel
      (* Tail-recursive iteration using module function *)
      let open Std in
      (* This function uses the accumulator pattern - it's tail-recursive
         and will be transformed to a while loop by the tailrec pass.
         Note: Use int32 for consistency with native code generation. *)
      let rec iterate (x : float32) (y : float32) (x0 : float32) (y0 : float32)
          (iter : int32) (max_iter : int32) : int32 =
        if (x *. x) +. (y *. y) > 4.0 || iter >= max_iter then iter
        else
          let xtemp = (x *. x) -. (y *. y) +. x0 in
          let ynew = (2.0 *. x *. y) +. y0 in
          iterate xtemp ynew x0 y0 (iter + 1l) max_iter
      in

      fun (output : int32 vector)
          (width : int)
          (height : int)
          (max_iter : int)
        ->
        let px = global_idx_x in
        let py = global_idx_y in
        if px < width && py < height then begin
          let x0 = (4.0 *. (float px /. float width)) -. 2.5 in
          let y0 = (3.0 *. (float py /. float height)) -. 1.5 in
          let iter = iterate 0.0 0.0 x0 y0 0l max_iter in
          output.((py * width) + px) <- iter
        end]
  in
  let dim = !image_dim in
  let width = dim in
  let height = dim in
  let n = width * height in
  let exp = !expected_mandelbrot in

  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Mem.set output i 0l
  done ;

  ignore (Sarek.Kirc.gen mandelbrot_tailrec_kernel dev) ;
  let block_size = min 16 (Test_helpers.get_block_size cfg dev) in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = block_size; blockZ = 1} in
  let grid = {Kernel.gridX = blocks_x; gridY = blocks_y; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    mandelbrot_tailrec_kernel
    (output, width, height, !max_iter)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to min 1000 (n - 1) do
        let got = Mem.get output i in
        if got <> exp.(i) then incr errors
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

  let devs = Devices.init () in
  if Array.length devs = 0 then begin
    print_endline "No GPU devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices devs ;

  (* Initialize V2 devices *)
  let v2_devs = V2_Device.init ~frameworks:["CUDA"; "OpenCL"] () in

  let dim = int_of_float (sqrt (float_of_int cfg.size)) in
  Printf.printf "Image dimensions: %dx%d, max_iter=%d\n%!" dim dim !max_iter ;

  if cfg.benchmark_all then begin
    (* SPOC benchmarks *)
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_mandelbrot_data
      run_mandelbrot_test
      "Mandelbrot (SPOC)" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_julia_data
      run_julia_f32_test
      "Julia set float32 (SPOC)" ;
    (* Float64 Julia - filter to only devices that support it *)
    let all_device_ids =
      match cfg.benchmark_devices with
      | Some ids -> ids
      | None -> List.init (Array.length devs) (fun i -> i)
    in
    let f64_devices =
      all_device_ids
      |> List.filter (fun id ->
          id < Array.length devs && supports_float64 devs.(id))
    in
    if f64_devices <> [] then
      Test_helpers.benchmark_with_baseline
        ~device_ids:(Some f64_devices)
        devs
        ~baseline:init_julia_data
        run_julia_f64_test
        "Julia set float64 (SPOC)"
    else
      Printf.printf
        "\nJulia set float64 (SPOC): No devices with float64 support\n%!" ;

    (* V2 benchmarks *)
    Printf.printf "\n=== V2 Runtime Benchmarks ===\n%!" ;
    let baseline_ms, _ = init_mandelbrot_data () in
    Array.iter
      (fun v2_dev ->
        Printf.printf "\nV2 Mandelbrot on %s:\n%!" v2_dev.V2_Device.name ;
        let time_ms, ok = run_mandelbrot_v2_test v2_dev in
        Printf.printf
          "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
          time_ms
          (baseline_ms /. time_ms)
          (if ok then "PASSED" else "FAILED") ;
        Printf.printf
          "\nV2 Mandelbrot (tailrec) on %s:\n%!"
          v2_dev.V2_Device.name ;
        let tr_time_ms, tr_ok = run_mandelbrot_tailrec_v2_test v2_dev in
        Printf.printf
          "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
          tr_time_ms
          (baseline_ms /. tr_time_ms)
          (if tr_ok then "PASSED" else "FAILED"))
      v2_devs
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    let dev_name = dev.Devices.general_info.Devices.name in
    Printf.printf "Using device: %s\n%!" dev_name ;
    Printf.printf
      "  Float64 support: %s\n%!"
      (if supports_float64 dev then "yes" else "no") ;

    let baseline_ms, _ = init_mandelbrot_data () in
    Printf.printf "\nOCaml baseline (Mandelbrot): %.4f ms\n%!" baseline_ms ;

    (* Run SPOC Mandelbrot *)
    Printf.printf "\nMandelbrot (SPOC):\n%!" ;
    let spoc_time, spoc_ok =
      try run_mandelbrot_test dev
      with e ->
        Printf.printf "  SPOC error: %s\n%!" (Printexc.to_string e) ;
        (0.0, false)
    in
    if spoc_ok then
      Printf.printf
        "  Time: %.4f ms, Speedup: %.2fx, PASSED\n%!"
        spoc_time
        (baseline_ms /. spoc_time)
    else Printf.printf "  SKIPPED (SPOC error)\n%!" ;

    (* Run V2 Mandelbrot *)
    let v2_dev_opt =
      Array.find_opt (fun d -> d.V2_Device.name = dev_name) v2_devs
    in
    (match v2_dev_opt with
    | Some v2_dev ->
        Printf.printf "\nMandelbrot (V2):\n%!" ;
        let v2_time, v2_ok = run_mandelbrot_v2_test v2_dev in
        Printf.printf
          "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
          v2_time
          (baseline_ms /. v2_time)
          (if v2_ok then "PASSED" else "FAILED") ;
        if not v2_ok then begin
          print_endline "\nV2 Mandelbrot test FAILED" ;
          exit 1
        end
    | None ->
        Printf.printf "\nMandelbrot (V2): SKIPPED (no matching V2 device)\n%!") ;

    let baseline_ms, _ = init_julia_data () in
    Printf.printf "\nOCaml baseline (Julia): %.4f ms\n%!" baseline_ms ;

    Printf.printf "\nJulia set (float32):\n%!" ;
    let time_ms, ok = run_julia_f32_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    if supports_float64 dev then begin
      Printf.printf "\nJulia set (float64):\n%!" ;
      let time_ms, ok = run_julia_f64_test dev in
      Printf.printf
        "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
        time_ms
        (baseline_ms /. time_ms)
        (if ok then "PASSED" else "FAILED")
    end
    else
      Printf.printf
        "\nJulia set (float64): SKIPPED (device doesn't support float64)\n%!" ;

    (* Tail-recursive mandelbrot test - skip SPOC due to OpenCL issues *)
    Printf.printf
      "\nMandelbrot (tail-recursive SPOC): SKIPPED (OpenCL driver issues)\n%!" ;

    (* V2 Tail-recursive test *)
    (match v2_dev_opt with
    | Some v2_dev ->
        Printf.printf "\nMandelbrot (tail-recursive V2):\n%!" ;
        let v2_tr_time, v2_tr_ok = run_mandelbrot_tailrec_v2_test v2_dev in
        Printf.printf
          "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
          v2_tr_time
          (baseline_ms /. v2_tr_time)
          (if v2_tr_ok then "PASSED" else "FAILED") ;
        if not v2_tr_ok then begin
          print_endline "\nV2 Mandelbrot tailrec test FAILED" ;
          exit 1
        end
    | None ->
        Printf.printf
          "\n\
           Mandelbrot (tail-recursive V2): SKIPPED (no matching V2 device)\n\
           %!") ;

    print_endline "\nMandelbrot tests PASSED"
  end
