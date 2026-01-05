(******************************************************************************
 * E2E test for Sarek PPX - Matrix Transpose
 *
 * Tests naive matrix transpose with 1D kernel.
 * Transpose is a memory-bound operation that benefits from coalescing.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std

(* Module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init () ;
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init ()

let cfg = Test_helpers.default_config ()

(* ========== Pure OCaml baseline ========== *)

let ocaml_transpose input output width height =
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let in_idx = (y * width) + x in
      let out_idx = (x * height) + y in
      output.(out_idx) <- input.(in_idx)
    done
  done

(* ========== Shared test data ========== *)

let input_data = ref [||]

let init_transpose_data size =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int size))) in
  let n = dim * dim in
  let inp = Array.init n (fun i -> float_of_int i) in
  input_data := inp

(* ========== Sarek kernel ========== *)

let transpose_naive_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int)
        (height : int) ->
      let open Std in
      let tid = global_thread_id in
      let n = width * height in
      if tid < n then begin
        let x = tid mod width in
        let y = tid / width in
        let in_idx = (y * width) + x in
        let out_idx = (x * height) + y in
        output.(out_idx) <- input.(in_idx)
      end]

(* ========== runtime test runner ========== *)

let run_transpose (dev : Device.t) size _block_size =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int size))) in
  let n = dim * dim in
  let _, kirc = transpose_naive_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Vector.set input i !input_data.(i) ;
    Vector.set output i 0.0
  done ;

  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int dim;
        Sarek.Execute.Int dim;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  ((t1 -. t0) *. 1000.0, Vector.to_array output)

(* ========== Main ========== *)

let () =
  Benchmarks.init () ;
  let size = Benchmarks.config.size in
  init_transpose_data size ;

  let baseline size =
    let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int size))) in
    let n = dim * dim in
    let inp = !input_data in
    let out = Array.make n 0.0 in
    ocaml_transpose inp out dim dim ;
    out
  in

  let verify result expected =
    let n = Array.length expected in
    let errors = ref 0 in
    let tolerance = 0.001 in
    for i = 0 to n - 1 do
      let diff = abs_float (result.(i) -. expected.(i)) in
      if diff > tolerance then begin
        if !errors < 5 then
          Printf.printf
            "  Mismatch at %d: expected %.2f, got %.2f\n"
            i
            expected.(i)
            result.(i) ;
        incr errors
      end
    done ;
    !errors = 0
  in

  Benchmarks.run ~baseline ~verify "Matrix Transpose" run_transpose ;
  Benchmarks.exit ()
