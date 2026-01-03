(******************************************************************************
 * E2E test: use a Sarek type registered in another module/file.
 * Uses V2 runtime only.
 ******************************************************************************)

module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

type float32 = float

(* Include types and functions from registered_defs.ml *)
let%sarek_include _ = "registered_defs.ml"

let () =
  let kernel =
    [%kernel
      fun (xs : float32 vector)
          (ys : float32 vector)
          (dst : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let open Registered_defs in
          let p : Registered_defs.vec2 = {x = xs.(tid); y = ys.(tid)} in
          dst.(tid) <- add_vec p]
  in
  let _, kirc = kernel in

  (* Print IR for verification *)
  print_endline "=== Cross-module Type IR ===" ;
  Sarek.Kirc_Ast.print_ast kirc.Sarek.Kirc_types.body ;
  print_endline "=============================" ;

  let devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then (
    print_endline "No devices found - IR generation test passed" ;
    exit 0) ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.V2_Device.name ;

  (* Get V2 IR and execute *)
  (match kirc.Sarek.Kirc_types.body_v2 with
  | None ->
      print_endline "No V2 IR - IR generation test PASSED"
  | Some ir ->
      let n = 64 in
      let xs = V2_Vector.create V2_Vector.float32 n in
      let ys = V2_Vector.create V2_Vector.float32 n in
      let dst = V2_Vector.create V2_Vector.float32 n in
      for i = 0 to n - 1 do
        V2_Vector.set xs i (float_of_int i) ;
        V2_Vector.set ys i (float_of_int (2 * i)) ;
        V2_Vector.set dst i 0.0
      done ;

      let threads = 64 in
      let grid_x = (n + threads - 1) / threads in
      Sarek.Execute.run_vectors
        ~device:dev
        ~ir
        ~args:
          [
            Sarek.Execute.Vec xs;
            Sarek.Execute.Vec ys;
            Sarek.Execute.Vec dst;
            Sarek.Execute.Int n;
          ]
        ~block:(Sarek.Execute.dims1d threads)
        ~grid:(Sarek.Execute.dims1d grid_x)
        () ;
      V2_Transfer.flush dev ;

      let ok = ref true in
      for i = 0 to n - 1 do
        let x = V2_Vector.get xs i in
        let y = V2_Vector.get ys i in
        let expected = x +. y in
        let got = V2_Vector.get dst i in
        if abs_float (got -. expected) > 1e-4 then (
          ok := false ;
          Printf.printf "Mismatch at %d: got %f expected %f\n%!" i got expected)
      done ;
      if !ok then print_endline "Cross-module registered type execution PASSED"
      else (
        print_endline "Cross-module registered type execution FAILED" ;
        exit 1))
