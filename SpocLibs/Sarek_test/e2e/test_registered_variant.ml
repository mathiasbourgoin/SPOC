(******************************************************************************
 * E2E test: register a Sarek variant type outside kernels via [@@sarek.type].
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

type color = Red | Value of float32 [@@sarek.type]

let () =
  let kernel =
    [%kernel
      fun (xs : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let c = if xs.(tid) > 0.0 then Value xs.(tid) else Red in
          match c with
          | Red -> dst.(tid) <- 0.0
          | Value v -> dst.(tid) <- v +. 1.0]
  in
  let _, kirc = kernel in

  (* Print IR for verification *)
  print_endline "=== Registered Variant IR ===" ;
  Sarek.Kirc_Ast.print_ast kirc.Sarek.Kirc_types.body ;
  print_endline "==============================" ;

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
      let n = 128 in
      let xs = V2_Vector.create V2_Vector.float32 n in
      let dst = V2_Vector.create V2_Vector.float32 n in
      for i = 0 to n - 1 do
        V2_Vector.set xs i (if i mod 2 = 0 then -1.0 else float_of_int i) ;
        V2_Vector.set dst i 0.0
      done ;

      let threads = 64 in
      let grid_x = (n + threads - 1) / threads in
      Sarek.Execute.run_vectors
        ~device:dev
        ~ir
        ~args:[Sarek.Execute.Vec xs; Sarek.Execute.Vec dst; Sarek.Execute.Int n]
        ~block:(Sarek.Execute.dims1d threads)
        ~grid:(Sarek.Execute.dims1d grid_x)
        () ;
      V2_Transfer.flush dev ;

      let ok = ref true in
      for i = 0 to n - 1 do
        let expected = if i mod 2 = 0 then 0.0 else float_of_int i +. 1.0 in
        let got = V2_Vector.get dst i in
        if abs_float (got -. expected) > 1e-3 then (
          ok := false ;
          Printf.printf "Mismatch at %d: got %f expected %f\n%!" i got expected)
      done ;
      if !ok then print_endline "Registered variant execution PASSED"
      else (
        print_endline "Registered variant execution FAILED" ;
        exit 1))
