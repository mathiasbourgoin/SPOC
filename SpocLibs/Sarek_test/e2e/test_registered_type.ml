(******************************************************************************
 * E2E test: register a Sarek record type outside kernels via [%sarek.type].
 ******************************************************************************)

open Spoc
module V2_Vector = Sarek_core.Vector
module V2_Device = Sarek_core.Device
module V2_Transfer = Sarek_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

type float32 = float

type point = {x : float32; y : float32} [@@sarek.type]

let kernel =
  [%kernel
    fun (xs : float32 vector)
        (ys : float32 vector)
        (dst : float32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < n then
        let p = {x = xs.(tid); y = ys.(tid)} in
        dst.(tid) <- sqrt ((p.x *. p.x) +. (p.y *. p.y))]

let run_v2 dev n bax bay =
  let xv = V2_Vector.create V2_Vector.float32 n in
  let yv = V2_Vector.create V2_Vector.float32 n in
  let dst = V2_Vector.create V2_Vector.float32 n in
  for i = 0 to n - 1 do
    V2_Vector.set xv i (Bigarray.Array1.get bax i) ;
    V2_Vector.set yv i (Bigarray.Array1.get bay i) ;
    V2_Vector.set dst i 0.0
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let _, kirc = kernel in
  let ir =
    match kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    ~ir
    ~args:
      [
        Sarek.Execute.Vec xv;
        Sarek.Execute.Vec yv;
        Sarek.Execute.Vec dst;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  V2_Transfer.flush dev ;
  (* Verify *)
  let ok = ref true in
  for i = 0 to n - 1 do
    let x = V2_Vector.get xv i in
    let y = V2_Vector.get yv i in
    let expected = sqrt ((x *. x) +. (y *. y)) in
    let got = V2_Vector.get dst i in
    if abs_float (got -. expected) > 1e-3 then (
      ok := false ;
      if i < 5 then
        Printf.printf "  Mismatch at %d: got %f expected %f\n%!" i got expected)
  done ;
  !ok

let () =
  let _, kirc_kernel = kernel in
  print_endline "=== Registered type IR ===" ;
  Sarek.Kirc.print_ast kirc_kernel.Sarek.Kirc.body ;
  print_endline "==========================" ;

  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in

  if Array.length v2_devs = 0 then (
    print_endline "No V2 devices found - skipping execution" ;
    exit 0) ;

  Printf.printf "Using device: %s\n%!" v2_devs.(0).V2_Device.name ;

  let n = 128 in
  let bax = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let bay = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  for i = 0 to n - 1 do
    Bigarray.Array1.set bax i (float_of_int i) ;
    Bigarray.Array1.set bay i (float_of_int (n - i))
  done ;

  (* V2 execution *)
  print_string "V2: " ;
  flush stdout ;
  (try
     let ok = run_v2 v2_devs.(0) n bax bay in
     if ok then print_endline "PASSED" else print_endline "FAIL (verification)"
   with e -> Printf.printf "FAIL (%s)\n%!" (Printexc.to_string e)) ;

  (* SPOC execution - disabled (record types not yet supported in legacy path) *)
  print_endline "SPOC: SKIP (record types not supported)"
