(******************************************************************************
 * E2E test: ktype record + helper function executed on device.
 * Uses V2 runtime only.
 ******************************************************************************)

(* V2 module aliases *)
module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer

(* Type alias for kernel parameter annotations *)
type ('a, 'b) vector = ('a, 'b) V2_Vector.t

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

let () =
  let length_kernel =
    [%kernel
      let module Types = struct
        type point = {x : float32; y : float32}
      end in
      let make_point (x : float32) (y : float32) : point = {x; y} in
      fun (xv : float32 vector)
          (yv : float32 vector)
          (dst : float32 vector)
          (n : int32)
        ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let p = make_point xv.(tid) yv.(tid) in
          dst.(tid) <- sqrt ((p.x *. p.x) +. (p.y *. p.y))]
  in

  let _, kirc = length_kernel in
  print_endline "=== ktype helper IR ===" ;
  (match kirc.Sarek.Kirc_types.body_v2 with
  | Some ir -> Sarek.Sarek_ir.print_kernel ir
  | None -> print_endline "(No V2 IR available)") ;
  print_endline "=======================" ;

  (* Run with V2 runtime *)
  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length v2_devs = 0 then (
    print_endline "No device found - IR generation test passed" ;
    exit 0) ;
  let dev = v2_devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.V2_Device.name ;

  let n = 256 in

  (match kirc.Sarek.Kirc_types.body_v2 with
  | None ->
      print_endline "No V2 IR - SKIPPED"
  | Some ir ->
      let xv = V2_Vector.create V2_Vector.float32 n in
      let yv = V2_Vector.create V2_Vector.float32 n in
      let dst = V2_Vector.create V2_Vector.float32 n in
      for i = 0 to n - 1 do
        V2_Vector.set xv i (float_of_int i) ;
        V2_Vector.set yv i (float_of_int (n - i)) ;
        V2_Vector.set dst i 0.0
      done ;

      let threads = 128 in
      let grid_x = (n + threads - 1) / threads in
      let block = Sarek.Execute.dims1d threads in
      let grid = Sarek.Execute.dims1d grid_x in

      Sarek.Execute.run_vectors
        ~device:dev
        ~ir
        ~args:[
          Sarek.Execute.Vec xv;
          Sarek.Execute.Vec yv;
          Sarek.Execute.Vec dst;
          Sarek.Execute.Int n;
        ]
        ~block
        ~grid
        () ;
      V2_Transfer.flush dev ;

      (* Verify results *)
      let ok = ref true in
      for i = 0 to n - 1 do
        let x = V2_Vector.get xv i in
        let y = V2_Vector.get yv i in
        let expected = sqrt ((x *. x) +. (y *. y)) in
        let got = V2_Vector.get dst i in
        if abs_float (got -. expected) > 1e-3 then (
          ok := false ;
          Printf.printf "Mismatch at %d: got %f expected %f\n%!" i got expected)
      done ;
      if !ok then print_endline "Execution check PASSED"
      else (
        print_endline "Execution check FAILED" ;
        exit 1)) ;
  print_endline "test_ktype_helper PASSED"
