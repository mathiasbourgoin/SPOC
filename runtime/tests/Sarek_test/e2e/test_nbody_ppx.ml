(******************************************************************************
 * Simplified N-Body example using Sarek PPX with ktype + helper.
 * Uses let x = mut ... syntax for mutable accumulators.
 * V2 runtime only.
 ******************************************************************************)

(* Shadow SPOC's vector with V2 vector for kernel compatibility *)
type ('a, 'b) vector = ('a, 'b) Spoc_core.Vector.t

(* V2 module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Force V2 backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

let () =
  let accel_kernel =
    [%kernel
      let module Types = struct
        type particle = {x : float32; y : float32; z : float32}
      end in
      let make_p (x : float32) (y : float32) (z : float32) : particle =
        {x; y; z}
      in
      fun (xs : float32 vector)
          (ys : float32 vector)
          (zs : float32 vector)
          (ax : float32 vector)
          (ay : float32 vector)
          (az : float32 vector)
          (n : int)
        ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then (
          let p = make_p xs.(tid) ys.(tid) zs.(tid) in
          let fx = mut 0.0 in
          let fy = mut 0.0 in
          let fz = mut 0.0 in
          let i = tid in
          for j = 0 to n - 1 do
            if j <> i then (
              let q = make_p xs.(j) ys.(j) zs.(j) in
              let dx = q.x -. p.x in
              let dy = q.y -. p.y in
              let dz = q.z -. p.z in
              let dist2 = (dx *. dx) +. (dy *. dy) +. (dz *. dz) +. 1e-9 in
              let inv = 1.0 /. (sqrt dist2 *. dist2) in
              fx := fx +. (dx *. inv) ;
              fy := fy +. (dy *. inv) ;
              fz := fz +. (dz *. inv))
          done ;
          ax.(tid) <- fx ;
          ay.(tid) <- fy ;
          az.(tid) <- fz)]
  in

  let _, kirc_kernel = accel_kernel in
  print_endline "=== NBody (PPX) IR ===" ;
  Sarek.Kirc_Ast.print_ast kirc_kernel.Sarek.Kirc_types.body ;
  print_endline "======================" ;

  print_endline "\n=== Running V2 path ===" ;
  let v2_devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length v2_devs = 0 then (
    print_endline "V2: No device found - IR test passed" ;
    exit 0) ;

  let dev = v2_devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Device.name ;

  (match kirc_kernel.Sarek.Kirc_types.body_ir with
  | None ->
      print_endline "V2: No V2 IR available - IR test passed" ;
      exit 0
  | Some ir ->
      let n = 64 in
      let threads = 64 in
      let grid_x = (n + threads - 1) / threads in

      (* Initialize positions *)
      let bax = Array.init n (fun i -> float_of_int (i mod 7) /. 10.0) in
      let bay = Array.init n (fun i -> float_of_int (i mod 5) /. 10.0) in
      let baz = Array.init n (fun i -> float_of_int (i mod 3) /. 10.0) in

      let v2_xs = Vector.create Vector.float32 n in
      let v2_ys = Vector.create Vector.float32 n in
      let v2_zs = Vector.create Vector.float32 n in
      let v2_ax = Vector.create Vector.float32 n in
      let v2_ay = Vector.create Vector.float32 n in
      let v2_az = Vector.create Vector.float32 n in
      for i = 0 to n - 1 do
        Vector.set v2_xs i bax.(i) ;
        Vector.set v2_ys i bay.(i) ;
        Vector.set v2_zs i baz.(i) ;
        Vector.set v2_ax i 0.0 ;
        Vector.set v2_ay i 0.0 ;
        Vector.set v2_az i 0.0
      done ;

      let block = Sarek.Execute.dims1d threads in
      let grid = Sarek.Execute.dims1d grid_x in

      Sarek.Execute.run_vectors
        ~device:dev
        ~ir
        ~args:
          [
            Sarek.Execute.Vec v2_xs;
            Sarek.Execute.Vec v2_ys;
            Sarek.Execute.Vec v2_zs;
            Sarek.Execute.Vec v2_ax;
            Sarek.Execute.Vec v2_ay;
            Sarek.Execute.Vec v2_az;
            Sarek.Execute.Int n;
          ]
        ~block
        ~grid
        () ;
      Transfer.flush dev ;

      (* Verify results against CPU reference *)
      let v2_ok = ref true in
      for i = 0 to n - 1 do
        let px = bax.(i) and py = bay.(i) and pz = baz.(i) in
        let rfx = ref 0.0 and rfy = ref 0.0 and rfz = ref 0.0 in
        for j = 0 to n - 1 do
          if j <> i then (
            let qx = bax.(j) and qy = bay.(j) and qz = baz.(j) in
            let dx = qx -. px in
            let dy = qy -. py in
            let dz = qz -. pz in
            let dist2 = (dx *. dx) +. (dy *. dy) +. (dz *. dz) +. 1e-9 in
            let inv = 1.0 /. (sqrt dist2 *. dist2) in
            rfx := !rfx +. (dx *. inv) ;
            rfy := !rfy +. (dy *. inv) ;
            rfz := !rfz +. (dz *. inv))
        done ;
        let gx = Vector.get v2_ax i
        and gy = Vector.get v2_ay i
        and gz = Vector.get v2_az i in
        if
          abs_float (gx -. !rfx) > 1e-3
          || abs_float (gy -. !rfy) > 1e-3
          || abs_float (gz -. !rfz) > 1e-3
        then (
          v2_ok := false ;
          Printf.printf
            "V2 Mismatch at %d: GPU (%f,%f,%f) CPU (%f,%f,%f)\n%!"
            i
            gx
            gy
            gz
            !rfx
            !rfy
            !rfz)
      done ;
      if !v2_ok then print_endline "NBody PPX (V2) PASSED"
      else (
        print_endline "NBody PPX (V2) FAILED" ;
        exit 1)) ;
  print_endline "NBody PPX tests PASSED"
