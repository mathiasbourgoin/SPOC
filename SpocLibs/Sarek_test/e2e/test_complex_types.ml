(******************************************************************************
 * E2E test for Sarek PPX - Complex types
 *
 * Tests custom record types, nested structs, and complex data structures.
 * This exercises inline type definitions and struct handling.
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

let cfg = Test_helpers.default_config ()

(** Point2D kernel - defined at top level for V2 access *)
let point2d_distance_kernel =
  [%kernel
    let module Types = struct
      type point2d = {x : float32; y : float32}
    end in
    fun (xs : float32 vector)
        (ys : float32 vector)
        (distances : float32 vector)
        (n : int32)
      ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let p : point2d = {x = xs.(tid); y = ys.(tid)} in
        let dx = p.x in
        let dy = p.y in
        distances.(tid) <- sqrt ((dx *. dx) +. (dy *. dy))
      end]

(** Run 2D point distance test - SPOC path *)
let run_point2d_test dev =
  Printf.printf "  Point2D distance: kernel generation test\n%!" ;
  let t0 = Unix.gettimeofday () in
  ignore (Sarek.Kirc.gen point2d_distance_kernel dev) ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (time_ms, true)

(** Run 2D point distance test - V2 path with execution *)
let run_point2d_test_v2 dev n =
  let xs = V2_Vector.create V2_Vector.float32 n in
  let ys = V2_Vector.create V2_Vector.float32 n in
  let distances = V2_Vector.create V2_Vector.float32 n in
  for i = 0 to n - 1 do
    V2_Vector.set xs i (float_of_int i) ;
    V2_Vector.set ys i (float_of_int (n - i)) ;
    V2_Vector.set distances i 0.0
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let _, kirc = point2d_distance_kernel in
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
        Sarek.Execute.Vec xs;
        Sarek.Execute.Vec ys;
        Sarek.Execute.Vec distances;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  V2_Transfer.flush dev ;
  (* Verify *)
  let ok = ref true in
  for i = 0 to n - 1 do
    let x = V2_Vector.get xs i in
    let y = V2_Vector.get ys i in
    let expected = sqrt ((x *. x) +. (y *. y)) in
    let got = V2_Vector.get distances i in
    if abs_float (got -. expected) > 1e-3 then (
      ok := false ;
      if i < 5 then
        Printf.printf "  Mismatch at %d: got %f expected %f\n%!" i got expected)
  done ;
  !ok

(** Run 3D point normalize test *)
let run_point3d_normalize_test dev =
  let point3d_normalize_kernel =
    [%kernel
      let module Types = struct
        type point3d = {x : float32; y : float32; z : float32}
      end in
      let make_point (x : float32) (y : float32) (z : float32) : point3d =
        {x; y; z}
      in
      fun (points : point3d vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = points.(tid) in
          let len = sqrt ((p.x *. p.x) +. (p.y *. p.y) +. (p.z *. p.z)) in
          if len > 0.0 then begin
            let nx = p.x /. len in
            let ny = p.y /. len in
            let nz = p.z /. len in
            points.(tid) <- make_point nx ny nz
          end
        end]
  in
  Printf.printf "  Point3D normalize: kernel generation test\n%!" ;
  let t0 = Unix.gettimeofday () in
  ignore (Sarek.Kirc.gen point3d_normalize_kernel dev) ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (time_ms, true)

(** Run particle update test *)
let run_particle_test dev =
  let particle_update_kernel =
    [%kernel
      let module Types = struct
        type particle = {
          pos_x : float32;
          pos_y : float32;
          vel_x : float32;
          vel_y : float32;
          mass : float32;
        }
      end in
      let make_particle (px : float32) (py : float32) (vx : float32)
          (vy : float32) (m : float32) : particle =
        {pos_x = px; pos_y = py; vel_x = vx; vel_y = vy; mass = m}
      in
      fun (particles : particle vector) (dt : float32) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = particles.(tid) in
          let new_pos_x = p.pos_x +. (p.vel_x *. dt) in
          let new_pos_y = p.pos_y +. (p.vel_y *. dt) in
          particles.(tid) <-
            make_particle new_pos_x new_pos_y p.vel_x p.vel_y p.mass
        end]
  in
  Printf.printf "  Particle update: kernel generation test\n%!" ;
  let t0 = Unix.gettimeofday () in
  ignore (Sarek.Kirc.gen particle_update_kernel dev) ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (time_ms, true)

(** Run color blend test *)
let run_color_blend_test dev =
  let color_blend_kernel =
    [%kernel
      let module Types = struct
        type color = {r : float32; g : float32; b : float32; a : float32}
      end in
      let make_color (r : float32) (g : float32) (b : float32) (a : float32) :
          color =
        {r; g; b; a}
      in
      fun (c1 : color vector)
          (c2 : color vector)
          (output : color vector)
          (t : float32)
          (n : int32)
        ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let a = c1.(tid) in
          let b = c2.(tid) in
          let one_minus_t = 1.0 -. t in
          let nr = (a.r *. one_minus_t) +. (b.r *. t) in
          let ng = (a.g *. one_minus_t) +. (b.g *. t) in
          let nb = (a.b *. one_minus_t) +. (b.b *. t) in
          let na = (a.a *. one_minus_t) +. (b.a *. t) in
          output.(tid) <- make_color nr ng nb na
        end]
  in
  Printf.printf "  Color blend: kernel generation test\n%!" ;
  let t0 = Unix.gettimeofday () in
  ignore (Sarek.Kirc.gen color_blend_kernel dev) ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (time_ms, true)

let () =
  let c = Test_helpers.parse_args "test_complex_types" in
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

  if cfg.benchmark_all then begin
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_point2d_test
      "Point2D distance" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_point3d_normalize_test
      "Point3D normalize" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_particle_test
      "Particle update" ;
    Test_helpers.benchmark_all
      ~device_ids:cfg.benchmark_devices
      devs
      run_color_blend_test
      "Color blend"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    Printf.printf "Testing complex types (kernel generation tests)\n%!" ;

    Printf.printf "\nPoint2D distance:\n%!" ;
    let time_ms, ok = run_point2d_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nPoint3D normalize:\n%!" ;
    let time_ms, ok = run_point3d_normalize_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nParticle update:\n%!" ;
    let time_ms, ok = run_particle_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nColor blend:\n%!" ;
    let time_ms, ok = run_color_blend_test dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nSPOC path tests PASSED"
  end ;

  (* V2 execution tests *)
  print_endline "\n=== V2 Path Tests ===" ;
  let v2_devs = V2_Device.init ~frameworks:["CUDA"; "OpenCL"] () in
  if Array.length v2_devs = 0 then
    print_endline "No V2 devices found - skipping V2 tests"
  else begin
    let v2_dev = v2_devs.(0) in
    Printf.printf "V2 device: %s\n%!" v2_dev.V2_Device.name ;

    Printf.printf "\nPoint2D distance (V2 execution):\n%!" ;
    (try
       let ok = run_point2d_test_v2 v2_dev 128 in
       Printf.printf "  %s\n%!" (if ok then "PASSED" else "FAILED")
     with e -> Printf.printf "  FAIL (%s)\n%!" (Printexc.to_string e)) ;

    print_endline "\nV2 path tests completed"
  end
