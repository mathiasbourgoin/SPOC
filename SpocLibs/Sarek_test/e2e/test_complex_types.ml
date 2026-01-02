(******************************************************************************
 * E2E test for Sarek PPX - Complex types
 *
 * Tests custom record types with both SPOC and V2 execution paths.
 * Uses [@@sarek.type] for V2-compatible custom vector types.
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

type float32 = float

(* Type definitions with [@@sarek.type] for V2 support *)
type point2d = {px : float32; py : float32} [@@sarek.type]

type point3d = {x : float32; y : float32; z : float32} [@@sarek.type]

type particle = {
  pos_x : float32;
  pos_y : float32;
  vel_x : float32;
  vel_y : float32;
  mass : float32;
}
[@@sarek.type]

type color = {r : float32; g : float32; b : float32; a : float32} [@@sarek.type]

(* ============================================================================
   Point2D distance - uses point2d struct vector
   ============================================================================ *)

let point2d_distance_kirc =
  snd
    [%kernel
      fun (points : point2d vector) (distances : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = points.(tid) in
          distances.(tid) <- sqrt ((p.px *. p.px) +. (p.py *. p.py))
        end]

let run_point2d_test dev =
  Printf.printf "  SPOC codegen: %!" ;
  let t0 = Unix.gettimeofday () in
  let kern =
    [%kernel
      fun (points : point2d vector) (distances : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = points.(tid) in
          distances.(tid) <- sqrt ((p.px *. p.px) +. (p.py *. p.py))
        end]
  in
  ignore (Sarek.Kirc.gen kern dev) ;
  let t1 = Unix.gettimeofday () in
  Printf.printf "%.2f ms\n%!" ((t1 -. t0) *. 1000.0) ;
  (true, (t1 -. t0) *. 1000.0)

let run_point2d_test_v2 dev n =
  let points = V2_Vector.create_custom point2d_custom_v2 n in
  let distances = V2_Vector.create V2_Vector.float32 n in
  for i = 0 to n - 1 do
    V2_Vector.set points i {px = float_of_int i; py = float_of_int (n - i)} ;
    V2_Vector.set distances i 0.0
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let ir =
    match point2d_distance_kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in
  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    ~ir
    ~args:
      [
        Sarek.Execute.Vec points;
        Sarek.Execute.Vec distances;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (* Verify *)
  let ok = ref true in
  for i = 0 to n - 1 do
    let p = V2_Vector.get points i in
    let expected = sqrt ((p.px *. p.px) +. (p.py *. p.py)) in
    let got = V2_Vector.get distances i in
    if abs_float (got -. expected) > 1e-3 then (
      ok := false ;
      if i < 3 then
        Printf.printf
          "    Mismatch at %d: got %f expected %f\n%!"
          i
          got
          expected)
  done ;
  (!ok, time_ms)

(* ============================================================================
   Point3D normalize - uses point3d vector
   ============================================================================ *)

let point3d_normalize_kirc =
  snd
    [%kernel
      fun (points : point3d vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = points.(tid) in
          let len = sqrt ((p.x *. p.x) +. (p.y *. p.y) +. (p.z *. p.z)) in
          if len > 0.0 then begin
            let nx = p.x /. len in
            let ny = p.y /. len in
            let nz = p.z /. len in
            points.(tid) <- {x = nx; y = ny; z = nz}
          end
        end]

let run_point3d_test dev =
  Printf.printf "  SPOC codegen: %!" ;
  let t0 = Unix.gettimeofday () in
  let kern =
    [%kernel
      fun (points : point3d vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = points.(tid) in
          let len = sqrt ((p.x *. p.x) +. (p.y *. p.y) +. (p.z *. p.z)) in
          if len > 0.0 then
            points.(tid) <- {x = p.x /. len; y = p.y /. len; z = p.z /. len}
        end]
  in
  ignore (Sarek.Kirc.gen kern dev) ;
  let t1 = Unix.gettimeofday () in
  Printf.printf "%.2f ms\n%!" ((t1 -. t0) *. 1000.0) ;
  (true, (t1 -. t0) *. 1000.0)

let run_point3d_test_v2 dev n =
  let points = V2_Vector.create_custom point3d_custom_v2 n in
  for i = 0 to n - 1 do
    let fi = float_of_int i in
    V2_Vector.set points i {x = fi; y = fi *. 2.0; z = fi *. 3.0}
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let ir =
    match point3d_normalize_kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in
  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    ~ir
    ~args:[Sarek.Execute.Vec points; Sarek.Execute.Int32 (Int32.of_int n)]
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (* Verify - normalized vectors should have length ~1.0 *)
  let ok = ref true in
  for i = 1 to min (n - 1) 10 do
    let p = V2_Vector.get points i in
    let len = sqrt ((p.x *. p.x) +. (p.y *. p.y) +. (p.z *. p.z)) in
    if abs_float (len -. 1.0) > 1e-3 then (
      ok := false ;
      if i < 3 then
        Printf.printf "    Point %d: len=%.4f (expected 1.0)\n%!" i len)
  done ;
  (!ok, time_ms)

(* ============================================================================
   Particle update - uses particle vector
   ============================================================================ *)

let particle_update_kirc =
  snd
    [%kernel
      fun (particles : particle vector) (dt : float32) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = particles.(tid) in
          let new_pos_x = p.pos_x +. (p.vel_x *. dt) in
          let new_pos_y = p.pos_y +. (p.vel_y *. dt) in
          particles.(tid) <-
            {
              pos_x = new_pos_x;
              pos_y = new_pos_y;
              vel_x = p.vel_x;
              vel_y = p.vel_y;
              mass = p.mass;
            }
        end]

let run_particle_test dev =
  Printf.printf "  SPOC codegen: %!" ;
  let t0 = Unix.gettimeofday () in
  let kern =
    [%kernel
      fun (particles : particle vector) (dt : float32) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = particles.(tid) in
          particles.(tid) <-
            {
              pos_x = p.pos_x +. (p.vel_x *. dt);
              pos_y = p.pos_y +. (p.vel_y *. dt);
              vel_x = p.vel_x;
              vel_y = p.vel_y;
              mass = p.mass;
            }
        end]
  in
  ignore (Sarek.Kirc.gen kern dev) ;
  let t1 = Unix.gettimeofday () in
  Printf.printf "%.2f ms\n%!" ((t1 -. t0) *. 1000.0) ;
  (true, (t1 -. t0) *. 1000.0)

let run_particle_test_v2 dev n =
  let particles = V2_Vector.create_custom particle_custom_v2 n in
  let dt = 0.1 in
  for i = 0 to n - 1 do
    let fi = float_of_int i in
    V2_Vector.set
      particles
      i
      {pos_x = fi; pos_y = fi *. 2.0; vel_x = 1.0; vel_y = 2.0; mass = 1.0}
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let ir =
    match particle_update_kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in
  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    ~ir
    ~args:
      [
        Sarek.Execute.Vec particles;
        Sarek.Execute.Float32 dt;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (* Verify - position should be updated by vel * dt *)
  let ok = ref true in
  for i = 0 to min (n - 1) 10 do
    let fi = float_of_int i in
    let p = V2_Vector.get particles i in
    let expected_x = fi +. (1.0 *. dt) in
    let expected_y = (fi *. 2.0) +. (2.0 *. dt) in
    if
      abs_float (p.pos_x -. expected_x) > 1e-3
      || abs_float (p.pos_y -. expected_y) > 1e-3
    then (
      ok := false ;
      if i < 3 then
        Printf.printf
          "    Particle %d: pos=(%.2f,%.2f) expected=(%.2f,%.2f)\n%!"
          i
          p.pos_x
          p.pos_y
          expected_x
          expected_y)
  done ;
  (!ok, time_ms)

(* ============================================================================
   Color blend - uses color vector
   ============================================================================ *)

let color_blend_kirc =
  snd
    [%kernel
      fun (c1 : color vector)
          (c2 : color vector)
          (output : color vector)
          (t : float32)
          (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let a = c1.(tid) in
          let b = c2.(tid) in
          let one_minus_t = 1.0 -. t in
          output.(tid) <-
            {
              r = (a.r *. one_minus_t) +. (b.r *. t);
              g = (a.g *. one_minus_t) +. (b.g *. t);
              b = (a.b *. one_minus_t) +. (b.b *. t);
              a = (a.a *. one_minus_t) +. (b.a *. t);
            }
        end]

let run_color_test dev =
  Printf.printf "  SPOC codegen: %!" ;
  let t0 = Unix.gettimeofday () in
  let kern =
    [%kernel
      fun (c1 : color vector)
          (c2 : color vector)
          (out : color vector)
          (t : float32)
          (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let a = c1.(tid) in
          let b = c2.(tid) in
          let omt = 1.0 -. t in
          out.(tid) <-
            {
              r = (a.r *. omt) +. (b.r *. t);
              g = (a.g *. omt) +. (b.g *. t);
              b = (a.b *. omt) +. (b.b *. t);
              a = (a.a *. omt) +. (b.a *. t);
            }
        end]
  in
  ignore (Sarek.Kirc.gen kern dev) ;
  let t1 = Unix.gettimeofday () in
  Printf.printf "%.2f ms\n%!" ((t1 -. t0) *. 1000.0) ;
  (true, (t1 -. t0) *. 1000.0)

let run_color_test_v2 dev n =
  let c1 = V2_Vector.create_custom color_custom_v2 n in
  let c2 = V2_Vector.create_custom color_custom_v2 n in
  let output = V2_Vector.create_custom color_custom_v2 n in
  let t = 0.5 in
  for i = 0 to n - 1 do
    V2_Vector.set c1 i {r = 1.0; g = 0.0; b = 0.0; a = 1.0} ;
    V2_Vector.set c2 i {r = 0.0; g = 1.0; b = 0.0; a = 1.0} ;
    V2_Vector.set output i {r = 0.0; g = 0.0; b = 0.0; a = 0.0}
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let ir =
    match color_blend_kirc.Sarek.Kirc.body_v2 with
    | Some ir -> ir
    | None -> failwith "Kernel has no V2 IR"
  in
  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    ~ir
    ~args:
      [
        Sarek.Execute.Vec c1;
        Sarek.Execute.Vec c2;
        Sarek.Execute.Vec output;
        Sarek.Execute.Float32 t;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  V2_Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (* Verify - blended color at t=0.5 should be (0.5, 0.5, 0.0, 1.0) *)
  let ok = ref true in
  for i = 0 to min (n - 1) 10 do
    let c = V2_Vector.get output i in
    if abs_float (c.r -. 0.5) > 1e-3 || abs_float (c.g -. 0.5) > 1e-3 then (
      ok := false ;
      if i < 3 then
        Printf.printf
          "    Color %d: (%.2f,%.2f,%.2f,%.2f) expected (0.5,0.5,0.0,1.0)\n%!"
          i
          c.r
          c.g
          c.b
          c.a)
  done ;
  (!ok, time_ms)

(* ============================================================================
   Main
   ============================================================================ *)

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

  let dev = Test_helpers.get_device cfg devs in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

  print_endline "\n=== SPOC Path ===" ;
  print_endline "Point2D distance:" ;
  ignore (run_point2d_test dev) ;
  print_endline "Point3D normalize:" ;
  ignore (run_point3d_test dev) ;
  print_endline "Particle update:" ;
  ignore (run_particle_test dev) ;
  print_endline "Color blend:" ;
  ignore (run_color_test dev) ;

  (* V2 execution tests *)
  print_endline "\n=== V2 Path ===" ;
  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length v2_devs = 0 then
    print_endline "No V2 devices found - skipping V2 tests"
  else begin
    let v2_dev = v2_devs.(0) in
    Printf.printf "V2 device: %s\n%!" v2_dev.V2_Device.name ;
    let n = 1024 in

    print_endline "Point2D distance:" ;
    (try
       let ok, time = run_point2d_test_v2 v2_dev n in
       Printf.printf
         "  V2 exec: %.2f ms, %s\n%!"
         time
         (if ok then "PASSED" else "FAILED")
     with e -> Printf.printf "  FAIL (%s)\n%!" (Printexc.to_string e)) ;

    print_endline "Point3D normalize:" ;
    (try
       let ok, time = run_point3d_test_v2 v2_dev n in
       Printf.printf
         "  V2 exec: %.2f ms, %s\n%!"
         time
         (if ok then "PASSED" else "FAILED")
     with e -> Printf.printf "  FAIL (%s)\n%!" (Printexc.to_string e)) ;

    print_endline "Particle update:" ;
    (try
       let ok, time = run_particle_test_v2 v2_dev n in
       Printf.printf
         "  V2 exec: %.2f ms, %s\n%!"
         time
         (if ok then "PASSED" else "FAILED")
     with e -> Printf.printf "  FAIL (%s)\n%!" (Printexc.to_string e)) ;

    print_endline "Color blend:" ;
    try
      let ok, time = run_color_test_v2 v2_dev n in
      Printf.printf
        "  V2 exec: %.2f ms, %s\n%!"
        time
        (if ok then "PASSED" else "FAILED")
    with e -> Printf.printf "  FAIL (%s)\n%!" (Printexc.to_string e)
  end
