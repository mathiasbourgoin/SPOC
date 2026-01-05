(******************************************************************************
 * E2E test for Sarek PPX - Complex types
 *
 * Tests custom record types with both SPOC and runtime execution paths.
 * Uses [@@sarek.type] for runtime-compatible custom vector types.
 ******************************************************************************)

[@@@warning "-32"]

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init () ;
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init ()

let cfg = Test_helpers.default_config ()

type float32 = float

(* Type definitions with [@@sarek.type] for runtime support *)
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

let run_point2d_test dev n =
  let points = Vector.create_custom point2d_custom n in
  let distances = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set points i {px = float_of_int i; py = float_of_int (n - i)} ;
    Vector.set distances i 0.0
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let ir =
    match point2d_distance_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
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
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (* Verify *)
  let ok = ref true in
  let error_count = ref 0 in
  let first_error_idx = ref (-1) in
  for i = 0 to n - 1 do
    let p = Vector.get points i in
    let expected = sqrt ((p.px *. p.px) +. (p.py *. p.py)) in
    let got = Vector.get distances i in
    (* Float32 has ~7 significant digits, use relative tolerance *)
    let tol = max 1e-4 (abs_float expected *. 1e-6) in
    if abs_float (got -. expected) > tol then (
      ok := false ;
      incr error_count ;
      if !first_error_idx < 0 then first_error_idx := i ;
      if !error_count <= 3 then
        Printf.printf
          "    Mismatch at %d: got %f expected %f (p={%.1f,%.1f})\n%!"
          i
          got
          expected
          p.px
          p.py)
  done ;
  if !error_count > 0 then
    Printf.printf "    Total errors: %d/%d, first at index %d\n%!" !error_count n !first_error_idx ;
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

let run_point3d_test dev n =
  let points = Vector.create_custom point3d_custom n in
  for i = 0 to n - 1 do
    let fi = float_of_int i in
    Vector.set points i {x = fi; y = fi *. 2.0; z = fi *. 3.0}
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let ir =
    match point3d_normalize_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in
  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    ~ir
    ~args:[Sarek.Execute.Vec points; Sarek.Execute.Int32 (Int32.of_int n)]
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (* Verify - normalized vectors should have length ~1.0 *)
  let ok = ref true in
  for i = 1 to min (n - 1) 10 do
    let p = Vector.get points i in
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

let run_particle_test dev n =
  let particles = Vector.create_custom particle_custom n in
  let dt = 0.1 in
  for i = 0 to n - 1 do
    let fi = float_of_int i in
    Vector.set
      particles
      i
      {pos_x = fi; pos_y = fi *. 2.0; vel_x = 1.0; vel_y = 2.0; mass = 1.0}
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let ir =
    match particle_update_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
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
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (* Verify - position should be updated by vel * dt *)
  let ok = ref true in
  for i = 0 to min (n - 1) 10 do
    let fi = float_of_int i in
    let p = Vector.get particles i in
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

let run_color_test dev n =
  let c1 = Vector.create_custom color_custom n in
  let c2 = Vector.create_custom color_custom n in
  let output = Vector.create_custom color_custom n in
  let t = 0.5 in
  for i = 0 to n - 1 do
    Vector.set c1 i {r = 1.0; g = 0.0; b = 0.0; a = 1.0} ;
    Vector.set c2 i {r = 0.0; g = 1.0; b = 0.0; a = 1.0} ;
    Vector.set output i {r = 0.0; g = 0.0; b = 0.0; a = 0.0}
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let ir =
    match color_blend_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
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
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (* Verify - blended color at t=0.5 should be (0.5, 0.5, 0.0, 1.0) *)
  let ok = ref true in
  for i = 0 to min (n - 1) 10 do
    let c = Vector.get output i in
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

let run_all_tests dev n =
  let all_ok = ref true in
  print_endline "\nPoint2D distance:" ;
  (try
     let ok, time = run_point2d_test dev n in
     Printf.printf
       "  %s (%s): %.2f ms, %s\n%!"
       dev.Device.name
       dev.Device.framework
       time
       (if ok then "OK" else "FAIL") ;
     if not ok then all_ok := false
   with e ->
     Printf.printf "  FAIL (%s)\n%!" (Printexc.to_string e) ;
     all_ok := false) ;

  print_endline "Point3D normalize:" ;
  (try
     let ok, time = run_point3d_test dev n in
     Printf.printf
       "  %s (%s): %.2f ms, %s\n%!"
       dev.Device.name
       dev.Device.framework
       time
       (if ok then "OK" else "FAIL") ;
     if not ok then all_ok := false
   with e ->
     Printf.printf "  FAIL (%s)\n%!" (Printexc.to_string e) ;
     all_ok := false) ;

  print_endline "Particle update:" ;
  (try
     let ok, time = run_particle_test dev n in
     Printf.printf
       "  %s (%s): %.2f ms, %s\n%!"
       dev.Device.name
       dev.Device.framework
       time
       (if ok then "OK" else "FAIL") ;
     if not ok then all_ok := false
   with e ->
     Printf.printf "  FAIL (%s)\n%!" (Printexc.to_string e) ;
     all_ok := false) ;

  print_endline "Color blend:" ;
  (try
     let ok, time = run_color_test dev n in
     Printf.printf
       "  %s (%s): %.2f ms, %s\n%!"
       dev.Device.name
       dev.Device.framework
       time
       (if ok then "OK" else "FAIL") ;
     if not ok then all_ok := false
   with e ->
     Printf.printf "  FAIL (%s)\n%!" (Printexc.to_string e) ;
     all_ok := false) ;
  !all_ok

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

  print_endline "=== Complex Types runtime Tests ===" ;
  Printf.printf "Size: %d elements\n\n" cfg.size ;

  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Vulkan"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices devs ;
  Printf.printf "\nFound %d runtime device(s)\n" (Array.length devs) ;

  (* Filter out Interpreter in benchmark mode - too slow *)
  let devs =
    if cfg.benchmark_all then
      Array.of_list
        (List.filter
           (fun d -> d.Device.framework <> "Interpreter")
           (Array.to_list devs))
    else devs
  in

  if cfg.benchmark_all then begin
    (* Benchmark mode: run on all devices *)
    let all_ok = ref true in
    Array.iter
      (fun dev ->
        Printf.printf "\n=== Device: %s (%s) ===\n%!" dev.Device.name dev.Device.framework ;
        if not (run_all_tests dev cfg.size) then all_ok := false)
      devs ;
    if !all_ok then print_endline "\n=== All complex types tests PASSED ==="
    else begin
      print_endline "\n=== Some complex types tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    (* Single device mode - respect -d if specified, else prefer Interpreter/Native *)
    let dev =
      if cfg.dev_id >= 0 then
        Test_helpers.get_device cfg devs
      else
        match Array.find_opt (fun d -> d.Device.framework = "Interpreter") devs with
        | Some d -> d
        | None -> (
            match Array.find_opt (fun d -> d.Device.framework = "Native") devs with
            | Some d -> d
            | None -> Test_helpers.get_device cfg devs)
    in
    Printf.printf "Using device: %s\n%!" dev.Device.name ;
    if run_all_tests dev cfg.size then
      print_endline "\n=== All complex types tests PASSED ==="
    else begin
      print_endline "\n=== Some complex types tests FAILED ===" ;
      exit 1
    end
  end
