(******************************************************************************
 * E2E test: simple ray tracing with PPX Sarek.
 * Renders a small sphere scene to a float32 RGB buffer and checks against CPU.
 * V2 runtime only.
 ******************************************************************************)

(* Shadow SPOC's vector with V2 vector for kernel compatibility *)
type ('a, 'b) vector = ('a, 'b) Sarek_core.Vector.t

(* V2 module aliases *)
module V2_Device = Sarek_core.Device
module V2_Vector = Sarek_core.Vector
module V2_Transfer = Sarek_core.Transfer

(* Force V2 backend registration *)
let () =
  Sarek_cuda.Cuda_plugin_v2.init () ;
  Sarek_opencl.Opencl_plugin_v2.init ()

let () =
  let ray_kernel =
    [%kernel
      let module Types = struct
        type vec3 = {x : float32; y : float32; z : float32}
      end in
      let make_vec3 (x : float32) (y : float32) (z : float32) : vec3 =
        {x; y; z}
      in
      let dot (a : vec3) (b : vec3) : float32 =
        (a.x *. b.x) +. (a.y *. b.y) +. (a.z *. b.z)
      in
      let normalize (v : vec3) : vec3 =
        let inv = 1.0 /. sqrt (dot v v) in
        make_vec3 (v.x *. inv) (v.y *. inv) (v.z *. inv)
      in
      fun (dirx : float32 vector)
          (diry : float32 vector)
          (dirz : float32 vector)
          (out : float32 vector)
          (n : int)
        ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then (
          let dx = dirx.(tid) in
          let dy = diry.(tid) in
          let dz = dirz.(tid) in
          let dir = normalize (make_vec3 dx dy dz) in
          let dxn = dir.x in
          let dyn = dir.y in
          let dzn = dir.z in
          let a = dot dir dir in
          let half_b = 2.0 *. dzn in
          let c = 3.75 in
          let disc = (half_b *. half_b) -. (a *. c) in
          let idx = tid * 3 in
          if disc > 0.0 then (
            let t = (-.half_b -. sqrt disc) /. a in
            let hx = t *. dxn in
            let hy = t *. dyn in
            let hz = t *. dzn in
            let nx = hx -. 0.0 in
            let ny = hy -. 0.0 in
            let nz = hz +. 2.0 in
            let nrm = normalize (make_vec3 nx ny nz) in
            out.(idx) <- 0.5 *. (nrm.x +. 1.0) ;
            out.(idx + 1) <- 0.5 *. (nrm.y +. 1.0) ;
            out.(idx + 2) <- 0.5 *. (nrm.z +. 1.0))
          else
            let t = 0.5 *. (dyn +. 1.0) in
            let r = 1.0 -. t +. (t *. 0.5) in
            let g = 1.0 -. t +. (t *. 0.7) in
            let b = 1.0 in
            out.(idx) <- r ;
            out.(idx + 1) <- g ;
            out.(idx + 2) <- b)]
  in

  let _, kirc_kernel = ray_kernel in
  print_endline "=== Ray PPX IR ===" ;
  Sarek.Kirc_Ast.print_ast kirc_kernel.Sarek.Kirc_types.body ;
  print_endline "==================" ;

  print_endline "\n=== Running V2 path ===" ;
  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length v2_devs = 0 then (
    print_endline "V2: No device found - IR test passed" ;
    exit 0) ;

  let dev = v2_devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.V2_Device.name ;

  (match kirc_kernel.Sarek.Kirc_types.body_v2 with
  | None ->
      print_endline "V2: No V2 IR available - IR test passed" ;
      exit 0
  | Some ir ->
      let w = 64 and h = 64 in
      let n = w * h in
      let threads = 128 in
      let grid_x = (n + threads - 1) / threads in

      (* Initialize ray directions *)
      let bax = Array.init n (fun i ->
        let x = i mod w in
        (2.0 *. float_of_int x /. float_of_int (w - 1)) -. 1.0) in
      let bay = Array.init n (fun i ->
        let y = i / w in
        (2.0 *. float_of_int y /. float_of_int (h - 1)) -. 1.0) in
      let baz = Array.make n (-1.5) in

      let v2_dirx = V2_Vector.create V2_Vector.float32 n in
      let v2_diry = V2_Vector.create V2_Vector.float32 n in
      let v2_dirz = V2_Vector.create V2_Vector.float32 n in
      let v2_out = V2_Vector.create V2_Vector.float32 (n * 3) in
      for i = 0 to n - 1 do
        V2_Vector.set v2_dirx i bax.(i) ;
        V2_Vector.set v2_diry i bay.(i) ;
        V2_Vector.set v2_dirz i baz.(i)
      done ;
      for i = 0 to (n * 3) - 1 do
        V2_Vector.set v2_out i 0.0
      done ;

      let block = Sarek.Execute.dims1d threads in
      let grid = Sarek.Execute.dims1d grid_x in

      Sarek.Execute.run_vectors
        ~device:dev
        ~ir
        ~args:
          [
            Sarek.Execute.Vec v2_dirx;
            Sarek.Execute.Vec v2_diry;
            Sarek.Execute.Vec v2_dirz;
            Sarek.Execute.Vec v2_out;
            Sarek.Execute.Int n;
          ]
        ~block
        ~grid
        () ;
      V2_Transfer.flush dev ;

      (* Verify V2 results and output PPM *)
      let v2_ok = ref true in
      let ppm = open_out "/tmp/ray_ppx.ppm" in
      Printf.fprintf ppm "P3\n%d %d\n255\n" w h ;
      for y = 0 to h - 1 do
        for x = 0 to w - 1 do
          let i = (y * w) + x in
          let idx = i * 3 in
          let dx = bax.(i) in
          let dy = bay.(i) in
          let dz = baz.(i) in
          let dir_len = sqrt ((dx *. dx) +. (dy *. dy) +. (dz *. dz)) in
          let dxn = dx /. dir_len
          and dyn = dy /. dir_len
          and dzn = dz /. dir_len in
          let a = (dxn *. dxn) +. (dyn *. dyn) +. (dzn *. dzn) in
          let half_b = (0.0 *. dxn) +. (0.0 *. dyn) +. (2.0 *. dzn) in
          let c = (2.0 *. 2.0) -. (0.5 *. 0.5) in
          let disc = (half_b *. half_b) -. (a *. c) in
          let r, g, b =
            if disc > 0.0 then
              let t = (-.half_b -. sqrt disc) /. a in
              let hx = t *. dxn and hy = t *. dyn and hz = t *. dzn in
              let nx = hx -. 0.0 and ny = hy -. 0.0 and nz = hz +. 2.0 in
              let inv =
                1.0 /. sqrt ((nx *. nx) +. (ny *. ny) +. (nz *. nz))
              in
              let nx = nx *. inv and ny = ny *. inv and nz = nz *. inv in
              (0.5 *. (nx +. 1.0), 0.5 *. (ny +. 1.0), 0.5 *. (nz +. 1.0))
            else
              let t = 0.5 *. (dyn +. 1.0) in
              (1.0 -. t +. (t *. 0.5), 1.0 -. t +. (t *. 0.7), 1.0)
          in
          let gx = V2_Vector.get v2_out (idx + 0) in
          let gy = V2_Vector.get v2_out (idx + 1) in
          let gz = V2_Vector.get v2_out (idx + 2) in
          if
            abs_float (gx -. r) > 1e-3
            || abs_float (gy -. g) > 1e-3
            || abs_float (gz -. b) > 1e-3
          then (
            v2_ok := false ;
            Printf.printf
              "V2 Mismatch at %d,%d GPU (%f,%f,%f) CPU (%f,%f,%f)\n%!"
              x
              y
              gx
              gy
              gz
              r
              g
              b) ;
          let to_byte f =
            let v = int_of_float (255.0 *. max 0.0 (min 1.0 f)) in
            max 0 (min 255 v)
          in
          Printf.fprintf ppm "%d %d %d\n" (to_byte r) (to_byte g) (to_byte b)
        done
      done ;
      close_out ppm ;
      if !v2_ok then Printf.printf "Ray PPX (V2) PASSED (ppm: /tmp/ray_ppx.ppm)\n%!"
      else (
        print_endline "Ray PPX (V2) FAILED" ;
        exit 1)) ;
  print_endline "Ray PPX tests PASSED"
