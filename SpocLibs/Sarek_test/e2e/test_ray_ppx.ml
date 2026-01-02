(******************************************************************************
 * E2E test: simple ray tracing with PPX Sarek.
 * Renders a small sphere scene to a float32 RGB buffer and checks against CPU.
 * Compares SPOC and V2 runtime paths.
 ******************************************************************************)

open Spoc

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
  Sarek.Kirc.print_ast kirc_kernel.Sarek.Kirc.body ;
  print_endline "==================" ;

  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No GPU devices found - skipping execution" ;
    exit 0) ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

  let w = 64 and h = 64 in
  let n = w * h in
  let bax = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let bay = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let baz = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let out_ba =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (n * 3)
  in
  let dirx = Vector.of_bigarray_shr Vector.float32 bax in
  let diry = Vector.of_bigarray_shr Vector.float32 bay in
  let dirz = Vector.of_bigarray_shr Vector.float32 baz in
  let out = Vector.of_bigarray_shr Vector.float32 out_ba in
  for y = 0 to h - 1 do
    for x = 0 to w - 1 do
      let i = (y * w) + x in
      let u = (2.0 *. float_of_int x /. float_of_int (w - 1)) -. 1.0 in
      let v = (2.0 *. float_of_int y /. float_of_int (h - 1)) -. 1.0 in
      Bigarray.Array1.set bax i u ;
      Bigarray.Array1.set bay i v ;
      Bigarray.Array1.set baz i (-1.5)
    done
  done ;

  let threads = 128 in
  let grid_x = (n + threads - 1) / threads in
  let block = {Kernel.blockX = threads; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = grid_x; gridY = 1; gridZ = 1} in

  let ray_kernel = Sarek.Kirc.gen ~keep_temp:true ray_kernel dev in
  Sarek.Kirc.run ray_kernel (dirx, diry, dirz, out, n) (block, grid) 0 dev ;
  Mem.to_cpu out () ;
  Devices.flush dev () ;

  (* CPU reference and ppm output *)
  let ok = ref true in
  let ppm = open_out "/tmp/ray_ppx.ppm" in
  Printf.fprintf ppm "P3\n%d %d\n255\n" w h ;
  for y = 0 to h - 1 do
    for x = 0 to w - 1 do
      let i = (y * w) + x in
      let idx = i * 3 in
      let dx = Bigarray.Array1.get bax i in
      let dy = Bigarray.Array1.get bay i in
      let dz = Bigarray.Array1.get baz i in
      let dir_len = sqrt ((dx *. dx) +. (dy *. dy) +. (dz *. dz)) in
      let dxn = dx /. dir_len and dyn = dy /. dir_len and dzn = dz /. dir_len in
      let a = (dxn *. dxn) +. (dyn *. dyn) +. (dzn *. dzn) in
      let half_b = (0.0 *. dxn) +. (0.0 *. dyn) +. (2.0 *. dzn) in
      let c = (2.0 *. 2.0) -. (0.5 *. 0.5) in
      let disc = (half_b *. half_b) -. (a *. c) in
      let r, g, b =
        if disc > 0.0 then
          let t = (-.half_b -. sqrt disc) /. a in
          let hx = t *. dxn and hy = t *. dyn and hz = t *. dzn in
          let nx = hx -. 0.0 and ny = hy -. 0.0 and nz = hz +. 2.0 in
          let inv = 1.0 /. sqrt ((nx *. nx) +. (ny *. ny) +. (nz *. nz)) in
          let nx = nx *. inv and ny = ny *. inv and nz = nz *. inv in
          (0.5 *. (nx +. 1.0), 0.5 *. (ny +. 1.0), 0.5 *. (nz +. 1.0))
        else
          let t = 0.5 *. (dyn +. 1.0) in
          (1.0 -. t +. (t *. 0.5), 1.0 -. t +. (t *. 0.7), 1.0)
      in
      let gx = Bigarray.Array1.get out_ba (idx + 0) in
      let gy = Bigarray.Array1.get out_ba (idx + 1) in
      let gz = Bigarray.Array1.get out_ba (idx + 2) in
      if
        abs_float (gx -. r) > 1e-3
        || abs_float (gy -. g) > 1e-3
        || abs_float (gz -. b) > 1e-3
      then (
        ok := false ;
        Printf.printf
          "Mismatch at %d,%d GPU (%f,%f,%f) CPU (%f,%f,%f)\n%!"
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
  if !ok then Printf.printf "Ray PPX (SPOC) PASSED (ppm: /tmp/ray_ppx.ppm)\n%!"
  else (
    print_endline "Ray PPX (SPOC) FAILED" ;
    exit 1) ;

  (* ========== V2 Path ========== *)
  print_endline "\n=== Running V2 path ===" ;
  let _, kirc = ray_kernel in
  let v2_devs =
    V2_Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  let v2_dev = if Array.length v2_devs > 0 then Some v2_devs.(0) else None in
  match v2_dev with
  | None ->
      print_endline "V2: No device found - SKIPPED" ;
      print_endline "Ray PPX tests PASSED"
  | Some dev -> (
      match kirc.Sarek.Kirc.body_v2 with
      | None ->
          print_endline "V2: No V2 IR available - SKIPPED" ;
          print_endline "Ray PPX tests PASSED"
      | Some ir ->
          let v2_dirx = V2_Vector.create V2_Vector.float32 n in
          let v2_diry = V2_Vector.create V2_Vector.float32 n in
          let v2_dirz = V2_Vector.create V2_Vector.float32 n in
          let v2_out = V2_Vector.create V2_Vector.float32 (n * 3) in
          for i = 0 to n - 1 do
            V2_Vector.set v2_dirx i (Bigarray.Array1.get bax i) ;
            V2_Vector.set v2_diry i (Bigarray.Array1.get bay i) ;
            V2_Vector.set v2_dirz i (Bigarray.Array1.get baz i)
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

          (* Verify V2 results *)
          let v2_ok = ref true in
          for y = 0 to h - 1 do
            for x = 0 to w - 1 do
              let i = (y * w) + x in
              let idx = i * 3 in
              let dx = Bigarray.Array1.get bax i in
              let dy = Bigarray.Array1.get bay i in
              let dz = Bigarray.Array1.get baz i in
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
                  b)
            done
          done ;
          if !v2_ok then print_endline "Ray PPX (V2) PASSED"
          else (
            print_endline "Ray PPX (V2) FAILED" ;
            exit 1) ;
          print_endline "Ray PPX tests PASSED")
