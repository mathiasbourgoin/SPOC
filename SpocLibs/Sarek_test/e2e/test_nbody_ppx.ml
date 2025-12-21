(******************************************************************************
 * Simplified N-Body example using Sarek PPX with ktype + helper.
 * Uses let x = mut ... syntax for mutable accumulators.
 ******************************************************************************)

open Spoc

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
  Sarek.Kirc.print_ast kirc_kernel.Sarek.Kirc.body ;
  print_endline "======================" ;

  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No GPU devices found - skipping execution" ;
    exit 0) ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

  let n = 64 in
  let bax = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let bay = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let baz = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  for i = 0 to n - 1 do
    Bigarray.Array1.set bax i (float_of_int (i mod 7) /. 10.0) ;
    Bigarray.Array1.set bay i (float_of_int (i mod 5) /. 10.0) ;
    Bigarray.Array1.set baz i (float_of_int (i mod 3) /. 10.0)
  done ;
  let ax_ba =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let ay_ba =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let az_ba =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let xs = Vector.of_bigarray_shr Vector.float32 bax in
  let ys = Vector.of_bigarray_shr Vector.float32 bay in
  let zs = Vector.of_bigarray_shr Vector.float32 baz in
  let ax = Vector.of_bigarray_shr Vector.float32 ax_ba in
  let ay = Vector.of_bigarray_shr Vector.float32 ay_ba in
  let az = Vector.of_bigarray_shr Vector.float32 az_ba in

  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let block = {Kernel.blockX = threads; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = grid_x; gridY = 1; gridZ = 1} in

  let accel_kernel = Sarek.Kirc.gen ~keep_temp:true accel_kernel dev in
  Sarek.Kirc.run accel_kernel (xs, ys, zs, ax, ay, az, n) (block, grid) 0 dev ;
  Mem.to_cpu ax () ;
  Mem.to_cpu ay () ;
  Mem.to_cpu az () ;
  Devices.flush dev () ;

  (* CPU reference *)
  let ok = ref true in
  for i = 0 to n - 1 do
    let px = Bigarray.Array1.get bax i
    and py = Bigarray.Array1.get bay i
    and pz = Bigarray.Array1.get baz i in
    let rfx = ref 0.0 and rfy = ref 0.0 and rfz = ref 0.0 in
    for j = 0 to n - 1 do
      if j <> i then (
        let qx = Bigarray.Array1.get bax j
        and qy = Bigarray.Array1.get bay j
        and qz = Bigarray.Array1.get baz j in
        let dx = qx -. px in
        let dy = qy -. py in
        let dz = qz -. pz in
        let dist2 = (dx *. dx) +. (dy *. dy) +. (dz *. dz) +. 1e-9 in
        let inv = 1.0 /. (sqrt dist2 *. dist2) in
        rfx := !rfx +. (dx *. inv) ;
        rfy := !rfy +. (dy *. inv) ;
        rfz := !rfz +. (dz *. inv))
    done ;
    let gx = Bigarray.Array1.get ax_ba i
    and gy = Bigarray.Array1.get ay_ba i
    and gz = Bigarray.Array1.get az_ba i in
    if
      abs_float (gx -. !rfx) > 1e-3
      || abs_float (gy -. !rfy) > 1e-3
      || abs_float (gz -. !rfz) > 1e-3
    then (
      ok := false ;
      Printf.printf
        "Mismatch at %d: GPU (%f,%f,%f) CPU (%f,%f,%f)\n%!"
        i
        gx
        gy
        gz
        !rfx
        !rfy
        !rfz)
  done ;
  if !ok then print_endline "NBody PPX execution PASSED"
  else (
    print_endline "NBody PPX execution FAILED" ;
    exit 1)
