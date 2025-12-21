(******************************************************************************
 * E2E test: use a globally registered Sarek constant across modules.
 ******************************************************************************)

open Spoc

type float32 = float

let () =
  let kernel =
    [%kernel
      fun (xs : float32 vector)
          (ys : float32 vector)
          (dst : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let open Registered_defs in
          dst.(tid) <- scale_fun (xs.(tid) +. ys.(tid))]
  in

  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No GPU devices found - skipping execution" ;
    exit 0) ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

  let n = 64 in
  let bax = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let bay = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let bad = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  for i = 0 to n - 1 do
    Bigarray.Array1.set bax i (float_of_int i) ;
    Bigarray.Array1.set bay i (float_of_int (n - i))
  done ;

  let xv = Vector.of_bigarray_shr Vector.float32 bax in
  let yv = Vector.of_bigarray_shr Vector.float32 bay in
  let dst = Vector.of_bigarray_shr Vector.float32 bad in

  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let block = {Kernel.blockX = threads; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = grid_x; gridY = 1; gridZ = 1} in

  let kernel = Sarek.Kirc.gen kernel dev in
  Sarek.Kirc.run kernel (xv, yv, dst, n) (block, grid) 0 dev ;
  Mem.to_cpu dst () ;
  Devices.flush dev () ;

  let ok = ref true in
  for i = 0 to n - 1 do
    let expected =
      Registered_defs.scale_fun
        (Bigarray.Array1.get bax i +. Bigarray.Array1.get bay i)
    in
    let got = Bigarray.Array1.get bad i in
    if abs_float (got -. expected) > 1e-4 then (
      ok := false ;
      Printf.printf "Mismatch at %d: got %f expected %f\n%!" i got expected)
  done ;
  if !ok then print_endline "Registered const execution PASSED"
  else (
    print_endline "Registered const execution FAILED" ;
    exit 1)
