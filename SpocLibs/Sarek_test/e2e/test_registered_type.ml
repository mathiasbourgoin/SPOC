(******************************************************************************
 * E2E test: register a Sarek record type outside kernels via [%sarek.type].
 ******************************************************************************)

open Spoc

type float32 = float

type point = {x : float32; y : float32} [@@sarek.type]

let () =
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
  in
  let _, kirc_kernel = kernel in
  print_endline "=== Registered type IR ===" ;
  Sarek.Kirc.print_ast kirc_kernel.Sarek.Kirc.body ;
  print_endline "==========================" ;

  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No GPU devices found - skipping execution" ;
    exit 0) ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

  let n = 128 in
  let bax = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let bay = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let bad = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let xv = Vector.of_bigarray_shr Vector.float32 bax in
  let yv = Vector.of_bigarray_shr Vector.float32 bay in
  let dst = Vector.of_bigarray_shr Vector.float32 bad in
  for i = 0 to n - 1 do
    Bigarray.Array1.set bax i (float_of_int i) ;
    Bigarray.Array1.set bay i (float_of_int (n - i))
  done ;

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
    let x = Bigarray.Array1.get bax i in
    let y = Bigarray.Array1.get bay i in
    let expected = sqrt ((x *. x) +. (y *. y)) in
    let got = Bigarray.Array1.get bad i in
    if abs_float (got -. expected) > 1e-3 then (
      ok := false ;
      Printf.printf "Mismatch at %d: got %f expected %f\n%!" i got expected)
  done ;
  if !ok then print_endline "Registered type execution PASSED"
  else (
    print_endline "Registered type execution FAILED" ;
    exit 1)
