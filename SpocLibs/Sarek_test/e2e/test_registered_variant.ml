(******************************************************************************
 * E2E test: register a Sarek variant type outside kernels via [@@sarek.type].
 ******************************************************************************)

open Spoc

type float32 = float

type color = Red | Value of float32 [@@sarek.type]

let () =
  let kernel =
    [%kernel
      fun (xs : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let c =
            if xs.(tid) > 0.0 then Value xs.(tid) else Red
          in
          match c with
          | Red -> dst.(tid) <- 0.0
          | Value v -> dst.(tid) <- v +. 1.0]
  in
  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No GPU devices found - skipping execution" ;
    exit 0) ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

  let n = 128 in
  let dst_ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let xs_ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  for i = 0 to n - 1 do
    Bigarray.Array1.set xs_ba i (if i mod 2 = 0 then -1.0 else float_of_int i)
  done ;

  let xs = Vector.of_bigarray_shr Vector.float32 xs_ba in
  let dst = Vector.of_bigarray_shr Vector.float32 dst_ba in

  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let block = {Kernel.blockX = threads; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = grid_x; gridY = 1; gridZ = 1} in
  let kernel = Sarek.Kirc.gen kernel dev in
  Sarek.Kirc.run kernel (xs, dst, n) (block, grid) 0 dev ;
  Mem.to_cpu dst () ;
  Devices.flush dev () ;

  let ok = ref true in
  for i = 0 to n - 1 do
    let expected = if i mod 2 = 0 then 0.0 else (float_of_int i) +. 1.0 in
    let got = Bigarray.Array1.get dst_ba i in
    if abs_float (got -. expected) > 1e-3 then (
      ok := false ;
      Printf.printf "Mismatch at %d: got %f expected %f\n%!" i got expected)
  done ;
  if !ok then print_endline "Registered variant execution PASSED"
  else (
    print_endline "Registered variant execution FAILED" ;
    exit 1)
