(******************************************************************************
 * E2E test: ktype record + helper function executed on device.
 ******************************************************************************)

open Spoc

let () =
  let length_kernel =
    [%kernel
      let module Types = struct
        type point = {x : float32; y : float32}
      end in
      let make_point (x : float32) (y : float32) : point = {x; y} in
      fun (xv : float32 vector) (yv : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + block_idx_x * block_dim_x in
        if tid < n then
          let p = make_point xv.(tid) yv.(tid) in
          dst.(tid) <- sqrt (p.x *. p.x +. p.y *. p.y)]
  in

  let _, kirc_kernel = length_kernel in
  print_endline "=== ktype helper IR ===" ;
  Sarek.Kirc.print_ast kirc_kernel.Sarek.Kirc.body ;
  print_endline "=======================" ;

  let devs = Devices.init () in
  if Array.length devs = 0 then (
    print_endline "No GPU devices found - skipping execution" ;
    exit 0) ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;

  let n = 256 in
  let xv = Vector.create Vector.float32 n in
  let yv = Vector.create Vector.float32 n in
  let dst = Vector.create Vector.float32 n in
  let bax = Vector.to_bigarray_shr xv in
  let bay = Vector.to_bigarray_shr yv in
  let bad = Vector.to_bigarray_shr dst in
  for i = 0 to n - 1 do
    Bigarray.Array1.set bax i (float_of_int i) ;
    Bigarray.Array1.set bay i (float_of_int (n - i))
  done ;

  let threads = 128 in
  let grid_x = (n + threads - 1) / threads in
  let block =
    { Kernel.blockX = threads; blockY = 1; blockZ = 1 }
  in
  let grid =
    { Kernel.gridX = grid_x; gridY = 1; gridZ = 1 }
  in

  let length_kernel = Sarek.Kirc.gen length_kernel dev in
  Sarek.Kirc.run length_kernel (xv, yv, dst, n) (block, grid) 0 dev ;

  Mem.to_cpu dst () ;
  Devices.flush dev () ;

  let ok = ref true in
  for i = 0 to n - 1 do
    let x = Bigarray.Array1.get bax i in
    let y = Bigarray.Array1.get bay i in
    let expected = sqrt (x *. x +. y *. y) in
    let got = Bigarray.Array1.get bad i in
    if abs_float (got -. expected) > 1e-3 then (
      ok := false ;
      Printf.printf "Mismatch at %d: got %f expected %f\n%!" i got expected)
  done ;
  if !ok then print_endline "Execution check PASSED"
  else (print_endline "Execution check FAILED" ; exit 1)
