(******************************************************************************
 * E2E test for native CPU runtime with custom type vectors.
 * This tests that custom record types work as kernel input/output vectors.
 ******************************************************************************)

open Spoc

type float32 = float

(* Define a custom point type - PPX generates point_custom for Vector.Custom *)
type point = {x : float32; y : float32} [@@sarek.type]

(* Kernel that takes point vectors as input/output.
   Wrapped in function to avoid value restriction. *)
let point_transform () =
  [%kernel
    fun (src : point vector) (dst : point vector) (n : int32) ->
      let tid = (block_idx_x * block_dim_x) + thread_idx_x in
      if tid < n then
        let p = src.(tid) in
        dst.(tid) <- {x = p.x +. 1.0; y = p.y *. 2.0}]

let () =
  let devices = Devices.init () in

  (* Find native device *)
  match Devices.find_native devices with
  | None ->
      print_endline "No native device found - skipping test" ;
      exit 0
  | Some dev ->
      let n = 1000 in

      (* Create custom type vectors using the generated point_custom *)
      let src = Vector.create (Vector.Custom point_custom) n in
      let dst = Vector.create (Vector.Custom point_custom) n in

      (* Initialize source with point values *)
      for i = 0 to n - 1 do
        Mem.set src i {x = float_of_int i; y = float_of_int (i * 2)}
      done ;

      (* Run kernel *)
      let threadsPerBlock = min 256 n in
      let blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock in
      let block = {Kernel.blockX = threadsPerBlock; blockY = 1; blockZ = 1} in
      let grid = {Kernel.gridX = blocksPerGrid; gridY = 1; gridZ = 1} in
      Sarek.Kirc.run (point_transform ()) (src, dst, n) (block, grid) 0 dev ;

      (* Verify results *)
      let errors = ref 0 in
      for i = 0 to n - 1 do
        let expected_x = float_of_int i +. 1.0 in
        let expected_y = float_of_int (i * 2) *. 2.0 in
        let result = Mem.get dst i in
        if
          abs_float (result.x -. expected_x) > 1e-3
          || abs_float (result.y -. expected_y) > 1e-3
        then begin
          if !errors < 5 then
            Printf.printf
              "Error at %d: expected {x=%.1f; y=%.1f}, got {x=%.1f; y=%.1f}\n"
              i
              expected_x
              expected_y
              result.x
              result.y ;
          incr errors
        end
      done ;

      if !errors = 0 then
        print_endline "Native runtime with custom type vectors: PASS"
      else begin
        Printf.printf
          "Native runtime with custom type vectors: FAIL (%d errors)\n"
          !errors ;
        exit 1
      end
