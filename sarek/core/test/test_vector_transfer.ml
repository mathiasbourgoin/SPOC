(******************************************************************************
 * Unit tests for Vector_transfer module
 *
 * Covers sync callback wiring and host pointer helpers.
 ******************************************************************************)

open Spoc_core

let test_sync_callback () =
  let called = ref false in
  let cb =
    {
      Vector_transfer.sync =
        (fun _ ->
          called := true ;
          true);
    }
  in
  Vector.register_sync_callback cb ;
  let v = Vector.create_float32 1 in
  Vector.ensure_cpu_sync v ;
  assert !called ;
  print_endline "  sync callback: OK"

let test_host_ptr_helpers () =
  let v = Vector.create_float32 1 in
  let ba = Vector.to_bigarray v in
  Bigarray.Array1.set ba 0 3.14 ;
  let ptr = Vector_transfer.host_ptr v in
  assert (ptr <> 0n) ;
  let void_ptr = Vector_transfer.to_ctypes_ptr v in
  assert (Ctypes.is_null void_ptr |> not) ;
  print_endline "  host_ptr helpers: OK"

let () =
  print_endline "Vector_transfer tests:" ;
  test_sync_callback () ;
  test_host_ptr_helpers () ;
  print_endline "All Vector_transfer tests passed!"
