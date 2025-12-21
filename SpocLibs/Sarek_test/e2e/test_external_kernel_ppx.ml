(* Smoke test for the external kernel PPX inside sarek_ppx. *)

open Spoc

(* Declare an external kernel implemented elsewhere. We don't execute it here;
   we just ensure the PPX rewrite compiles and produces a value. *)
[%%kernel
external ext_dummy
  : (float, Bigarray.float32_elt) Vector.vector ->
    (float, Bigarray.float32_elt) Vector.vector -> unit =
  "dummy.cl" "dummy_fun"]

let () =
  (* Show that the external value exists and has a params method. *)
  let (_ : < file : string; kern : string; params : _ list >) = ext_dummy in
  print_endline "External kernel declaration compiled"
