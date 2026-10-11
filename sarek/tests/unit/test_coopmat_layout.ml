(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The interpreter, the oracle of every backend, on the [layout] of [CM_load]
    and [CM_store]: element (r, c) is at [index + r * stride + c] row-major and
    at [index + c * stride + r] column-major. Checked against a host reference
    for every combination of layouts of A, B, C and the stored D, with an index
    and a stride that are not the fragment size, so a lowering that confused the
    two, ignored the index or used the fragment width as the stride would be
    caught. Integer configuration: exact, so the comparison is equality. *)

open Sarek_ir_types
open Alcotest
module Interp = Sarek.Sarek_ir_interp
module C = Sarek_coopmat

let mnk = 16

let base = 3

let stride = 20

(* Large enough for a 16 x 16 fragment at [base] with [stride] in either
   layout, and a few elements past it that nothing may write. *)
let len = base + (15 * stride) + 16 + 5

let cfg =
  {
    C.cfg_shape = {C.m = mnk; n = mnk; k = mnk};
    cfg_a = C.Uint8;
    cfg_b = C.Uint8;
    cfg_c = C.Sint32;
    cfg_result = C.Sint32;
    cfg_saturating = false;
    cfg_scope = C.Subgroup;
  }

let frag_a, frag_b, frag_c, frag_d = C.fragments_of_config cfg

let i32 n = EConst (CInt32 (Int32.of_int n))

let var id name ty =
  {var_name = name; var_id = id; var_type = ty; var_mutable = false}

let vec_param v elt = DParam (v, Some {arr_elttype = elt; arr_memspace = Global})

let layout_name = C.layout_name

(** [d = a * b + c] with each matrix in its own layout. *)
let tile_ir ~la ~lb ~lc ~ld =
  let cm op = SCoopmat op in
  let load dst frag src layout =
    cm (CM_load {dst; frag; src; index = i32 base; stride = i32 stride; layout})
  in
  {
    default_kernel with
    kern_name = "coopmat_layout_tile";
    kern_params =
      [
        vec_param (var 0 "a" (TVec TUint8)) TUint8;
        vec_param (var 1 "b" (TVec TUint8)) TUint8;
        vec_param (var 2 "c" (TVec TInt32)) TInt32;
        vec_param (var 3 "d" (TVec TInt32)) TInt32;
      ];
    kern_body =
      SSeq
        [
          cm (CM_decl {name = "fa"; frag = frag_a});
          cm (CM_decl {name = "fb"; frag = frag_b});
          cm (CM_decl {name = "fc"; frag = frag_c});
          load "fa" frag_a "a" la;
          load "fb" frag_b "b" lb;
          load "fc" frag_c "c" lc;
          cm (CM_muladd {dst = "fc"; a = "fa"; b = "fb"; c = "fc"; cfg});
          cm
            (CM_store
               {
                 src = "fc";
                 frag = frag_d;
                 dst = "d";
                 index = i32 base;
                 stride = i32 stride;
                 layout = ld;
               });
        ];
  }

(* Buffer position of logical element (r, c). *)
let at layout r c =
  match layout with
  | C.Row_major -> base + (r * stride) + c
  | C.Col_major -> base + (c * stride) + r

let buf_a = Array.init len (fun i -> ((i * 37) + 11) land 0xff)

let buf_b = Array.init len (fun i -> ((i * 101) + 7) land 0xff)

let buf_c = Array.init len (fun i -> (i * 7919 mod 2003) - 1000)

let sentinel = -123456

let host ~la ~lb ~lc ~ld =
  let d = Array.make len sentinel in
  for r = 0 to mnk - 1 do
    for c = 0 to mnk - 1 do
      let acc = ref buf_c.(at lc r c) in
      for k = 0 to mnk - 1 do
        acc := !acc + (buf_a.(at la r k) * buf_b.(at lb k c))
      done ;
      d.(at ld r c) <- !acc
    done
  done ;
  d

let interp ~la ~lb ~lc ~ld =
  let ints arr = Array.map (fun x -> Interp.VInt32 (Int32.of_int x)) arr in
  let d = ints (Array.make len sentinel) in
  Interp.run_kernel
    (tile_ir ~la ~lb ~lc ~ld)
    ~block:(32, 1, 1)
    ~grid:(1, 1, 1)
    [
      ("a", Interp.ArgArray (ints buf_a));
      ("b", Interp.ArgArray (ints buf_b));
      ("c", Interp.ArgArray (ints buf_c));
      ("d", Interp.ArgArray d);
    ] ;
  Array.map
    (function Interp.VInt32 n -> Int32.to_int n | _ -> Alcotest.fail "d")
    d

let layouts = [C.Row_major; C.Col_major]

let combos =
  List.concat_map
    (fun la ->
      List.concat_map
        (fun lb ->
          List.concat_map
            (fun lc -> List.map (fun ld -> (la, lb, lc, ld)) layouts)
            layouts)
        layouts)
    layouts

let test_combo (la, lb, lc, ld) () =
  check
    (array int)
    (Printf.sprintf
       "A %s, B %s, C %s, D %s"
       (layout_name la)
       (layout_name lb)
       (layout_name lc)
       (layout_name ld))
    (host ~la ~lb ~lc ~ld)
    (interp ~la ~lb ~lc ~ld)

(* The comparison can fail: the column-major B result is not the row-major
   one on the same buffer. *)
let test_layout_matters () =
  let r = C.Row_major and c = C.Col_major in
  let row = interp ~la:r ~lb:r ~lc:r ~ld:r in
  let col = interp ~la:r ~lb:c ~lc:r ~ld:r in
  check bool "column-major B differs from row-major B" true (row <> col)

(** {1 Sarek_gemm.Coopmat with B stored transposed} *)

let gemm_d ~b_layout ~m ~n ~k ~a ~b =
  let f x = Interp.VFloat32 x in
  let d = Array.make (m * n) (f 0.0) in
  Interp.run_kernel
    (Sarek_gemm.Coopmat.ir ~b_layout ())
    ~block:(64, 1, 1)
    ~grid:(m / 32 * (n / 32) / 2, 1, 1)
    [
      ("a", Interp.ArgArray (Array.map f a));
      ("b", Interp.ArgArray (Array.map f b));
      ("c", Interp.ArgArray (Array.make (m * n) (f 1.0)));
      ("d", Interp.ArgArray d);
      ("m", Interp.ArgScalar (Interp.VInt32 (Int32.of_int m)));
      ("n", Interp.ArgScalar (Interp.VInt32 (Int32.of_int n)));
      ("k", Interp.ArgScalar (Interp.VInt32 (Int32.of_int k)));
    ] ;
  Array.map (function Interp.VFloat32 x -> x | _ -> Alcotest.fail "d") d

(* Small integers: every product and partial sum is exact in binary32, so the
   two layouts and the host reference agree exactly. *)
let test_gemm_b_transposed () =
  C.accept_relaxed_accumulation ~reason:"layout test: exact small integers" ;
  let m = 64 and n = 32 and k = 48 in
  let a = Array.init (m * k) (fun i -> float_of_int ((i * 7 mod 9) - 4)) in
  let b = Array.init (k * n) (fun i -> float_of_int ((i * 5 mod 7) - 3)) in
  let bt = Array.init (n * k) (fun i -> b.((i mod k * n) + (i / k))) in
  let host =
    Array.init (m * n) (fun o ->
        let r = o / n and c = o mod n in
        let acc = ref 1.0 in
        for kk = 0 to k - 1 do
          acc := !acc +. (a.((r * k) + kk) *. b.((kk * n) + c))
        done ;
        !acc)
  in
  let row = gemm_d ~b_layout:C.Row_major ~m ~n ~k ~a ~b in
  let col = gemm_d ~b_layout:C.Col_major ~m ~n ~k ~a ~b:bt in
  check (array (float 0.0)) "row-major B = host" host row ;
  check (array (float 0.0)) "column-major B^T = host" host col

let () =
  run
    "coopmat_layout"
    [
      ( "interpreter",
        List.map
          (fun ((la, lb, lc, ld) as combo) ->
            test_case
              (String.concat
                 "/"
                 (List.map
                    (function C.Row_major -> "R" | C.Col_major -> "C")
                    [la; lb; lc; ld]))
              `Quick
              (test_combo combo))
          combos
        @ [test_case "layout matters" `Quick test_layout_matters] );
      ("gemm", [test_case "B transposed" `Quick test_gemm_b_transposed]);
    ]
