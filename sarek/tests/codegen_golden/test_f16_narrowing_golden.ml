(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Which f32 -> f16 narrowings keep sarek_f32_barrier (specs/f16-narrowing-
 * barrier.md).
 *
 * The barrier is needed only where the compiler could round once where the
 * DSL rounds twice. It is dropped when the narrowed f32 value is exact (no
 * operation feeding it rounds) and when it is a sum or difference of two
 * binary16 values (double rounding through f32 is innocuous there). Everything
 * the analysis cannot prove keeps it. Each case below states which, and why.
 ******************************************************************************)

open Sarek_ir_types

let var ?(mut = false) name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = mut}

let vec name elt =
  DParam (var name (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})

let scalar name ty = DParam (var name ty, None)

let i32 n = EConst (CInt32 (Int32.of_int n))

let f32c x = EConst (CFloat32 x)

let tid = var "tid" TInt32

let read a = EArrayRead (a, EVar tid)

let widen e = ECast (TFloat32, e)

let narrow e = ECast (TFloat16, e)

(* int -> float as the DSL lowers [float n]. *)
let to_float e = EIntrinsic (["Sarek_stdlib"; "Gpu"], "float", [e])

let ( *: ) a b = EBinop (Mul, a, b)

let ( +: ) a b = EBinop (Add, a, b)

let ( -: ) a b = EBinop (Sub, a, b)

let ( &: ) a b = EBinop (BitAnd, a, b)

let ( >>: ) a b = EBinop (Shr, a, b)

let ( <<: ) a b = EBinop (Shl, a, b)

let store e = SAssign (LArrayElem ("out", EVar tid), narrow e)

(* out: f16, h/g: f16, q: int32, x: f32 (read-only), w: f32 (written). *)
let kernel ?(name = "k") ?(funcs = []) ?(extra = []) body =
  {
    default_kernel with
    kern_name = name;
    kern_params =
      [
        vec "out" TFloat16;
        vec "h" TFloat16;
        vec "g" TFloat16;
        vec "q" TInt32;
        vec "x" TFloat32;
      ]
      @ extra;
    kern_body = SLet (tid, EIntrinsic ([], "global_thread_id", []), body);
    kern_funcs = funcs;
  }

let gen k = Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:[] k

let count needle hay =
  let nl = String.length needle and hl = String.length hay in
  let rec go i acc =
    if i + nl > hl then acc
    else if String.sub hay i nl = needle then go (i + 1) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

let barriered = "__float2half(sarek_f32_barrier("

let plain = "__float2half("

let expect_barriers what k n =
  let src = gen k in
  let calls = count barriered src in
  Alcotest.(check int) (what ^ ": barrier calls") n calls ;
  Alcotest.(check bool) (what ^ ": narrowing emitted") true (count plain src > 0)

let elided what e = expect_barriers what (kernel (store e)) 0

let kept what e = expect_barriers what (kernel (store e)) 1

let h = widen (read "h")

let g = widen (read "g")

let qv = read "q"

(* An int8 as Q8_0 reads it: [(w lsl 24) asr 24] is in [-128, 127]. *)
let int8 = qv <<: i32 24 >>: i32 24

(* ---- elided: the narrowed value is exact in f32 ---- *)

let test_widen_alone () = elided "f16(widen h)" h

let test_f32_param_alone () = elided "f16(x) (read-only f32 buffer)" (read "x")

let test_scalar_param () =
  expect_barriers
    "f16(s) (f32 kernel argument)"
    (kernel ~extra:[scalar "s" TFloat32] (store (EVar (var "s" TFloat32))))
    0

let test_times_two () = elided "f16(h * 2)" (h *: f32c 2.0)

let test_q8 () = elided "f16(h * float int8)" (h *: to_float int8)

let test_q8_cast () =
  elided "f16(h * (float) int8)" (h *: ECast (TFloat32, int8))

let test_q6 () =
  let q6 = (qv >>: i32 8 &: i32 63) -: i32 32 in
  elided "f16(h * float int8 * float q6)" (h *: to_float int8 *: to_float q6)

let test_let_bound () =
  let p = var "p" TFloat32 in
  expect_barriers
    "let p = h * float (q land 15) in f16(p)"
    (kernel (SLet (p, h *: to_float (qv &: i32 15), store (EVar p))))
    0

let test_integer_sum () =
  let byte e = to_float (e &: i32 255) in
  elided
    "f16(float b * float b + float n)"
    ((byte qv *: byte (qv >>: i32 8)) +: to_float (qv &: i32 15))

(* ---- elided: f16 + f16, double rounding through f32 is innocuous ---- *)

let test_f16_sum () = elided "f16(h + g)" (h +: g)

let test_f16_plus_const () = elided "f16(h + 1000)" (h +: f32c 1000.0)

let test_f16_neg () = elided "f16(0 - h)" (f32c 0.0 -: h)

(* ---- kept ---- *)

let test_inexact_const () = kept "f16(h * 1.1)" (h *: f32c 1.1)

let test_let_bound_inexact () =
  let p = var "p" TFloat32 in
  expect_barriers
    "let p = h * 1.1 in f16(p)"
    (kernel (SLet (p, h *: f32c 1.1, store (EVar p))))
    1

let test_q4_difference () =
  let m = to_float (qv >>: i32 4 &: i32 63) in
  kept
    "f16(h * float q4 - g * float m)"
    ((h *: to_float (qv &: i32 15)) -: (g *: m))

let test_f32_scale () = kept "f16(x * float int8)" (read "x" *: to_float int8)

let test_three_term_sum () = kept "f16(h + g + h)" (h +: g +: h)

let test_wide_int () =
  kept "f16(h * float (q land 0xFFFFF))" (h *: to_float (qv &: i32 0xFFFFF))

let test_unknown_int () = kept "f16(float q)" (to_float qv)

let test_division () = kept "f16(h / 3)" (EBinop (Div, h, f32c 3.0))

let test_mutable () =
  let acc = var ~mut:true "acc" TFloat32 in
  expect_barriers
    "acc := acc * h; f16(acc)"
    (kernel
       (SLetMut
          ( acc,
            f32c 1.0,
            SSeq [SAssign (LVar acc, EVar acc *: h); store (EVar acc)] )))
    1

let test_written_buffer () =
  expect_barriers
    "w.(i) <- x * x; f16(w.(i))"
    (kernel
       ~extra:[vec "w" TFloat32]
       (SSeq
          [
            SAssign (LArrayElem ("w", EVar tid), read "x" *: read "x");
            store (read "w");
          ]))
    1

let test_helper_param () =
  let p = var "p" TFloat32 in
  let helper =
    {
      hf_name = "narrow_it";
      hf_params = [p];
      hf_ret_type = TFloat16;
      hf_body = SReturn (narrow (EVar p));
    }
  in
  expect_barriers
    "helper (p) = f16(p)"
    (kernel
       ~funcs:[helper]
       (SAssign
          ( LArrayElem ("out", EVar tid),
            EApp (EVar (var "narrow_it" TFloat16), [h *: f32c 1.1]) )))
    1

let () =
  let case name f = Alcotest.test_case name `Quick f in
  Alcotest.run
    "f16_narrowing_barrier"
    [
      ( "elided (exact)",
        [
          case "widen alone" test_widen_alone;
          case "read-only f32 buffer" test_f32_param_alone;
          case "f32 kernel argument" test_scalar_param;
          case "h * 2" test_times_two;
          case "q8 (float intrinsic)" test_q8;
          case "q8 (cast)" test_q8_cast;
          case "q6 triple product" test_q6;
          case "let-bound product" test_let_bound;
          case "integer sum" test_integer_sum;
        ] );
      ( "elided (innocuous double rounding)",
        [
          case "h + g" test_f16_sum;
          case "h + 1000" test_f16_plus_const;
          case "0 - h" test_f16_neg;
        ] );
      ( "kept",
        [
          case "inexact constant" test_inexact_const;
          case "let-bound inexact product" test_let_bound_inexact;
          case "q4 difference" test_q4_difference;
          case "f32 scale" test_f32_scale;
          case "three-term sum" test_three_term_sum;
          case "int wider than 13 bits" test_wide_int;
          case "unbounded int" test_unknown_int;
          case "division" test_division;
          case "mutable accumulator" test_mutable;
          case "f32 buffer written in the kernel" test_written_buffer;
          case "helper parameter" test_helper_param;
        ] );
    ]
