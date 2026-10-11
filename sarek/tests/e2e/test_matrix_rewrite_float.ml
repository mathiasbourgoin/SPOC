(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Matrix-product rewrite, float cases (specs/matrix-pattern-rewrite.md): the
 * refusal before the opt-ins, then for each precision the error against the
 * exact product under the derived bound, and on the interpreter bit-identity
 * with a host model of its summation order.
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
open Matrix_rewrite_support
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* ---- Float32 accumulation: error against the exact product ----------------- *)

let u32 = ldexp 1.0 (-24)

let gamma n = float n *. u32 /. (1.0 -. (float n *. u32))

(* [bound k] is the derived bound per unit of [sum |a*b|] for one precision
   (specs/matrix-pattern-rewrite.md): operand rounding (relative [u_in] per
   operand, so [2 u_in + u_in^2] per product) plus the reordered binary32
   accumulation of [n] additions. *)
type precision_case = {
  label : string;
  precision : Sarek_coopmat.operand_precision option;
  f16_inputs : bool;
  bound : int -> float;
}

let float_cases =
  let rounding u k = (2.0 *. u) +. (u *. u) +. gamma k in
  [
    {label = "f16 inputs"; precision = None; f16_inputs = true; bound = gamma};
    {
      label = "f32 as f16";
      precision = Some Sarek_coopmat.Operands_f16;
      f16_inputs = false;
      bound = rounding (ldexp 1.0 (-11));
    };
    {
      label = "f32 as bf16";
      precision = Some Sarek_coopmat.Operands_bf16;
      f16_inputs = false;
      bound = rounding (ldexp 1.0 (-8));
    };
    {
      label = "f32 as bf16x3";
      precision = Some Sarek_coopmat.Operands_bf16x3;
      f16_inputs = false;
      bound = (fun k -> u32 +. gamma (6 * k));
    };
  ]

let round16 x = Sarek_float16.to_float16 x

let vec_f16 arr =
  let v = Vector.create Vector.float16 (Array.length arr) in
  Array.iteri (Vector.set v) arr ;
  v

let vec_f32 arr =
  let v = Vector.create Vector.float32 (Array.length arr) in
  Array.iteri (Vector.set v) arr ;
  v

(* Worst |err| / (gamma_k * sum|a*b|) and the worst ratio to the case's own
   bound, over all outputs. *)
let float_errors ~m ~n ~k a b got =
  let worst_gamma = ref 0.0 and worst = ref 0.0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let exact = ref 0.0 and mag = ref 0.0 in
      for x = 0 to k - 1 do
        let p = a.((i * k) + x) *. b.((x * n) + j) in
        exact := !exact +. p ;
        mag := !mag +. Float.abs p
      done ;
      let err = Float.abs (got.((i * n) + j) -. !exact) in
      if !mag > 0.0 then (
        worst_gamma := Float.max !worst_gamma (err /. (gamma k *. !mag)) ;
        worst := Float.max !worst (err /. !mag))
    done
  done ;
  (!worst_gamma, !worst)

(* ---- The interpreter's order, reproduced on the host ------------------------ *)

(* On the interpreter the rewritten kernel accumulates in ONE known order: per
   reduction step of 16, per product of the scheme (in its list order), per k,
   acc := fl32 (acc + a * b). Reproducing it gives a bit-exact oracle, which
   catches a dropped or wrong product that the gamma bound is too loose to
   see. *)
let fl32 x = Int32.float_of_bits (Int32.bits_of_float x)

let bf16 x =
  let bits = Int32.bits_of_float x in
  let lsb = Int32.logand (Int32.shift_right_logical bits 16) 1l in
  Int32.float_of_bits
    (Int32.logand (Int32.add bits (Int32.add 0x7fffl lsb)) 0xffff0000l)

let planes pc x =
  match pc.precision with
  | None -> [x]
  | Some Sarek_coopmat.Operands_f16 -> [round16 x]
  | Some Sarek_coopmat.Operands_bf16 -> [bf16 x]
  | Some Sarek_coopmat.Operands_bf16x3 ->
      let hi = bf16 x in
      let r1 = fl32 (x -. hi) in
      let mid = bf16 r1 in
      [hi; mid; bf16 (fl32 (r1 -. mid))]

let products pc =
  match pc.precision with
  | Some Sarek_coopmat.Operands_bf16x3 ->
      [(0, 0); (0, 1); (0, 2); (1, 0); (1, 1); (2, 0)]
  | _ -> [(0, 0)]

let interpreter_model pc ~m ~n ~k a b =
  let pa = Array.map (fun x -> Array.of_list (planes pc x)) a in
  let pb = Array.map (fun x -> Array.of_list (planes pc x)) b in
  Array.init (m * n) (fun idx ->
      let i = idx / n and j = idx mod n in
      let acc = ref 0.0 in
      for step = 0 to ((k + 15) / 16) - 1 do
        List.iter
          (fun (p, q) ->
            for kk = step * 16 to min k ((step + 1) * 16) - 1 do
              acc :=
                fl32 (!acc +. (pa.((i * k) + kk).(p) *. pb.((kk * n) + j).(q)))
            done)
          (products pc)
      done ;
      !acc)

let run_float ?(positive = false) dev pc (m, n, k) =
  (* [positive]: operands in [0.5, 1), so no error cancels: with a short K this
     is the case where dropping a product of the bf16x3 split shows. *)
  let rnd () =
    if positive then 0.5 +. Random.float 0.5 else Random.float 2.0 -. 1.0
  in
  let gen len =
    Array.init len (fun _ ->
        let x = rnd () in
        if pc.f16_inputs then round16 x
        else Int32.float_of_bits (Int32.bits_of_float x))
  in
  let a = gen (m * k) and b = gen (k * n) in
  let cv = vec_f32 (Array.make (m * n) 0.0) in
  let v arr =
    if pc.f16_inputs then Execute.Vec (vec_f16 arr)
    else Execute.Vec (vec_f32 arr)
  in
  let i x = Execute.Int32 (Int32.of_int x) in
  let ir = ir_of (if pc.f16_inputs then k_f16 else k_f32) in
  let block, grid = geo_2d ~m ~n ~nb:1 in
  let outcome =
    Sarek_matrix.run
      ?precision:pc.precision
      ~device:dev
      ~ir
      ~args:[v a; v b; Execute.Vec cv; i m; i n; i k]
      ~block
      ~grid
      ()
  in
  Transfer.flush dev ;
  let got = Vector.to_array cv in
  let what =
    Printf.sprintf
      "%s %dx%dx%d%s on %s"
      pc.label
      m
      n
      k
      (if positive then " (positive)" else "")
      dev.Device.name
  in
  match outcome with
  | Sarek_matrix.Original why -> fail "%s: not rewritten (%s)" what why
  | Sarek_matrix.Rewritten _ ->
      if dev.Device.framework = "Interpreter" then
        ignore
          (compare_floats
             (what ^ " vs the interpreter-order model")
             (interpreter_model pc ~m ~n ~k a b)
             got) ;
      let wg, w = float_errors ~m ~n ~k a b got in
      let limit = pc.bound k in
      if w <= limit then
        ok
          "%s: max err = %.3g x gamma_K sum|ab| (%.3g of its own bound)"
          what
          wg
          (w /. limit)
      else
        fail
          "%s: max err %.3g sum|ab| exceeds the bound %.3g (%.3g x gamma_K)"
          what
          w
          limit
          wg

(* Without the opt-ins a float product must run as the original. *)
(* [refused dev pc what]: a 4x4x4 product of case [pc] runs as the original. *)
let refused dev pc what =
  let v arr =
    if pc.f16_inputs then Execute.Vec (vec_f16 arr)
    else Execute.Vec (vec_f32 arr)
  in
  let i x = Execute.Int32 (Int32.of_int x) in
  let block, grid = geo_2d ~m:4 ~n:4 ~nb:1 in
  let ir = ir_of (if pc.f16_inputs then k_f16 else k_f32) in
  match
    Sarek_matrix.run
      ?precision:pc.precision
      ~device:dev
      ~ir
      ~args:
        [
          v (Array.make 16 1.0);
          v (Array.make 16 1.0);
          Execute.Vec (vec_f32 (Array.make 16 0.0));
          i 4;
          i 4;
          i 4;
        ]
      ~block
      ~grid
      ()
  with
  | Sarek_matrix.Original why ->
      ok "%s on %s: original (%s)" what dev.Device.name why
  | Sarek_matrix.Rewritten _ -> fail "%s on %s: rewritten" what dev.Device.name

let case label = List.find (fun pc -> pc.label = label) float_cases

let () =
  Random.init 20261011 ;
  let devices = Array.to_list (Device.init ()) in
  let rewriters = List.filter offers_float devices in
  List.iter
    (fun dev ->
      refused dev (case "f16 inputs") "f16 before accept_relaxed_accumulation" ;
      refused dev (case "f32 as f16") "f32 before any opt-in")
    rewriters ;
  Sarek_coopmat.accept_relaxed_accumulation
    ~reason:"test_matrix_rewrite: the bounds are under test" ;
  List.iter
    (fun dev ->
      refused dev (case "f32 as bf16") "f32/bf16 before accept_operand_rounding")
    rewriters ;
  Sarek_coopmat.accept_operand_rounding
    ~precision:Sarek_coopmat.Operands_f16
    ~reason:"test_matrix_rewrite" ;
  List.iter
    (fun dev ->
      refused
        dev
        (case "f32 as bf16")
        "f32/bf16 with only f16 rounding accepted")
    rewriters ;
  List.iter
    (fun p ->
      Sarek_coopmat.accept_operand_rounding
        ~precision:p
        ~reason:"test_matrix_rewrite")
    [
      Sarek_coopmat.Operands_f16;
      Sarek_coopmat.Operands_bf16;
      Sarek_coopmat.Operands_bf16x3;
    ] ;
  List.iter
    (fun dev ->
      let shapes =
        if dev.Device.framework = "Interpreter" then [(40, 24, 20)]
        else [(64, 64, 128); (100, 37, 129); (256, 256, 1024)]
      in
      List.iter (fun pc -> List.iter (run_float dev pc) shapes) float_cases ;
      List.iter
        (fun pc -> run_float ~positive:true dev pc (32, 32, 16))
        float_cases)
    rewriters ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
