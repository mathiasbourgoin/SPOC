(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Column-major float16 operands through the launch-time matrix rewrite
 * (specs/matrix-pattern-dynamic.md): on the interpreter and gfx11 HIP, an
 * aligned region with a column-major B (B stored as [n][k]), or with A and C
 * column-major, takes the DIRECT variant (fragments loaded straight from
 * global memory in the operand's own layout); an unaligned one the staged
 * variant. Either way the result is within gamma_K of the exact product.
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@alert "-test_only"]

[@@@warning "-33"]

open Sarek
open Matrix_rewrite_support
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Spec = Sarek_execute.Matrix_specialise

(* B stored transposed: element (i, col) of B at [col * k + i]. *)
let k_f16_bt =
  [%kernel
    fun (a : float16 vector)
        (b : float16 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum :=
            fma
              (float32_of_float16 a.((row * k) + i))
              (float32_of_float16 b.((col * k) + i))
              sum
        done ;
        c.((row * n) + col) <- sum
      end]

(* A and C column-major, B row-major. *)
let k_f16_ac =
  [%kernel
    fun (a : float16 vector)
        (b : float16 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum :=
            fma
              (float32_of_float16 a.((i * m) + row))
              (float32_of_float16 b.((i * n) + col))
              sum
        done ;
        c.((col * m) + row) <- sum
      end]

type case = {
  name : string;
  kernel : unit -> Sarek_ir_types.kernel;
  a_at : m:int -> k:int -> int -> int -> int;  (** (row, i) -> position *)
  b_at : n:int -> k:int -> int -> int -> int;  (** (i, col) -> position *)
  c_at : m:int -> n:int -> int -> int -> int;  (** (row, col) -> position *)
}

let cases =
  [
    {
      name = "B column-major";
      kernel = (fun () -> ir_of k_f16_bt);
      a_at = (fun ~m:_ ~k r i -> (r * k) + i);
      b_at = (fun ~n:_ ~k i c -> (c * k) + i);
      c_at = (fun ~m:_ ~n r c -> (r * n) + c);
    };
    {
      name = "A and C column-major";
      kernel = (fun () -> ir_of k_f16_ac);
      a_at = (fun ~m ~k:_ r i -> (i * m) + r);
      b_at = (fun ~n ~k:_ i c -> (i * n) + c);
      c_at = (fun ~m ~n:_ r c -> (c * m) + r);
    };
  ]

(* Worst |got - exact| / (gamma_K * sum |a*b|) over the m x n outputs. *)
let worst_gamma case ~m ~n ~k ~a ~b got =
  let worst = ref 0.0 and gk = float k *. ldexp 1.0 (-24) in
  for r = 0 to m - 1 do
    for c = 0 to n - 1 do
      let exact = ref 0.0 and mag = ref 0.0 in
      for x = 0 to k - 1 do
        let p = a.(case.a_at ~m ~k r x) *. b.(case.b_at ~n ~k x c) in
        exact := !exact +. p ;
        mag := !mag +. Float.abs p
      done ;
      let g = got.(case.c_at ~m ~n r c) in
      if !mag > 0.0 then
        worst := Float.max !worst (Float.abs (g -. !exact) /. (gk *. !mag))
    done
  done ;
  !worst

let check case dev ~m ~n ~k ~expect_direct =
  let rnd () = Sarek_float16.to_float16 (Random.float 2.0 -. 1.0) in
  let a = Array.init (m * k) (fun _ -> rnd ())
  and b = Array.init (k * n) (fun _ -> rnd ()) in
  let cv = vec_f32_zero (m * n) in
  let i x = Execute.Int32 (Int32.of_int x) in
  let block, grid = geo_2d ~m ~n ~nb:1 in
  let outcome =
    Execute.run_vectors_outcome
      ~explicit:(dev.Device.framework = "Interpreter")
      ~device:dev
      ~ir:(case.kernel ())
      ~args:
        [
          Execute.Vec (vec_f16_of a);
          Execute.Vec (vec_f16_of b);
          Execute.Vec cv;
          i m;
          i n;
          i k;
        ]
      ~block
      ~grid
      ()
  in
  Transfer.flush dev ;
  let what =
    Printf.sprintf "f16 %s %dx%dx%d on %s" case.name m n k dev.Device.name
  in
  match outcome with
  | Spec.Original why -> fail "%s: not rewritten (%s)" what why
  | Spec.Specialised s -> (
      let direct =
        match s.variant with Spec.Direct _ -> true | Spec.Staged -> false
      in
      let worst = worst_gamma case ~m ~n ~k ~a ~b (Vector.to_array cv) in
      match () with
      | () when direct <> expect_direct ->
          fail "%s: unexpected variant %s" what (Spec.variant_name s.variant)
      | () when worst > 1.0 -> fail "%s: max err %.3g x gamma_K" what worst
      | () ->
          ok
            "%s: %s, max err %.3g x gamma_K"
            what
            (Spec.variant_name s.variant)
            worst)

let () =
  Random.init 20261011 ;
  Sarek_coopmat.accept_relaxed_accumulation ~reason:"test_matrix_col_major" ;
  Sarek_coopmat.accept_automatic_float_rewrite ~reason:"test_matrix_col_major" ;
  Spec.set_mode Spec.Auto ;
  let rewriters = List.filter offers_int (Array.to_list (Device.init ())) in
  List.iter
    (fun dev ->
      List.iter
        (fun case ->
          check case dev ~m:64 ~n:64 ~k:64 ~expect_direct:true ;
          check case dev ~m:64 ~n:96 ~k:48 ~expect_direct:true ;
          check case dev ~m:40 ~n:24 ~k:20 ~expect_direct:false)
        cases)
    rewriters ;
  if rewriters = [] then print_endline "[SKIP] no device rewrites" ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
