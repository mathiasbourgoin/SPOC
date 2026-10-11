(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Launch-time matrix specialisation (specs/matrix-pattern-dynamic.md).
 *
 * Execute.run_vectors itself now rewrites a matrix product when the mode and
 * the opt-ins allow it. Checked on every device: the default (Auto) rewrites
 * int32 on the interpreter and gfx11 HIP and nothing else; Off and Exact do
 * what they say; the aligned float16 size class takes the direct variant and
 * the unaligned one the staged variant; the decision is reported once and
 * the rewritten kernel is cached by IR, device, mode and size class.
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
open Matrix_rewrite_support
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Spec = Sarek_execute.Matrix_specialise

let is_rewritten = function
  | Spec.Specialised _ -> true
  | Spec.Original _ -> false

(* The int32 2-D kernel through Execute.run_vectors_outcome and through the
   plain launch, from the same C. *)
let int_run dev ~m ~n ~k =
  let a = Array.init (m * k) (fun _ -> random_int32 ()) in
  let b = Array.init (k * n) (fun _ -> random_int32 ()) in
  let c0 = Array.init (m * n) (fun i -> Int32.add sentinel (Int32.of_int i)) in
  let block, grid = geo_2d ~m ~n ~nb:1 in
  let go plain =
    let cv = vec c0 in
    let i x = Execute.Int32 (Int32.of_int x) in
    let args =
      [Execute.Vec (vec a); Execute.Vec (vec b); Execute.Vec cv; i m; i n; i k]
    in
    let ir = ir_of k_2d in
    let outcome =
      if plain then (
        Execute.run_vectors_plain ~device:dev ~ir ~args ~block ~grid () ;
        Spec.Original "plain")
      else Execute.run_vectors_outcome ~device:dev ~ir ~args ~block ~grid ()
    in
    Transfer.flush dev ;
    (Vector.to_array cv, outcome)
  in
  let original, _ = go true in
  let got, outcome = go false in
  (original, got, outcome)

let check_mode dev ~mode ~expect =
  Spec.set_mode mode ;
  let original, got, outcome = int_run dev ~m:40 ~n:24 ~k:20 in
  let what =
    Printf.sprintf "int32, mode %s, on %s" (Spec.mode_name mode) dev.Device.name
  in
  if is_rewritten outcome <> expect then
    fail "%s: %s" what (Spec.outcome_text outcome)
  else if compare_arrays what original got then
    ok "%s: %s, bit-identical" what (Spec.outcome_text outcome)

(* float16 operands: aligned -> direct, unaligned -> staged; within gamma_K. *)
let check_size_class ?(accumulate = false) dev ~m ~n ~k ~expect_direct =
  let rnd () = Sarek_float16.to_float16 (Random.float 2.0 -. 1.0) in
  let a = Array.init (m * k) (fun _ -> rnd ())
  and b = Array.init (k * n) (fun _ -> rnd ()) in
  let c0 =
    Array.init (m * n) (fun _ ->
        if accumulate then Random.float 2.0 -. 1.0 else 0.0)
  in
  let cv = vec_f32_zero (m * n) in
  Array.iteri (Vector.set cv) c0 ;
  let i x = Execute.Int32 (Int32.of_int x) in
  let block, grid = geo_2d ~m ~n ~nb:1 in
  let outcome =
    Execute.run_vectors_outcome
      ~device:dev
      ~ir:(ir_of (if accumulate then k_f16_accum else k_f16))
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
    Printf.sprintf
      "f16 %dx%dx%d%s on %s"
      m
      n
      k
      (if accumulate then " into C" else "")
      dev.Device.name
  in
  match outcome with
  | Spec.Original why -> fail "%s: not rewritten (%s)" what why
  | Spec.Specialised s ->
      let direct =
        match s.variant with Spec.Direct _ -> true | Spec.Staged -> false
      in
      if direct <> expect_direct then
        fail "%s: unexpected variant %s" what (Spec.variant_name s.variant)
      else
        let worst = ref 0.0 in
        let got = Vector.to_array cv in
        for r = 0 to m - 1 do
          for c = 0 to n - 1 do
            let exact = ref c0.((r * n) + c)
            and mag = ref (Float.abs c0.((r * n) + c)) in
            for x = 0 to k - 1 do
              let p = a.((r * k) + x) *. b.((x * n) + c) in
              exact := !exact +. p ;
              mag := !mag +. Float.abs p
            done ;
            let gk = float k *. ldexp 1.0 (-24) in
            if !mag > 0.0 then
              worst :=
                Float.max
                  !worst
                  (Float.abs (got.((r * n) + c) -. !exact) /. (gk *. !mag))
          done
        done ;
        if !worst <= 1.0 then
          ok
            "%s: %s, max err %.3g x gamma_K"
            what
            (Spec.variant_name s.variant)
            !worst
        else fail "%s: max err %.3g x gamma_K" what !worst

(* Two launches of one kernel on one device reuse the rewritten IR. *)
let check_cache dev =
  Spec.set_mode Spec.Auto ;
  let _, _, o1 = int_run dev ~m:17 ~n:33 ~k:5 in
  let _, _, o2 = int_run dev ~m:17 ~n:33 ~k:5 in
  match (o1, o2) with
  | Spec.Specialised s1, Spec.Specialised s2 when s1.ir == s2.ir ->
      ok "cache on %s: one rewritten IR for both launches" dev.Device.name
  | _ -> fail "cache on %s: rewritten IR not reused" dev.Device.name

let check_report dev =
  let seen = ref [] in
  Spec.set_reporter (Some (fun r -> seen := r :: !seen)) ;
  let _ = int_run dev ~m:8 ~n:8 ~k:8 in
  Spec.set_reporter None ;
  match !seen with
  | {Spec.outcome = Spec.Specialised _; _} :: _ ->
      ok "report on %s: the rewrite was reported" dev.Device.name
  | _ -> fail "report on %s: no rewrite reported" dev.Device.name

let () =
  Random.init 20261011 ;
  let devices = Array.to_list (Device.init ()) in
  List.iter
    (fun dev ->
      let rewrites = offers_int dev in
      check_mode dev ~mode:Spec.Auto ~expect:rewrites ;
      check_mode dev ~mode:Spec.Exact ~expect:rewrites ;
      check_mode dev ~mode:Spec.Off ~expect:false ;
      Spec.set_mode Spec.Auto ;
      if rewrites then (
        check_cache dev ;
        check_report dev))
    devices ;
  let rewriters = List.filter offers_int devices in
  Sarek_coopmat.accept_relaxed_accumulation ~reason:"test_matrix_dynamic" ;
  List.iter
    (fun dev ->
      Spec.set_mode Spec.Exact ;
      let _ = Random.bits () in
      (match
         Execute.run_vectors_outcome
           ~device:dev
           ~ir:(ir_of k_f16)
           ~args:
             [
               Execute.Vec (vec_f16_of (Array.make 16 1.0));
               Execute.Vec (vec_f16_of (Array.make 16 1.0));
               Execute.Vec (vec_f32_zero 16);
               Execute.Int32 4l;
               Execute.Int32 4l;
               Execute.Int32 4l;
             ]
           ~block:(Execute.dims3d 16 16 1)
           ~grid:(Execute.dims3d 1 1 1)
           ()
       with
      | Spec.Original why ->
          ok "f16 in mode exact on %s: original (%s)" dev.Device.name why
      | Spec.Specialised _ ->
          fail "f16 rewritten in mode exact on %s" dev.Device.name) ;
      Spec.set_mode Spec.Auto ;
      let big = dev.Device.framework <> "Interpreter" in
      check_size_class dev ~m:64 ~n:64 ~k:64 ~expect_direct:true ;
      check_size_class dev ~m:40 ~n:24 ~k:20 ~expect_direct:false ;
      (* Aligned but accumulating into C: staged (see Sarek_matrix_direct). *)
      check_size_class
        ~accumulate:true
        dev
        ~m:64
        ~n:64
        ~k:64
        ~expect_direct:false ;
      check_size_class
        ~accumulate:true
        dev
        ~m:40
        ~n:24
        ~k:20
        ~expect_direct:false ;
      if big then (
        check_size_class dev ~m:256 ~n:256 ~k:1024 ~expect_direct:true ;
        check_size_class dev ~m:100 ~n:37 ~k:129 ~expect_direct:false))
    rewriters ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
