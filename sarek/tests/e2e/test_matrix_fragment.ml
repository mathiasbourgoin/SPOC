(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The rewrite templates are parameterised by the fragment edge
 * (specs/matrix-pattern-rewrite.md, "Fragment shape"): 16 where the matrix unit
 * offers 16x16x16 (HIP gfx11, CUDA, Vulkan), 8 where it offers 8x8x8 only
 * (Metal simdgroup_matrix). On the interpreter, the oracle, the f16 rewrite is
 * laid out for fragment 8 (a 4x4 grid of fragments per wave, two chained k-steps
 * of 8 per staged step of 16) and for fragment 16, at subgroup width 32, and
 * must equal
 *   - the original kernel bit for bit when C starts at zero (the chain over k
 *     is the same: acc := fl32 (acc + a*b), products of binary16 values being
 *     exact in binary32), and
 *   - the host model of the chain, C added after the chain.
 * The launch the rewrite chose is read back from the report, so a fragment
 * that was ignored shows as a wrong configuration. Positive control: with a
 * full-mantissa C the rewrite (C after the chain) must DIFFER from the original
 * (C first); a comparison that could not go red proves nothing.
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@alert "-test_only"]

[@@@warning "-33"]

open Sarek
open Matrix_rewrite_support
module Spec = Sarek_execute.Matrix_specialise
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let shapes = [(1, 1, 1); (17, 33, 5); (40, 24, 20); (33, 65, 17); (64, 64, 64)]

let fl32 x = Int32.float_of_bits (Int32.bits_of_float x)

let round16 x = Sarek_float16.to_float16 x

let vec_f32 arr =
  let v = Vector.create Vector.float32 (Array.length arr) in
  Array.iteri (Vector.set v) arr ;
  v

(* The chain over k, from [init]: acc := fl32 (acc + a*b). *)
let chain ~m ~n ~k ~init a b =
  Array.init (m * n) (fun idx ->
      let i = idx / n and j = idx mod n in
      let acc = ref (init idx) in
      for x = 0 to k - 1 do
        acc := fl32 (!acc +. (a.((i * k) + x) *. b.((x * n) + j)))
      done ;
      !acc)

let last_shape = ref None

let () =
  Spec.set_reporter
    (Some
       (fun r ->
         match r.Spec.outcome with
         | Spec.Specialised s ->
             last_shape := Some s.Spec.config.Sarek_coopmat.cfg_shape
         | Spec.Original _ -> last_shape := None))

let launch_once dev kernel ~c0 a b (m, n, k) ~rewrite =
  let block, grid = geo_2d ~m ~n ~nb:1 in
  let cv = vec_f32 c0 in
  let i x = Execute.Int32 (Int32.of_int x) in
  let args =
    [
      Execute.Vec (vec_f16_of a);
      Execute.Vec (vec_f16_of b);
      Execute.Vec cv;
      i m;
      i n;
      i k;
    ]
  in
  last_shape := None ;
  let outcome = launch ~rewrite dev kernel ~args ~block ~grid in
  Transfer.flush dev ;
  (Vector.to_array cv, outcome)

let check_shape what ~frag outcome =
  match (outcome, !last_shape) with
  | Sarek_matrix.Rewritten _, Some s
    when s.Sarek_coopmat.m = frag && s.n = frag && s.k = frag ->
      ()
  | Sarek_matrix.Rewritten _, _ ->
      fail "%s: not laid out for %dx%dx%d fragments" what frag frag frag
  | Sarek_matrix.Original why, _ -> fail "%s: not rewritten (%s)" what why

let operands (m, n, k) =
  let f16 () = round16 (Random.float 2.0 -. 1.0) in
  (Array.init (m * k) (fun _ -> f16 ()), Array.init (k * n) (fun _ -> f16 ()))

let run_zero_c dev ~frag ((m, n, k) as shape) =
  let what = Printf.sprintf "f16 %dx%dx%d, fragment %d" m n k frag in
  let a, b = operands shape in
  let c0 = Array.make (m * n) 0.0 in
  let kernel = ir_of k_f16 in
  let original, _ = launch_once dev kernel ~c0 a b shape ~rewrite:false in
  let rewritten, outcome = launch_once dev kernel ~c0 a b shape ~rewrite:true in
  check_shape what ~frag outcome ;
  let model = chain ~m ~n ~k ~init:(fun _ -> 0.0) a b in
  let same_original =
    compare_floats (what ^ ": rewrite vs original") original rewritten
  in
  let same_model =
    compare_floats (what ^ ": rewrite vs the chain over k") model rewritten
  in
  ignore (same_original && same_model)

(* C accumulated: the rewrite adds C to the finished chain, the original starts
   the chain from C. *)
let run_accumulate dev ~frag ((m, n, k) as shape) =
  let what = Printf.sprintf "f16 += %dx%dx%d, fragment %d" m n k frag in
  let a, b = operands shape in
  let c0 =
    Array.init (m * n) (fun _ ->
        fl32 (ldexp (1.0 +. Random.float 1.0) (Random.int 4)))
  in
  let kernel = ir_of k_f16_accum in
  let original, _ = launch_once dev kernel ~c0 a b shape ~rewrite:false in
  let rewritten, outcome = launch_once dev kernel ~c0 a b shape ~rewrite:true in
  check_shape what ~frag outcome ;
  let sums = chain ~m ~n ~k ~init:(fun _ -> 0.0) a b in
  let c_after = Array.mapi (fun idx s -> fl32 (c0.(idx) +. s)) sums in
  let c_first = chain ~m ~n ~k ~init:(fun idx -> c0.(idx)) a b in
  ignore
    (compare_floats (what ^ ": original vs C-first chain") c_first original) ;
  ignore
    (compare_floats (what ^ ": rewrite vs C-after-chain") c_after rewritten) ;
  (* The control: the two orders differ on this input, so the comparisons
     above can tell them apart (skipped where k is too short to round). *)
  if k >= 17 then
    if c_after = c_first then
      fail "%s: control cannot go red (C-after equals C-first)" what
    else ok "%s: control (C-after differs from C-first)" what

let () =
  Random.init 20261012 ;
  Sarek_coopmat.accept_relaxed_accumulation
    ~reason:"test_matrix_fragment: the f16 chain is under test" ;
  let dev =
    List.hd
      (List.filter
         (fun d -> d.Device.framework = "Interpreter")
         (Array.to_list (Device.init ())))
  in
  List.iter
    (fun frag ->
      Spec.fragment_override_for_tests (Some frag) ;
      Spec.clear () ;
      List.iter (run_zero_c dev ~frag) shapes ;
      List.iter (run_accumulate dev ~frag) shapes)
    [8; 16] ;
  Spec.fragment_override_for_tests None ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
