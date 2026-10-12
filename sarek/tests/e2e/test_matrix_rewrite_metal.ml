(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The matrix-product rewrite on Metal (specs/matrix-pattern-rewrite.md,
 * "Metal"), end to end on an Apple GPU; alias e2e-metal. Self-skipping: with no
 * Metal device it prints [SKIP] and exits 0.
 *
 *   1. before any opt-in, and for integer products, the original runs and the
 *      reason is reported (no integer simdgroup_matrix on Metal);
 *   2. with accept_relaxed_accumulation, Sarek_matrix.run rewrites an f16 GEMM
 *      to 8x8x8 fragments (the report names the shape) and the result is, bit
 *      for bit, the same as
 *        - the original kernel on the same GPU,
 *        - the original kernel on the Interpreter,
 *        - the host model of the closed form (docs/design/f16-relaxed-accuracy.md
 *          section 2: acc := fl32 (acc + a*b) over k in index order),
 *      on shapes that are not multiples of the tile, with C starting at zero;
 *   3. with C accumulated (c := c + a.b) the rewrite adds C after the chain
 *      (the original starts the chain from C): the rewrite equals the
 *      C-after model, the original equals the C-first model, and the two
 *      differ (positive control: a comparison that cannot go red proves
 *      nothing); a model with one product dropped must be rejected;
 *   4. float32 operands rounded to f16 (accept_operand_rounding) follow the
 *      chain over the rounded operands.
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
open Matrix_rewrite_support
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let fl32 x = Int32.float_of_bits (Int32.bits_of_float x)

let round16 x = Sarek_float16.to_float16 x

let vec_f32 arr =
  let v = Vector.create Vector.float32 (Array.length arr) in
  Array.iteri (Vector.set v) arr ;
  v

(* acc := fl32 (acc + a*b) over k in index order, from [init]; [drop_last]
   leaves out the last product (the mutant of the positive control). *)
let chain ?(drop_last = false) ~m ~n ~k ~init a b =
  Array.init (m * n) (fun idx ->
      let i = idx / n and j = idx mod n in
      let acc = ref (init idx) in
      for x = 0 to if drop_last then k - 2 else k - 1 do
        acc := fl32 (!acc +. (a.((i * k) + x) *. b.((x * n) + j)))
      done ;
      !acc)

type inputs = {a : float array; b : float array; c0 : float array}

let f16_inputs ~full_c (m, n, k) =
  let f16 () = round16 (Random.float 2.0 -. 1.0) in
  {
    a = Array.init (m * k) (fun _ -> f16 ());
    b = Array.init (k * n) (fun _ -> f16 ());
    c0 =
      Array.init (m * n) (fun _ ->
          if full_c then fl32 (ldexp (1.0 +. Random.float 1.0) (Random.int 4))
          else 0.0);
  }

(* One launch of [kernel]: [`Rewrite] through Sarek_matrix.run, [`Plain] the
   original. [f16] selects float16 or float32 operand vectors. *)
let launch_once dev kernel ~f16 inp (m, n, k) ?precision mode =
  let block, grid = geo_2d ~m ~n ~nb:1 in
  let cv = vec_f32 inp.c0 in
  let operand arr =
    if f16 then Execute.Vec (vec_f16_of arr) else Execute.Vec (vec_f32 arr)
  in
  let i x = Execute.Int32 (Int32.of_int x) in
  let args = [operand inp.a; operand inp.b; Execute.Vec cv; i m; i n; i k] in
  let outcome =
    match mode with
    | `Rewrite ->
        Sarek_matrix.run ?precision ~device:dev ~ir:kernel ~args ~block ~grid ()
    | `Plain ->
        Execute.run_vectors_plain ~device:dev ~ir:kernel ~args ~block ~grid () ;
        Sarek_matrix.Original "(original requested)"
  in
  Transfer.flush dev ;
  (Vector.to_array cv, outcome)

let shape8 = {Sarek_coopmat.m = 8; n = 8; k = 8}

let check_rewritten what = function
  | Sarek_matrix.Rewritten {config; _}
    when config.Sarek_coopmat.cfg_shape = shape8 ->
      ok "%s: rewritten at 8x8x8" what
  | Sarek_matrix.Rewritten _ -> fail "%s: rewritten, but not at 8x8x8" what
  | Sarek_matrix.Original why -> fail "%s: not rewritten (%s)" what why

let differs expected got =
  let r = ref false in
  Array.iteri
    (fun i e ->
      if Int32.bits_of_float e <> Int32.bits_of_float got.(i) then r := true)
    expected ;
  !r

(* The comparison must reject [wrong]: it is what makes the "bit-identical"
   lines above mean something. *)
let control what ~wrong got =
  if differs wrong got then ok "%s: control rejected" what
  else fail "%s: control accepted (the comparison cannot go red)" what

let run_zero_c ~interp dev ((m, n, k) as shape) =
  let what = Printf.sprintf "metal f16 %dx%dx%d" m n k in
  let inp = f16_inputs ~full_c:false shape in
  let kernel = ir_of k_f16 in
  let rewritten, outcome =
    launch_once dev kernel ~f16:true inp shape `Rewrite
  in
  check_rewritten what outcome ;
  let original, _ = launch_once dev kernel ~f16:true inp shape `Plain in
  let oracle, _ = launch_once interp kernel ~f16:true inp shape `Plain in
  let model = chain ~m ~n ~k ~init:(fun _ -> 0.0) inp.a inp.b in
  ignore
    (compare_floats (what ^ ": rewrite vs original (Metal)") original rewritten) ;
  ignore
    (compare_floats
       (what ^ ": rewrite vs original (Interpreter)")
       oracle
       rewritten) ;
  ignore
    (compare_floats (what ^ ": rewrite vs the chain over k") model rewritten) ;
  if k >= 2 then
    control
      what
      ~wrong:(chain ~drop_last:true ~m ~n ~k ~init:(fun _ -> 0.0) inp.a inp.b)
      rewritten

let run_accumulate dev ((m, n, k) as shape) =
  let what = Printf.sprintf "metal f16 += %dx%dx%d" m n k in
  let inp = f16_inputs ~full_c:true shape in
  let kernel = ir_of k_f16_accum in
  let rewritten, outcome =
    launch_once dev kernel ~f16:true inp shape `Rewrite
  in
  check_rewritten what outcome ;
  let original, _ = launch_once dev kernel ~f16:true inp shape `Plain in
  let sums = chain ~m ~n ~k ~init:(fun _ -> 0.0) inp.a inp.b in
  let c_after = Array.mapi (fun idx s -> fl32 (inp.c0.(idx) +. s)) sums in
  let c_first = chain ~m ~n ~k ~init:(fun idx -> inp.c0.(idx)) inp.a inp.b in
  ignore
    (compare_floats (what ^ ": rewrite vs C-after-chain") c_after rewritten) ;
  ignore
    (compare_floats (what ^ ": original vs C-first chain") c_first original) ;
  if k >= 17 then control what ~wrong:c_first rewritten

let run_f32_as_f16 ~interp dev ((m, n, k) as shape) =
  let what = Printf.sprintf "metal f32 as f16 %dx%dx%d" m n k in
  let inp = f16_inputs ~full_c:false shape in
  let wide arr = Array.map (fun x -> fl32 (x *. 1.0009765625)) arr in
  let inp = {inp with a = wide inp.a; b = wide inp.b} in
  let precision = Sarek_coopmat.Operands_f16 in
  let kernel = ir_of k_f32 in
  let rewritten, outcome =
    launch_once dev kernel ~f16:false inp shape ~precision `Rewrite
  in
  check_rewritten what outcome ;
  let rounded =
    {inp with a = Array.map round16 inp.a; b = Array.map round16 inp.b}
  in
  let model = chain ~m ~n ~k ~init:(fun _ -> 0.0) rounded.a rounded.b in
  ignore
    (compare_floats
       (what ^ ": rewrite vs the chain over rounded operands")
       model
       rewritten) ;
  let oracle, _ =
    launch_once interp kernel ~f16:false inp shape ~precision `Rewrite
  in
  ignore
    (compare_floats
       (what ^ ": rewrite vs the rewrite on the Interpreter")
       oracle
       rewritten)

let refusals dev =
  let i x = Execute.Int32 (Int32.of_int x) in
  let block, grid = geo_2d ~m:4 ~n:4 ~nb:1 in
  let expect what ~needle kernel args =
    match
      Sarek_matrix.run ~device:dev ~ir:(ir_of kernel) ~args ~block ~grid ()
    with
    | Sarek_matrix.Original why ->
        if Test_helpers.string_contains ~needle why then
          ok "%s: original (%s)" what why
        else
          fail
            "%s: declined, but the reason does not name %S: %s"
            what
            needle
            why
    | Sarek_matrix.Rewritten _ -> fail "%s: rewritten" what
  in
  expect
    "int32 on Metal"
    ~needle:"integer simdgroup_matrix"
    k_2d
    [
      Execute.Vec (vec (Array.make 16 1l));
      Execute.Vec (vec (Array.make 16 1l));
      Execute.Vec (vec (Array.make 16 0l));
      i 4;
      i 4;
      i 4;
    ] ;
  expect
    "f16 before accept_relaxed_accumulation"
    ~needle:"accept_relaxed_accumulation"
    k_f16
    [
      Execute.Vec (vec_f16_of (Array.make 16 1.0));
      Execute.Vec (vec_f16_of (Array.make 16 1.0));
      Execute.Vec (vec_f32 (Array.make 16 0.0));
      i 4;
      i 4;
      i 4;
    ]

let () =
  Random.init 20261012 ;
  let devices = Array.to_list (Device.init ()) in
  match List.filter (fun d -> d.Device.framework = "Metal") devices with
  | [] -> print_endline "[SKIP] no Metal device: nothing ran"
  | metal ->
      let interp =
        List.find (fun d -> d.Device.framework = "Interpreter") devices
      in
      List.iter refusals metal ;
      Sarek_coopmat.accept_relaxed_accumulation
        ~reason:"test_matrix_rewrite_metal: the f16 chain is under test" ;
      Sarek_coopmat.accept_operand_rounding
        ~precision:Sarek_coopmat.Operands_f16
        ~reason:"test_matrix_rewrite_metal" ;
      let shapes =
        [
          (1, 1, 1); (17, 33, 5); (64, 64, 128); (100, 37, 129); (256, 256, 1024);
        ]
      in
      List.iter
        (fun dev ->
          Printf.printf "device %s\n" dev.Device.name ;
          List.iter (run_zero_c ~interp dev) shapes ;
          List.iter (run_accumulate dev) shapes ;
          List.iter (run_f32_as_f16 ~interp dev) [(17, 33, 5); (100, 37, 129)])
        metal ;
      if !failures > 0 then (
        Printf.printf "%d failure(s)\n" !failures ;
        exit 1)
      else print_endline "PASS"
