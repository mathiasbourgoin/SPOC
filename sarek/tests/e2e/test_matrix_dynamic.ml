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

[@@@alert "-test_only"]

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
let int_run ?(explicit = false) dev ~m ~n ~k =
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
      else
        Execute.run_vectors_outcome
          ~explicit
          ~device:dev
          ~ir
          ~args
          ~block
          ~grid
          ()
    in
    Transfer.flush dev ;
    (Vector.to_array cv, outcome)
  in
  let original, _ = go true in
  let got, outcome = go false in
  (original, got, outcome)

let check_mode ?(explicit = false) dev ~mode ~expect =
  Spec.set_mode mode ;
  let original, got, outcome = int_run ~explicit dev ~m:40 ~n:24 ~k:20 in
  let what =
    Printf.sprintf
      "int32, mode %s%s, on %s"
      (Spec.mode_name mode)
      (if explicit then ", explicit" else "")
      dev.Device.name
  in
  if is_rewritten outcome <> expect then
    fail "%s: %s" what (Spec.outcome_text outcome)
  else if compare_arrays what original got then
    ok "%s: %s, bit-identical" what (Spec.outcome_text outcome)

(* Worst |got - exact| / (gamma_K * (sum |a*b| + |c0|)) over [nb] batched
   m x n x k products, A and B row-major, batches stacked. *)
let worst_gamma ~m ~n ~k ~nb ~a ~b ~c0 got =
  let worst = ref 0.0 and gk = float k *. ldexp 1.0 (-24) in
  for z = 0 to nb - 1 do
    for r = 0 to m - 1 do
      for c = 0 to n - 1 do
        let o = (z * m * n) + (r * n) + c in
        let exact = ref c0.(o) and mag = ref (Float.abs c0.(o)) in
        for x = 0 to k - 1 do
          let p =
            a.((z * m * k) + (r * k) + x) *. b.((z * k * n) + (x * n) + c)
          in
          exact := !exact +. p ;
          mag := !mag +. Float.abs p
        done ;
        if !mag > 0.0 then
          worst :=
            Float.max !worst (Float.abs (got.(o) -. !exact) /. (gk *. !mag))
      done
    done
  done ;
  !worst

(* float16 operands: aligned -> direct, unaligned -> staged; within gamma_K. *)
let launch_f16 dev ~accumulate ~a ~b cv ~m ~n ~k =
  let i x = Execute.Int32 (Int32.of_int x) in
  let block, grid = geo_2d ~m ~n ~nb:1 in
  let o =
    Execute.run_vectors_outcome
      ~explicit:(dev.Device.framework = "Interpreter")
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
  o

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
  let outcome = launch_f16 dev ~accumulate ~a ~b cv ~m ~n ~k in
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
        let worst =
          ref (worst_gamma ~m ~n ~k ~nb:1 ~a ~b ~c0 (Vector.to_array cv))
        in
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
  let explicit = dev.Device.framework = "Interpreter" in
  let _, _, o1 = int_run ~explicit dev ~m:17 ~n:33 ~k:5 in
  let _, _, o2 = int_run ~explicit dev ~m:17 ~n:33 ~k:5 in
  match (o1, o2) with
  | Spec.Specialised s1, Spec.Specialised s2 when s1.ir == s2.ir ->
      ok "cache on %s: one rewritten IR for both launches" dev.Device.name
  | _ -> fail "cache on %s: rewritten IR not reused" dev.Device.name

let check_report dev =
  let seen = ref [] in
  Spec.set_reporter (Some (fun r -> seen := r :: !seen)) ;
  let _ =
    int_run ~explicit:(dev.Device.framework = "Interpreter") dev ~m:8 ~n:8 ~k:8
  in
  Spec.set_reporter None ;
  match !seen with
  | {Spec.outcome = Spec.Specialised _; _} :: _ ->
      ok "report on %s: the rewrite was reported" dev.Device.name
  | _ -> fail "report on %s: no rewrite reported" dev.Device.name

(* ---- Review regressions ----------------------------------------------- *)

(* The reduction length reads block_dim_x: it changes meaning under the
   rewritten launch's own block, so the rewrite must decline. *)
let k_f16_blockdim =
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
        for i = 0 to block_dim_x - 1l do
          sum :=
            sum
            +. float32_of_float16 a.((row * k) + i)
               *. float32_of_float16 b.((i * n) + col)
        done ;
        c.((row * n) + col) <- sum
      end]

let small_f16_args ~m ~n ~k extra =
  [
    Execute.Vec (vec_f16_of (Array.make (m * k) 0.5));
    Execute.Vec (vec_f16_of (Array.make (k * n) 0.25));
    Execute.Vec (vec_f32_zero (m * n));
  ]
  @ List.map (fun x -> Execute.Int32 (Int32.of_int x)) extra

let check_blockdim dev =
  match
    Execute.run_vectors_outcome
      ~explicit:true
      ~device:dev
      ~ir:(ir_of k_f16_blockdim)
      ~args:(small_f16_args ~m:64 ~n:64 ~k:16 [64; 64; 16])
      ~block:(Execute.dims3d 16 16 1)
      ~grid:(Execute.dims3d 4 4 1)
      ()
  with
  | Spec.Original why ->
      ok "block_dim bound on %s: original (%s)" dev.Device.name why
  | Spec.Specialised _ ->
      fail "block_dim bound on %s: rewritten" dev.Device.name

(* Direct with one wave per block (an odd tile count) and with batches on
   grid y. *)
let k_f16_batched =
  [%kernel
    fun (a : float16 vector)
        (b : float16 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32)
        (nb : int32) ->
      let z = global_idx_z in
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n && z < nb then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum :=
            sum
            +. float32_of_float16 a.((z * m * k) + (row * k) + i)
               *. float32_of_float16 b.((z * k * n) + (i * n) + col)
        done ;
        c.((z * m * n) + (row * n) + col) <- sum
      end]

let check_direct_batched dev ~m ~n ~k ~nb ~waves =
  let rnd () = Sarek_float16.to_float16 (Random.float 2.0 -. 1.0) in
  let a = Array.init (nb * m * k) (fun _ -> rnd ()) in
  let b = Array.init (nb * k * n) (fun _ -> rnd ()) in
  let cv = vec_f32_zero ((nb * m * n) + 16) in
  let i x = Execute.Int32 (Int32.of_int x) in
  let outcome =
    Execute.run_vectors_outcome
      ~explicit:(dev.Device.framework = "Interpreter")
      ~device:dev
      ~ir:(ir_of k_f16_batched)
      ~args:
        [
          Execute.Vec (vec_f16_of a);
          Execute.Vec (vec_f16_of b);
          Execute.Vec cv;
          i m;
          i n;
          i k;
          i nb;
        ]
      ~block:(Execute.dims3d 16 16 1)
      ~grid:(Execute.dims3d (cdiv n 16) (cdiv m 16) nb)
      ()
  in
  Transfer.flush dev ;
  let got = Vector.to_array cv in
  let what =
    Printf.sprintf "f16 %dx%dx%d x%d batches on %s" m n k nb dev.Device.name
  in
  match outcome with
  | Spec.Specialised {variant = Spec.Direct w; _} when w = waves ->
      let worst =
        ref
          (worst_gamma ~m ~n ~k ~nb ~a ~b ~c0:(Array.make (nb * m * n) 0.0) got)
      in
      let tail_clean =
        Array.for_all (fun x -> x = 0.0) (Array.sub got (nb * m * n) 16)
      in
      if !worst <= 1.0 && tail_clean then
        ok
          "%s: direct (%d waves), %.3g x gamma_K, nothing written past C"
          what
          w
          !worst
      else fail "%s: %.3g x gamma_K, tail clean %b" what !worst tail_clean
  | o ->
      fail "%s: %s (expected direct, %d waves)" what (Spec.outcome_text o) waves

let k_not_a_gemm =
  [%kernel
    fun (a : int32 vector) (n : int32) ->
      let i = global_idx_x in
      if i < n then a.(i) <- a.(i) + 1l]

let check_not_reported dev =
  let seen = ref 0 in
  Spec.set_reporter (Some (fun _ -> incr seen)) ;
  Execute.run_vectors
    ~device:dev
    ~ir:(ir_of k_not_a_gemm)
    ~args:[Execute.Vec (vec (Array.make 64 1l)); Execute.Int32 64l]
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush dev ;
  Spec.set_reporter None ;
  if !seen = 0 then ok "non-matrix kernel on %s: not reported" dev.Device.name
  else fail "non-matrix kernel on %s: reported" dev.Device.name

(* f32 without ~precision after accepting f16 and bf16x3 rounding: float16
   (bf16x3 only on request). *)
let check_auto_precision dev =
  let outcome =
    Execute.run_vectors_outcome
      ~explicit:(dev.Device.framework = "Interpreter")
      ~device:dev
      ~ir:(ir_of k_f32)
      ~args:
        [
          Execute.Vec (vec_f32_zero 1024);
          Execute.Vec (vec_f32_zero 1024);
          Execute.Vec (vec_f32_zero 1024);
          Execute.Int32 32l;
          Execute.Int32 32l;
          Execute.Int32 32l;
        ]
      ~block:(Execute.dims3d 16 16 1)
      ~grid:(Execute.dims3d 2 2 1)
      ()
  in
  match outcome with
  | Spec.Specialised {lowering = Sarek_matrix_rewrite.F16_operands; _} ->
      ok "f32 auto precision on %s: f16" dev.Device.name
  | o ->
      fail "f32 auto precision on %s: %s" dev.Device.name (Spec.outcome_text o)

let int_phase dev =
  let rewrites = offers_int dev in
  (* Explicit-request-only devices: the interpreter stays the oracle, CUDA/C
     has compiler-output evidence only. *)
  let interp =
    dev.Device.framework = "Interpreter" || dev.Device.framework = "CUDA/C"
  in
  check_mode dev ~mode:Spec.Auto ~expect:(rewrites && not interp) ;
  check_mode dev ~mode:Spec.Exact ~expect:(rewrites && not interp) ;
  check_mode dev ~mode:Spec.Off ~expect:false ;
  check_mode ~explicit:true dev ~mode:Spec.Auto ~expect:rewrites ;
  check_mode ~explicit:true dev ~mode:Spec.Off ~expect:false ;
  Spec.set_mode Spec.Auto ;
  if rewrites then (
    check_cache dev ;
    check_report dev)

(* Mode exact leaves float16 alone even after the opt-in (explicit request,
   so the interpreter is asked too). *)
let check_exact_f16 dev =
  Spec.set_mode Spec.Exact ;
  let outcome =
    Execute.run_vectors_outcome
      ~explicit:true
      ~device:dev
      ~ir:(ir_of k_f16)
      ~args:(small_f16_args ~m:4 ~n:4 ~k:4 [4; 4; 4])
      ~block:(Execute.dims3d 16 16 1)
      ~grid:(Execute.dims3d 1 1 1)
      ()
  in
  Spec.set_mode Spec.Auto ;
  match outcome with
  | Spec.Original why ->
      ok "f16 in mode exact on %s: original (%s)" dev.Device.name why
  | Spec.Specialised _ ->
      fail "f16 rewritten in mode exact on %s" dev.Device.name

let float_phase dev =
  check_exact_f16 dev ;
  check_size_class dev ~m:64 ~n:64 ~k:64 ~expect_direct:true ;
  check_size_class dev ~m:40 ~n:24 ~k:20 ~expect_direct:false ;
  (* Aligned but accumulating into C: staged (see Sarek_matrix_direct). *)
  check_size_class ~accumulate:true dev ~m:64 ~n:64 ~k:64 ~expect_direct:false ;
  check_size_class ~accumulate:true dev ~m:40 ~n:24 ~k:20 ~expect_direct:false ;
  check_blockdim dev ;
  check_direct_batched dev ~m:32 ~n:96 ~k:32 ~nb:1 ~waves:1 ;
  check_direct_batched dev ~m:64 ~n:64 ~k:48 ~nb:3 ~waves:2 ;
  if dev.Device.framework <> "Interpreter" then (
    check_size_class dev ~m:256 ~n:256 ~k:1024 ~expect_direct:true ;
    check_size_class dev ~m:100 ~n:37 ~k:129 ~expect_direct:false)

(* A defect in the analysis must not fail a launch: inject one, launch a
   fresh (never analysed) kernel by the default path, check it ran. *)
let k_add_one =
  [%kernel
    fun (a : int32 vector) (n : int32) ->
      let i = global_idx_x in
      if i < n then a.(i) <- a.(i) + 2l]

let check_analysis_failure dev =
  Spec.fail_analysis_for_tests true ;
  let v = vec (Array.make 64 1l) in
  let result =
    try
      Execute.run_vectors
        ~device:dev
        ~ir:(ir_of k_add_one)
        ~args:[Execute.Vec v; Execute.Int32 64l]
        ~block:(Execute.dims1d 64)
        ~grid:(Execute.dims1d 1)
        () ;
      Transfer.flush dev ;
      Ok (Vector.to_array v)
    with e -> Error (Printexc.to_string e)
  in
  Spec.fail_analysis_for_tests false ;
  Spec.clear () ;
  match result with
  | Ok out when Array.for_all (fun x -> x = 3l) out ->
      ok "injected analysis failure on %s: the launch ran" dev.Device.name
  | Ok _ -> fail "injected analysis failure on %s: wrong result" dev.Device.name
  | Error e -> fail "injected analysis failure on %s: %s" dev.Device.name e

let () =
  Random.init 20261011 ;
  let devices = Array.to_list (Device.init ()) in
  List.iter int_phase devices ;
  let rewriters = List.filter offers_float devices in
  Sarek_coopmat.accept_relaxed_accumulation ~reason:"test_matrix_dynamic" ;
  Sarek_coopmat.accept_automatic_float_rewrite ~reason:"test_matrix_dynamic" ;
  List.iter float_phase rewriters ;
  List.iter check_not_reported devices ;
  List.iter check_analysis_failure devices ;
  List.iter
    (fun p ->
      Sarek_coopmat.accept_operand_rounding
        ~precision:p
        ~reason:"test_matrix_dynamic")
    [Sarek_coopmat.Operands_f16; Sarek_coopmat.Operands_bf16x3] ;
  List.iter check_auto_precision rewriters ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
