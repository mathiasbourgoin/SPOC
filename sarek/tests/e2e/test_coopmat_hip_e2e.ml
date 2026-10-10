(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * SCoopmat on HIP (RDNA3 WMMA) end to end — specs/coopmat-hip-wmma.md.
 *
 * Order matters, because the float opt-in is process-wide and one-way:
 *   1. codegen: the emitted HIP/CUDA source names the instructions;
 *   2. refusals BEFORE the opt-in: every device and the interpreter refuse
 *      the f16 configuration; the gfx11 HIP device refuses it for the POLICY
 *      reason, every other HIP device for the device reason; integer kernels
 *      are permitted on gfx11 and refused elsewhere; a partial-wave block is
 *      refused;
 *   3. integer configurations on gfx11: bit-identical to the interpreter over
 *      all 65536 8-bit operand pairs, for the four sign combinations and both
 *      32-bit accumulators, with a C-dropping control rejected;
 *   4. the opt-in, then f16: one tile and a tiled GEMM against the exact
 *      result under the Regime B bound, the interpreter under the same bound,
 *      both §5.4 controls rejected, repeated runs and three tilings bit-equal.
 *
 * Skips (exit 0) the device parts when no gfx11 HIP device is present; the
 * refusal part runs on whatever devices exist.
 ******************************************************************************)

open Sarek
open Coopmat_hip_support
module Gemm = Sarek_gemm.Coopmat
module Fw = Spoc_framework.Framework_sig

let () =
  Backend_hip.init () ;
  Test_helpers.Benchmarks.init_backends ()

let failures = ref 0

let fail fmt =
  Printf.ksprintf
    (fun s ->
      incr failures ;
      print_endline ("  FAIL " ^ s))
    fmt

let ok fmt = Printf.ksprintf (fun s -> print_endline ("  OK  " ^ s)) fmt

let offers dev cfg =
  Sarek_capability.permits
    (C.verdict ~support:dev.Device.capabilities.Fw.coopmat cfg)

let is_hip dev = dev.Device.framework = "HIP"

let int_cfg = config C.Uint8 C.Uint8 C.Sint32

(* ---------------------------------------------------------------- 1 *)

let check_codegen () =
  let src ir = Sarek_codegen.Sarek_ir_cuda.generate ir in
  let gemm = src (Gemm.ir ()) in
  let int_src = src (tile_ir (config C.Uint8 C.Sint8 C.Sint32)) in
  let expect what needle s =
    if contains ~needle s then ok "%s contains %S" what needle
    else fail "%s does not contain %S" what needle
  in
  expect "f16 GEMM" "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(" gemm ;
  expect "f16 GEMM" "nvcuda::wmma::mma_sync(" gemm ;
  expect "f16 GEMM" "Sarek_coopmat.accept_relaxed_accumulation" gemm ;
  expect
    "u8*s8 tile"
    "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, fa, true, fb, fc, false)"
    int_src ;
  if contains ~needle:"accept_relaxed_accumulation" int_src then
    fail "an integer kernel carries the relaxed-accumulation header"

(* ---------------------------------------------------------------- 2 *)

let gate dev ir () = Execute.check_device_capabilities ~device:dev ir

let check_f16_refusals devices =
  let ir = Gemm.ir () in
  List.iter
    (fun dev ->
      match refusal (gate dev ir) with
      | None ->
          fail "%s permitted f16 coopmat before the opt-in" dev.Device.name
      | Some msg when offers dev f16_config ->
          if contains ~needle:"accept_relaxed_accumulation" msg then
            ok "%s refuses f16 for the policy reason" dev.Device.name
          else
            fail "%s: refusal does not name the opt-in: %s" dev.Device.name msg
      | Some msg when is_hip dev ->
          if contains ~needle:"HIP:" msg then
            ok "%s (HIP, no WMMA) refuses with the HIP rule" dev.Device.name
          else
            fail
              "%s: HIP refusal does not name the rule: %s"
              dev.Device.name
              msg
      | Some _ -> ok "%s refuses f16 coopmat" dev.Device.name)
    devices ;
  let a = f16_vec (operand 256) and b = f16_vec (operand 256) in
  let c = f32_vec (addend 256) and d = f32_vec (Array.make 256 0.0) in
  match
    refusal (fun () ->
        run_interp
          (tile_ir f16_config)
          Execute.[Vec a; Vec b; Vec c; Vec d]
          ~block:(Execute.dims1d 32)
          ~grid:(Execute.dims1d 1))
  with
  | None -> fail "the interpreter evaluated f16 coopmat before the opt-in"
  | Some _ -> ok "the interpreter refuses f16 coopmat before the opt-in"

let check_int_gate devices =
  let ir = tile_ir int_cfg in
  List.iter
    (fun dev ->
      let permitted = refusal (gate dev ir) = None in
      match (is_hip dev, offers dev int_cfg, permitted) with
      | true, true, true -> ok "%s (gfx11) permits u8*u8+s32" dev.Device.name
      | true, false, false ->
          ok "%s (no WMMA) refuses u8*u8+s32" dev.Device.name
      | _, offered, true when not offered ->
          fail "%s permits a configuration it does not offer" dev.Device.name
      | true, true, false ->
          fail "%s offers u8*u8+s32 but refuses" dev.Device.name
      | _ -> ())
    devices

let check_block_refusal dev =
  let v n = i32_vec (Array.make n 0l) in
  let args =
    Execute.
      [
        Vec (u8_vec (Array.make 256 1));
        Vec (u8_vec (Array.make 256 1));
        Vec (v 256);
        Vec (v 256);
      ]
  in
  match
    refusal (fun () ->
        run_gpu
          dev
          (tile_ir int_cfg)
          args
          ~block:(Execute.dims1d 16)
          ~grid:(Execute.dims1d 1))
  with
  | None ->
      fail "%s launched a coopmat kernel with a 16-thread block" dev.Device.name
  | Some msg when contains ~needle:"32-wide" msg -> ok "16-thread block refused"
  | Some msg -> fail "16-thread block refused for another reason: %s" msg

(* ---------------------------------------------------------------- 3 *)

(* A[i][k] = 16k + i, B[k][j] = 16t + j: tile t covers 4096 byte pairs and the
   sixteen tiles cover all 65536 exactly once (test_coopmat_integer_e2e). *)
let int_inputs t =
  let a = Array.init 256 (fun p -> ((16 * (p mod 16)) + (p / 16)) land 0xff) in
  let b = Array.init 256 (fun p -> ((16 * t) + (p mod 16)) land 0xff) in
  let c = Array.init 256 (fun p -> Int32.of_int ((p * 37 mod 1009) - 500)) in
  (a, b, c)

let int_args t =
  let a, b, c = int_inputs t in
  let d = i32_vec (Array.make 256 0l) in
  (Execute.[Vec (u8_vec a); Vec (u8_vec b); Vec (i32_vec c); Vec d], d)

let run_int_tile ~gpu ir t =
  let args, d = int_args t in
  let one = Execute.dims1d 1 and wave = Execute.dims1d 32 in
  (match gpu with
  | Some dev -> run_gpu dev ir args ~block:wave ~grid:one
  | None -> run_interp ir args ~block:wave ~grid:one) ;
  Vector.to_array d

let check_int_config dev cfg =
  let ir = tile_ir cfg and control = tile_ir ~drop_c:true cfg in
  let diverged = ref 0 and rejected = ref 0 in
  for t = 0 to 15 do
    let g = run_int_tile ~gpu:(Some dev) ir t in
    if g <> run_int_tile ~gpu:None ir t then incr diverged ;
    if g <> run_int_tile ~gpu:None control t then incr rejected
  done ;
  if !diverged > 0 then
    fail
      "%s: %d of 16 tiles differ from the interpreter"
      (C.config_name cfg)
      !diverged
  else if !rejected <> 16 then
    fail
      "%s: the C-dropping control was accepted on %d tiles"
      (C.config_name cfg)
      (16 - !rejected)
  else ok "%s: 65536 pairs bit-identical, control rejected" (C.config_name cfg)

let check_integer dev =
  List.iter
    (fun (a, b) ->
      List.iter
        (fun acc -> check_int_config dev (config a b acc))
        [C.Sint32; C.Uint32])
    [
      (C.Uint8, C.Uint8);
      (C.Uint8, C.Sint8);
      (C.Sint8, C.Uint8);
      (C.Sint8, C.Sint8);
    ]

(* ---------------------------------------------------------------- 4 *)

(** Run the GEMM kernel ([tile_ir] when [tiling] is [None] and the shape is one
    tile) on [gpu] or the interpreter; return [d]. *)
let run_gemm ?tiling ~gpu ~a ~b ~c ~m ~n ~k () =
  let va = f16_vec a and vb = f16_vec b and vc = f32_vec c in
  let vd = f32_vec (Array.make (m * n) 0.0) in
  let t = Option.value tiling ~default:Gemm.default_tiling in
  (match gpu with
  | Some dev ->
      Gemm.run ~device:dev ~tiling:t ~a:va ~b:vb ~c:vc ~d:vd ~m ~n ~k () ;
      Transfer.flush dev
  | None ->
      Gemm.check_shape t ~m ~n ~k ;
      let block, grid = Gemm.launch_dims t ~m ~n in
      let int x = Execute.Int32 (Int32.of_int x) in
      run_interp
        (Gemm.ir ~tiling:t ())
        Execute.[Vec va; Vec vb; Vec vc; Vec vd; int m; int n; int k]
        ~block
        ~grid) ;
  to_floats vd

let check_bounded ~what ~exact ~absterms ~adds got =
  match bound_check ~adds ~exact ~absterms got with
  | 0, worst ->
      ok
        "%s: within gamma_%d of the exact result (worst |err|/bound %.3g)"
        what
        adds
        worst
  | bad, _ ->
      fail "%s: %d of %d outside gamma_%d" what bad (Array.length got) adds

let check_rejected ~what ~exact ~absterms ~adds got =
  match bound_check ~adds ~exact ~absterms got with
  | 0, _ -> fail "control %s was ACCEPTED by the gate" what
  | bad, _ ->
      ok "control %s rejected (%d of %d outside)" what bad (Array.length got)

(** The result on [dev] is bounded, repeatable, and the same under three tilings
    (each output is the same chain of multiply-adds whatever wave computes it,
    §1.4's dispatch-shape variation). *)
let check_f16_device dev ~what ~a ~b ~c ~m ~n ~k ~exact ~absterms =
  let gpu = run_gemm ~gpu:(Some dev) ~a ~b ~c ~m ~n ~k () in
  check_bounded ~what:(what ^ " gpu") ~exact ~absterms ~adds:k gpu ;
  if run_gemm ~gpu:(Some dev) ~a ~b ~c ~m ~n ~k () <> gpu then
    fail "%s: two runs differ" what
  else ok "%s: two runs bit-identical" what ;
  let differing =
    List.filter
      (fun tiling ->
        run_gemm ~tiling ~gpu:(Some dev) ~a ~b ~c ~m ~n ~k () <> gpu)
      Gemm.
        [
          {frags_m = 1; frags_n = 1; waves = 1};
          {frags_m = 2; frags_n = 1; waves = 2};
        ]
  in
  if differing = [] then ok "%s: three tilings bit-identical" what
  else fail "%s: %d tiling(s) give different bits" what (List.length differing) ;
  gpu

(** The two §5.4 controls must be rejected by the same gate. *)
let check_f16_controls dev ~a ~b ~c ~m ~n ~k ~exact ~absterms =
  check_rejected
    ~what:"f16-accumulating"
    ~exact
    ~absterms
    ~adds:k
    (f16_accumulating ~a ~b ~c ~m ~n ~k) ;
  let c0 = Array.make (m * n) 0.0 in
  check_rejected
    ~what:"C-dropping"
    ~exact
    ~absterms
    ~adds:k
    (run_gemm ~gpu:(Some dev) ~a ~b ~c:c0 ~m ~n ~k ())

let check_f16 dev ~m ~n ~k ~interp =
  let a = operand (m * k) and b = operand (k * n) and c = addend (m * n) in
  assert_exact_in_binary64 ~a ~b ~c ~k ;
  let exact, absterms = exact_gemm ~a ~b ~c ~m ~n ~k in
  let what = Printf.sprintf "%dx%dx%d" m n k in
  let gpu = check_f16_device dev ~what ~a ~b ~c ~m ~n ~k ~exact ~absterms in
  check_f16_controls dev ~a ~b ~c ~m ~n ~k ~exact ~absterms ;
  if interp then begin
    let ref_ = run_gemm ~gpu:None ~a ~b ~c ~m ~n ~k () in
    check_bounded ~what:(what ^ " interpreter") ~exact ~absterms ~adds:k ref_ ;
    (* Information, not a gate: both are admissible, neither must equal the
       other. Records whether the matrix unit happens to round like the
       interpreter's sequential binary32 order. *)
    let same = ref 0 in
    Array.iteri (fun i x -> if x = ref_.(i) then incr same) gpu ;
    Printf.printf
      "  info %s: gpu equals the interpreter's sequential order on %d of %d\n%!"
      what
      !same
      (Array.length gpu)
  end

let () =
  let devices = Array.to_list (Device.init ()) in
  List.iter
    (fun d ->
      Printf.printf "device: %s [%s]\n%!" d.Device.name d.Device.framework)
    devices ;
  check_codegen () ;
  check_f16_refusals devices ;
  check_int_gate devices ;
  (match List.filter (fun d -> is_hip d && offers d int_cfg) devices with
  | [] ->
      print_endline "[SKIP] no gfx11 HIP device: the device checks did not run"
  | gfx11 ->
      List.iter check_block_refusal gfx11 ;
      List.iter check_integer gfx11 ;
      C.accept_relaxed_accumulation
        ~reason:"test_coopmat_hip_e2e: Regime B is what is under test" ;
      List.iter
        (fun dev ->
          check_f16 dev ~m:64 ~n:64 ~k:128 ~interp:true ;
          check_f16 dev ~m:256 ~n:256 ~k:3072 ~interp:false)
        gfx11) ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
