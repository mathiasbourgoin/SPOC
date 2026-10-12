(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A [Kernel]-scoped grant changes the decision for that kernel and for no
   other (specs/optimisation-policy.md, M2): the matrix rewrite's opt-ins
   (accumulation order, f16 operand rounding, automatic float rewrite) and the
   epilogue fusion's cross-kernel contraction. Narrowing and the coopmat
   verdict are covered in codegen_golden and spoc/ir/test. No device, and no
   Process grant (the store is one-way).

   The rewrite is decided on a device that exists only as a record: the
   decision is taken before the device is used further. A kernel that is
   declined at the device (no probed matrix support) was not declined by the
   policy, which is what these cases look at. *)

[@@@warning "-33"]

[@@@alert "-sarek_matrix"]

open Sarek
module Spec = Sarek_execute.Matrix_specialise
module Adv = Sarek_matrix_advice
module LE = Sarek_execute.Launch_epilogue

let contains ~needle s =
  let n = String.length needle in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = needle || at (i + 1))
  in
  at 0

let kernel_grant name g =
  match Sarek_policy.grant (Sarek_policy.Kernel name) g ~reason:"test" with
  | Ok () -> ()
  | Error _ -> Alcotest.fail "grant refused"

let accumulation = Sarek_policy.Deviation Sarek_policy.Accumulation_order

let f16_rounding =
  Sarek_policy.Deviation
    (Sarek_policy.Operand_rounding Sarek_coopmat.Operands_f16)

let automatic = Sarek_policy.Consent Sarek_policy.Automatic_float_rewrite

let lowering name ~explicit case precision =
  Spec.lowering_for
    ~budget:(Sarek_policy.budget ~kernels:[name])
    ~mode:Spec.Auto
    ~explicit
    case
    precision

let is_ok = function Ok _ -> true | Error _ -> false

let test_automatic_rewrite () =
  kernel_grant "mks_auto_a" automatic ;
  Alcotest.(check bool)
    "kernel A, f16"
    true
    (is_ok (lowering "mks_auto_a" ~explicit:false Adv.F16_inputs None)) ;
  Alcotest.(check bool)
    "kernel B, f16"
    false
    (is_ok (lowering "mks_auto_b" ~explicit:false Adv.F16_inputs None))

let test_f16_rounding () =
  kernel_grant "mks_round_a" automatic ;
  kernel_grant "mks_round_b" automatic ;
  kernel_grant "mks_round_a" f16_rounding ;
  Alcotest.(check bool)
    "kernel A, f32 with f16 rounding"
    true
    (is_ok
       (lowering
          "mks_round_a"
          ~explicit:false
          Adv.F32_inputs
          (Some Sarek_coopmat.Operands_f16))) ;
  Alcotest.(check bool)
    "kernel B, f32 with f16 rounding"
    false
    (is_ok
       (lowering
          "mks_round_b"
          ~explicit:false
          Adv.F32_inputs
          (Some Sarek_coopmat.Operands_f16))) ;
  Alcotest.(check bool)
    "kernel A, no precision named: f16 is preferred"
    true
    (is_ok (lowering "mks_round_a" ~explicit:false Adv.F32_inputs None))

let k_f16 =
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
            sum
            +. float32_of_float16 a.((row * k) + i)
               *. float32_of_float16 b.((i * n) + col)
        done ;
        c.((row * n) + col) <- sum
      end]

let fake_hip : Spoc_core.Device.t =
  {
    id = 0;
    backend_id = 0;
    name = "fake HIP";
    framework = "HIP";
    capabilities =
      {
        max_threads_per_block = 256;
        max_block_dims = (256, 256, 64);
        max_grid_dims = (65535, 65535, 65535);
        shared_mem_per_block = 16384;
        total_global_mem = 1073741824L;
        compute_capability = (0, 0);
        device_features = [];
        coopmat = None;
        supports_atomics = false;
        warp_size = 32;
        max_registers_per_block = 0;
        clock_rate_khz = 0;
        multiprocessor_count = 1;
        is_cpu = false;
      };
  }

(* The same product under two names: A holds every opt-in as Kernel grants, B
   holds none. A gets past the policy checks of [choose], B is declined by them. *)
let test_choose () =
  let ir =
    match (snd k_f16).Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> Alcotest.fail "no IR"
  in
  let named n = {ir with Sarek_ir_types.kern_name = n} in
  Spec.set_mode Spec.Auto ;
  List.iter
    (fun g -> kernel_grant "mks_choose_a" g)
    [accumulation; f16_rounding; automatic] ;
  let why n =
    match
      Spec.choose ~device:fake_hip ~ir:(named n) ~launch:(fun () -> None) ()
    with
    | Spec.Original why -> why
    | Spec.Specialised _ -> Alcotest.fail "rewritten without a device"
  in
  let policy_words =
    [
      "accept_automatic_float_rewrite";
      "accept_relaxed_accumulation";
      "accept_operand_rounding";
    ]
  in
  let why_a = why "mks_choose_a" and why_b = why "mks_choose_b" in
  List.iter
    (fun w ->
      Alcotest.(check bool)
        ("kernel A not declined for " ^ w ^ " (" ^ why_a ^ ")")
        false
        (contains ~needle:w why_a))
    policy_words ;
  Alcotest.(check bool)
    ("kernel B declined by the policy (" ^ why_b ^ ")")
    true
    (List.exists (fun w -> contains ~needle:w why_b) policy_words)

let test_epilogue () =
  let contraction =
    Sarek_policy.Deviation Sarek_policy.Cross_kernel_contraction
  in
  let decide p c = LE.decide_pair ~backend:"OpenCL" ~producer:p ~consumer:c in
  let applied = function Sarek_policy.Applied _ -> true | _ -> false in
  kernel_grant "mks_ep_p" contraction ;
  Alcotest.(check bool)
    "producer only"
    false
    (applied (decide "mks_ep_p" "mks_ep_c")) ;
  kernel_grant "mks_ep_c" contraction ;
  Alcotest.(check bool) "both" true (applied (decide "mks_ep_p" "mks_ep_c")) ;
  Alcotest.(check bool)
    "another pair"
    false
    (applied (decide "mks_ep_p" "mks_ep_other"))

let () =
  let case name f = Alcotest.test_case name `Quick f in
  Alcotest.run
    "policy_kernel_scope"
    [
      ( "kernel grants",
        [
          case "automatic float rewrite" test_automatic_rewrite;
          case "f16 operand rounding" test_f16_rounding;
          case "choose" test_choose;
          case "cross-kernel contraction" test_epilogue;
        ] );
    ]
