(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The process half of SAREK_OPT_DISABLE and SAREK_EXPLAIN
   (specs/optimisation-policy.md, M3). The variables are read once, when the
   process starts, and every restriction is one-way: one scenario per process,
   chosen by argv.(1), with the environment set by the dune rule that runs it
   (see dune). No device: the decision points are driven through fake devices,
   the pure code generators and the policy's own entry points.

   Scenarios:
   - baseline: no variable; every default, to compare the others against.
   - window, epilogue, matrix, narrowing-proof, unroll: that optimisation is
     taken away and no other.
   - single-rounding, cross-kernel-contraction, automatic-float-rewrite: the
     grant is revoked; a program that grants it is recorded but not obeyed.
   - unknown: an unknown item warns (stderr is compared with a golden) and the
     rest applies.
   - removed: the five replaced variables are set and change nothing.
   - explain, explain-off, explain-unknown, explain-replaced: SAREK_EXPLAIN. *)

[@@@warning "-33"]

[@@@alert "-sarek_matrix"]

open Sarek
open Sarek_ir_types
module Spec = Sarek_execute.Matrix_specialise
module Lp = Sarek_execute.Launch_policy
module LE = Sarek_execute.Launch_epilogue
module Cm = Sarek_coopmat
module Adv = Sarek_matrix_advice

(* ------------------------------------------------------------------------ *)
(* Fixtures                                                                  *)
(* ------------------------------------------------------------------------ *)

let var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let vec name elt =
  DParam (var name (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})

let i32 n = EConst (CInt32 (Int32.of_int n))

let tid = var "tid" TInt32

let read a = EArrayRead (a, EVar tid)

let widen e = ECast (TFloat32, e)

let to_float e = EIntrinsic (["Sarek_stdlib"; "Gpu"], "float", [e])

let ( *: ) a b = EBinop (Mul, a, b)

let ( +: ) a b = EBinop (Add, a, b)

let ( -: ) a b = EBinop (Sub, a, b)

let ( &: ) a b = EBinop (BitAnd, a, b)

let ( >>: ) a b = EBinop (Shr, a, b)

let store e = SAssign (LArrayElem ("out", EVar tid), ECast (TFloat16, e))

let narrowing_kernel name body =
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
      ];
    kern_body = SLet (tid, EIntrinsic ([], "global_thread_id", []), body);
  }

let h = widen (read "h")

let g = widen (read "g")

(* One rounding provably equals two: no barrier with or without a grant. *)
let t2_f16_sum = narrowing_kernel "t2_f16_sum" (store (h +: g))

(* Not provable: keeps its barrier unless single rounding is granted. *)
let q4k_like =
  let m = to_float (read "q" >>: i32 4 &: i32 63) in
  narrowing_kernel
    "q4k_like"
    (store ((h *: to_float (read "q" &: i32 15)) -: (g *: m)))

(* A constant-trip loop: Sarek_ir_unroll hints it. *)
let unroll_loop =
  let i = var "i" TInt32 in
  {
    default_kernel with
    kern_name = "unroll_loop";
    kern_params = [vec "acc" TFloat32; vec "x" TFloat32];
    kern_body =
      SLet
        ( tid,
          EIntrinsic ([], "global_thread_id", []),
          SFor
            ( i,
              i32 0,
              i32 8,
              Upto,
              SAssign
                ( LArrayElem ("acc", EVar tid),
                  EArrayRead ("acc", EVar tid) +: read "x" ) ) );
  }

(* A float cooperative-matrix multiply-add: the explicit coopmat gate. *)
let coopmat_kernel =
  let cfg =
    {
      Cm.cfg_shape = {Cm.m = 16; n = 16; k = 16};
      cfg_a = Cm.Float16;
      cfg_b = Cm.Float16;
      cfg_c = Cm.Float32;
      cfg_result = Cm.Float32;
      cfg_saturating = false;
      cfg_scope = Cm.Subgroup;
    }
  in
  {
    default_kernel with
    kern_name = "coopmat_f16";
    kern_body =
      SCoopmat (CM_muladd {dst = "fc"; a = "fa"; b = "fb"; c = "fc"; cfg});
  }

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

let gemm_ir =
  match (snd k_f16).Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

let fake_device ~framework ~is_cpu : Spoc_core.Device.t =
  {
    id = 0;
    backend_id = 0;
    name = "fake " ^ framework;
    framework;
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
        is_cpu;
      };
  }

let fake_hip = fake_device ~framework:"HIP" ~is_cpu:false

let fake_native = fake_device ~framework:"Native" ~is_cpu:true

(* ------------------------------------------------------------------------ *)
(* Helpers                                                                   *)
(* ------------------------------------------------------------------------ *)

let cuda_c k = Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:[] k

let opencl k = Sarek_codegen.Sarek_ir_opencl.generate_with_types ~types:[] k

let metal k = Sarek_codegen.Sarek_ir_metal.generate_with_types ~types:[] k

let glsl k = Sarek_codegen.Sarek_ir_glsl.generate_with_types ~types:[] k

let count ~needle s =
  let n = String.length needle in
  let rec go i acc =
    if i + n > String.length s then acc
    else if String.sub s i n = needle then go (i + n) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

let contains ~needle s = count ~needle s > 0

let barriers k = count ~needle:"__float2half(sarek_f32_barrier(" (cuda_c k)

let unroll_hints () =
  List.map
    (fun gen -> count ~needle:"#pragma unroll\n" (gen unroll_loop))
    [cuda_c; opencl; metal]
  @ [count ~needle:"[[unroll]]" (glsl unroll_loop)]

let deferred ?(streamed = false) ?(soa_params = []) ?(kernel = "env_kernel")
    device =
  Lp.deferred ~kernel ~device ~streamed ~soa_params

let epilogue backend =
  LE.decide_pair ~backend ~producer:"env_p" ~consumer:"env_c"

let decision_text = function
  | Sarek_policy.Applied Sarek_policy.Exact_by_construction -> "exact"
  | Sarek_policy.Applied (Sarek_policy.Granted {reason; _}) ->
      "granted: " ^ reason
  | Sarek_policy.Applied (Sarek_policy.Proof p) -> "proof: " ^ p
  | Sarek_policy.Refused why -> "refused: " ^ why

let disabled_text = "refused: disabled (SAREK_OPT_DISABLE)"

let check_restricted expected =
  List.iter
    (fun o ->
      Alcotest.(check bool)
        (Sarek_policy.optimisation_name o)
        (List.mem o expected)
        (Sarek_policy.restricted o))
    Sarek_policy.all_optimisations

let ok_or_fail = function
  | Ok () -> ()
  | Error r -> Alcotest.fail (Sarek_policy.refusal_message r)

let single = Sarek_policy.Deviation Sarek_policy.Single_rounding

let contraction = Sarek_policy.Deviation Sarek_policy.Cross_kernel_contraction

let automatic = Sarek_policy.Consent Sarek_policy.Automatic_float_rewrite

let lowering ?precision ~explicit case =
  Spec.lowering_for
    ~budget:(Sarek_policy.budget ~kernels:[])
    ~explicit
    case
    precision

let is_ok = function Ok _ -> true | Error _ -> false

(* ------------------------------------------------------------------------ *)
(* Scenarios                                                                 *)
(* ------------------------------------------------------------------------ *)

let test_baseline () =
  check_restricted [] ;
  Alcotest.(check bool) "window defers on HIP" true (deferred fake_hip) ;
  Alcotest.(check string)
    "epilogue on HIP"
    "exact"
    (decision_text (epilogue "HIP")) ;
  Alcotest.(check (list int)) "unroll hints" [1; 1; 1; 1] (unroll_hints ()) ;
  Alcotest.(check int) "provable narrowing: no barrier" 0 (barriers t2_f16_sum) ;
  Alcotest.(check bool)
    "unprovable narrowing: barrier"
    true
    (barriers q4k_like > 0) ;
  Alcotest.(check bool)
    "int32 is lowered"
    true
    (is_ok (lowering ~explicit:false Adv.Int32_inputs))

let test_window () =
  check_restricted [Sarek_policy.Launch_window] ;
  Alcotest.(check bool) "no deferral on HIP" false (deferred fake_hip) ;
  Alcotest.(check string)
    "epilogue unaffected"
    "exact"
    (decision_text (epilogue "HIP")) ;
  Alcotest.(check (list int)) "unroll unaffected" [1; 1; 1; 1] (unroll_hints ())

let test_unroll () =
  check_restricted [Sarek_policy.Auto_unroll] ;
  Alcotest.(check (list int))
    "no unroll pragma in any generator"
    [0; 0; 0; 0]
    (unroll_hints ()) ;
  Alcotest.(check bool) "window unaffected" true (deferred fake_hip) ;
  Alcotest.(check int) "narrowing unaffected" 0 (barriers t2_f16_sum)

let test_narrowing_proof () =
  check_restricted [Sarek_policy.Narrowing_proof] ;
  Alcotest.(check bool)
    "barriers kept on a provable kernel"
    true
    (barriers t2_f16_sum > 0) ;
  Alcotest.(check (list int)) "unroll unaffected" [1; 1; 1; 1] (unroll_hints ())

let test_matrix () =
  check_restricted [Sarek_policy.Matrix_rewrite] ;
  (match
     Spec.choose ~device:fake_hip ~ir:gemm_ir ~launch:(fun () -> None) ()
   with
  | Spec.Original why ->
      Alcotest.(check string)
        "nothing rewritten"
        "disabled (SAREK_OPT_DISABLE)"
        why
  | Spec.Specialised _ -> Alcotest.fail "rewritten") ;
  Alcotest.(check bool) "window unaffected" true (deferred fake_hip)

let test_epilogue () =
  check_restricted [Sarek_policy.Epilogue_fusion] ;
  Alcotest.(check string)
    "no epilogue fusion on HIP"
    disabled_text
    (decision_text (epilogue "HIP")) ;
  Alcotest.(check bool) "window unaffected" true (deferred fake_hip)

(* The program grants single rounding for q4k_like; the process revokes it. *)
let test_single_rounding () =
  check_restricted [] ;
  ok_or_fail
    (Sarek_f16_policy.accept_single_rounding
       (Sarek_f16_policy.Kernel "q4k_like")
       ~reason:"the program accepts it") ;
  Alcotest.(check bool) "revoked" true (Sarek_policy.revoked single) ;
  Alcotest.(check (option string))
    "the reader sees no grant"
    None
    (Sarek_policy.reason ~kernel:"q4k_like" single) ;
  Alcotest.(check bool)
    "the budget holds none"
    false
    (Sarek_policy.Budget.permits
       (Sarek_policy.budget ~kernels:["q4k_like"])
       single) ;
  Alcotest.(check bool)
    "barriers kept: the grant has no effect"
    true
    (barriers q4k_like > 0) ;
  Alcotest.(check int) "provable kernel unchanged" 0 (barriers t2_f16_sum) ;
  let events = ref [] in
  Sarek_policy.set_reporter (Some (fun e -> events := e :: !events)) ;
  ignore (cuda_c q4k_like) ;
  Sarek_policy.set_reporter None ;
  let lines = List.rev_map Sarek_policy.Env.explain_line !events in
  Alcotest.(check bool)
    "explain says the grant is revoked"
    true
    (List.mem
       "sarek-explain q4k_like - policy refused grant-revoked single-rounding"
       lines)

let test_contraction () =
  ok_or_fail
    (Sarek_policy.grant
       Sarek_policy.Process
       contraction
       ~reason:"the program accepts one contraction") ;
  Alcotest.(check bool) "revoked" true (Sarek_policy.revoked contraction) ;
  Alcotest.(check string)
    "no epilogue on OpenCL despite the grant"
    "refused: grant cross-kernel-contraction revoked (SAREK_OPT_DISABLE)"
    (decision_text (epilogue "OpenCL")) ;
  Alcotest.(check string)
    "no epilogue on Vulkan either"
    "refused: grant cross-kernel-contraction revoked (SAREK_OPT_DISABLE)"
    (decision_text (epilogue "Vulkan")) ;
  Alcotest.(check string)
    "HIP needs no grant"
    "exact"
    (decision_text (epilogue "HIP"))

let test_automatic_float_rewrite () =
  Cm.accept_relaxed_accumulation ~reason:"env scenario" ;
  Cm.accept_operand_rounding ~precision:Cm.Operands_f16 ~reason:"env scenario" ;
  Cm.accept_automatic_float_rewrite ~reason:"env scenario" ;
  Alcotest.(check bool) "revoked" true (Sarek_policy.revoked automatic) ;
  (match lowering ~explicit:false Adv.F16_inputs with
  | Ok _ -> Alcotest.fail "f16 rewritten automatically despite the revocation"
  | Error why ->
      Alcotest.(check bool)
        ("the reason names the variable (" ^ why ^ ")")
        true
        (contains ~needle:"SAREK_OPT_DISABLE" why)) ;
  Alcotest.(check bool)
    "f32 not rewritten automatically either"
    false
    (is_ok (lowering ~explicit:false ~precision:Cm.Operands_f16 Adv.F32_inputs)) ;
  Alcotest.(check bool)
    "int32 is still rewritten"
    true
    (is_ok (lowering ~explicit:false Adv.Int32_inputs)) ;
  Alcotest.(check bool)
    "an explicit request needs no consent"
    true
    (is_ok (lowering ~explicit:true Adv.F16_inputs)) ;
  Alcotest.(check bool)
    "the accuracy opt-ins are untouched"
    true
    (Cm.relaxed_accumulation_reason () <> None)

let test_unknown () =
  check_restricted [Sarek_policy.Launch_window; Sarek_policy.Auto_unroll] ;
  Alcotest.(check bool) "window off" false (deferred fake_hip) ;
  Alcotest.(check (list int)) "unroll off" [0; 0; 0; 0] (unroll_hints ()) ;
  Alcotest.(check string) "epilogue on" "exact" (decision_text (epilogue "HIP"))

(* The five replaced variables are set (dune): none of them does anything. *)
let test_removed () =
  check_restricted [] ;
  Alcotest.(check bool) "window defers" true (deferred fake_hip) ;
  Alcotest.(check string) "epilogue on" "exact" (decision_text (epilogue "HIP")) ;
  Alcotest.(check (list int)) "unroll hints" [1; 1; 1; 1] (unroll_hints ()) ;
  Cm.accept_relaxed_accumulation ~reason:"env scenario" ;
  Cm.accept_operand_rounding ~precision:Cm.Operands_f16 ~reason:"env scenario" ;
  Cm.accept_automatic_float_rewrite ~reason:"env scenario" ;
  Alcotest.(check bool)
    "float rewrite allowed (the removed mode variable is ignored)"
    true
    (is_ok (lowering ~explicit:false Adv.F16_inputs)) ;
  match
    Spec.choose ~device:fake_hip ~ir:gemm_ir ~launch:(fun () -> None) ()
  with
  | Spec.Original why ->
      Alcotest.(check bool)
        ("not switched off (" ^ why ^ ")")
        false
        (contains ~needle:"switched off" why
        || contains ~needle:"SAREK_OPT_DISABLE" why)
  | Spec.Specialised _ -> Alcotest.fail "rewritten without a device"

(* ------------------------------------------------------------------------ *)
(* SAREK_EXPLAIN                                                             *)
(* ------------------------------------------------------------------------ *)

let test_explain_off () =
  Alcotest.(check bool) "no reporter" false (Sarek_policy.reporting ()) ;
  ignore (cuda_c unroll_loop) ;
  ignore (deferred fake_hip)

let test_explain_replaced () =
  Alcotest.(check bool)
    "the default reporter is installed"
    true
    (Sarek_policy.reporting ()) ;
  let seen = ref 0 in
  Sarek_policy.set_reporter (Some (fun _ -> incr seen)) ;
  ignore (cuda_c unroll_loop) ;
  Alcotest.(check int) "the program's reporter gets the record" 1 !seen

(* Every decision point that needs no device, once; the lines go to stderr and
   are compared with explain.expected. *)
let test_explain_golden () =
  Alcotest.(check bool) "reporting" true (Sarek_policy.reporting ()) ;
  (* codegen: unroll per generator, narrowing per kernel *)
  List.iter
    (fun gen -> ignore (gen unroll_loop))
    [cuda_c; cuda_c; opencl; metal; glsl] ;
  ignore (cuda_c t2_f16_sum) ;
  ignore (cuda_c q4k_like) ;
  ok_or_fail
    (Sarek_f16_policy.accept_single_rounding
       (Sarek_f16_policy.Kernel "q4k_like")
       ~reason:"profile fast") ;
  ignore (cuda_c q4k_like) ;
  (* the launch window: deferred, on a CPU device, on a stream *)
  ignore (deferred ~kernel:"env_gemm" fake_hip) ;
  ignore (deferred ~kernel:"env_gemm" fake_hip) ;
  ignore (deferred ~kernel:"env_gemm" fake_native) ;
  ignore (deferred ~kernel:"env_gemm" ~streamed:true fake_hip) ;
  ignore (deferred ~kernel:"env_gemm" ~soa_params:["v"] fake_hip) ;
  (* the matrix rewrite: declined for the missing consent *)
  ignore (Spec.choose ~device:fake_hip ~ir:gemm_ir ~launch:(fun () -> None) ()) ;
  (* the explicit coopmat gate: refused, then granted *)
  Lp.coopmat_gate ~device:fake_hip coopmat_kernel ;
  Cm.accept_relaxed_accumulation ~reason:"profile fast" ;
  Lp.coopmat_gate ~device:fake_hip coopmat_kernel ;
  (* a grant on a generated name never counts, and the report says so *)
  ok_or_fail
    (Sarek_policy.grant
       (Sarek_policy.Kernel "env_fused_1")
       single
       ~reason:"on a generated name") ;
  Sarek_policy.register_fused ~name:"env_fused_1" ~members:["env_gemm"] ;
  ignore (deferred ~kernel:"env_fused_1" fake_hip)

(* ------------------------------------------------------------------------ *)

let scenarios =
  let case name f = Alcotest.test_case name `Quick f in
  [
    ("baseline", [case "defaults" test_baseline]);
    ("window", [case "window" test_window]);
    ("epilogue", [case "epilogue" test_epilogue]);
    ("matrix", [case "matrix" test_matrix]);
    ("narrowing-proof", [case "narrowing-proof" test_narrowing_proof]);
    ("unroll", [case "unroll" test_unroll]);
    ("single-rounding", [case "single-rounding revoked" test_single_rounding]);
    ( "cross-kernel-contraction",
      [case "cross-kernel-contraction revoked" test_contraction] );
    ( "automatic-float-rewrite",
      [case "automatic-float-rewrite revoked" test_automatic_float_rewrite] );
    ("unknown", [case "unknown item" test_unknown]);
    ("removed", [case "replaced variables" test_removed]);
    ("explain", [case "golden" test_explain_golden]);
    ("explain-off", [case "off" test_explain_off]);
    ("explain-unknown", [case "off" test_explain_off]);
    ("explain-replaced", [case "replaced" test_explain_replaced]);
  ]

let () =
  match Array.to_list Sys.argv with
  | [_; scenario] when List.mem_assoc scenario scenarios ->
      (* Verbose: Alcotest otherwise captures stderr, which the golden
         scenarios compare. *)
      Alcotest.run
        ~verbose:true
        ~argv:[|Sys.argv.(0)|]
        "policy_env"
        [(scenario, List.assoc scenario scenarios)]
  | _ ->
      prerr_endline "usage: test_policy_env <scenario>" ;
      exit 2
