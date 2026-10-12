(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Which backends run epilogue fusion (specs/epilogue-fusion.md, FR-104):
 * without any grant, HIP (every kernel compiled with -ffp-contract=off) and
 * Metal (every kernel starts with #pragma METAL fp contract(off)). On CUDA,
 * OpenCL and Vulkan the fused kernel can differ from the unfused pair by one
 * contraction per element, so fusion there needs the
 * [Cross_kernel_contraction] grant of every kernel involved
 * ([Sarek_policy.decide]); Native, Interpreter and WebGPU never fuse.
 * [set_backend_enabled] only switches a backend off, and back.
 *
 * Kernels are named per case and granted per kernel: the grant store is
 * one-way and process-wide.
 ******************************************************************************)

module LE = Sarek_execute.Launch_epilogue

let decision =
  Alcotest.testable
    (fun ppf -> function
      | Sarek_policy.Applied _ -> Format.pp_print_string ppf "applied"
      | Sarek_policy.Refused why -> Format.fprintf ppf "refused: %s" why)
    (fun a b ->
      match (a, b) with
      | Sarek_policy.Applied _, Sarek_policy.Applied _ -> true
      | Sarek_policy.Refused _, Sarek_policy.Refused _ -> true
      | _ -> false)

let applied = Sarek_policy.Applied Sarek_policy.Exact_by_construction

let refused = Sarek_policy.Refused ""

let pair backend ~p ~c = LE.decide_pair ~backend ~producer:p ~consumer:c

let test_exact_backends () =
  List.iter
    (fun b ->
      Alcotest.check
        decision
        (b ^ " fuses without a grant")
        applied
        (pair b ~p:"def_p" ~c:"def_c"))
    ["HIP"; "Metal"]

let granted_backends = ["CUDA/PTX"; "CUDA/C"; "OpenCL"; "Vulkan"]

let test_grant_backends_need_the_grant () =
  List.iter
    (fun b ->
      LE.set_backend_enabled ~backend:b true ;
      Alcotest.check
        decision
        (b ^ " does not fuse without a grant, even switched on")
        refused
        (pair b ~p:"def_p" ~c:"def_c"))
    granted_backends

let test_never_backends () =
  List.iter
    (fun b ->
      Alcotest.check
        decision
        (b ^ " never fuses")
        refused
        (pair b ~p:"def_p" ~c:"def_c"))
    ["WebGPU"; "Native"; "Interpreter"]

let grant name =
  match
    Sarek_policy.grant
      (Sarek_policy.Kernel name)
      (Sarek_policy.Deviation Sarek_policy.Cross_kernel_contraction)
      ~reason:"test"
  with
  | Ok () -> ()
  | Error _ -> Alcotest.fail "grant refused"

let test_grant_enables () =
  grant "g_p" ;
  Alcotest.check
    decision
    "producer only"
    refused
    (pair "OpenCL" ~p:"g_p" ~c:"g_c") ;
  grant "g_c" ;
  List.iter
    (fun b ->
      Alcotest.check
        decision
        (b ^ " fuses with both kernels granted")
        applied
        (pair b ~p:"g_p" ~c:"g_c"))
    granted_backends ;
  LE.set_backend_enabled ~backend:"OpenCL" false ;
  Alcotest.check
    decision
    "switched off wins over the grant"
    refused
    (pair "OpenCL" ~p:"g_p" ~c:"g_c") ;
  LE.set_backend_enabled ~backend:"OpenCL" true ;
  Alcotest.check
    decision
    "switched back on"
    applied
    (pair "OpenCL" ~p:"g_p" ~c:"g_c")

(* [set_backend_enabled true] is no grant, whatever the backend. *)
let test_enabling_is_not_a_grant () =
  LE.set_backend_enabled ~backend:"CUDA/C" true ;
  Alcotest.check decision "CUDA/C" refused (pair "CUDA/C" ~p:"e_p" ~c:"e_c")

(* A backend that lost its exactness guarantee at run time (Metal without the
   conformant compile options) fuses nothing, grant or not; backends not
   marked are unaffected. *)
let test_inexact_backend_off () =
  LE.set_backend_enabled ~backend:"Metal" true ;
  Alcotest.(check bool) "Metal on before the mark" true (LE.enabled "Metal") ;
  Spoc_framework_registry.Fp_exactness.mark_inexact
    ~backend:"Metal"
    ~reason:"test: conformant compile options unavailable" ;
  Alcotest.(check bool) "Metal off once marked" false (LE.enabled "Metal") ;
  grant "m_p" ;
  grant "m_c" ;
  Alcotest.check
    decision
    "Metal refused even with the grant"
    refused
    (pair "Metal" ~p:"m_p" ~c:"m_c") ;
  Alcotest.(check bool) "HIP unaffected" true (LE.enabled "HIP")

let () =
  Alcotest.run
    "epilogue defaults"
    [
      ( "policy (in order)",
        [
          Alcotest.test_case "HIP and Metal" `Quick test_exact_backends;
          Alcotest.test_case
            "CUDA, OpenCL, Vulkan need the grant"
            `Quick
            test_grant_backends_need_the_grant;
          Alcotest.test_case "never backends" `Quick test_never_backends;
          Alcotest.test_case "a grant enables" `Quick test_grant_enables;
          Alcotest.test_case
            "enabling is not a grant"
            `Quick
            test_enabling_is_not_a_grant;
          Alcotest.test_case
            "inexact backend off"
            `Quick
            test_inexact_backend_off;
        ] );
    ]
