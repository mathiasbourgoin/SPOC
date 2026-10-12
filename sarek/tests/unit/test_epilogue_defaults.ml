(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Which backends run epilogue fusion by default (specs/epilogue-fusion.md,
 * FR-104): HIP, whose every kernel is compiled with -ffp-contract=off, and
 * Metal, whose every kernel starts with #pragma METAL fp contract(off). On
 * any other backend the fused kernel can differ from the unfused pair by an
 * ulp (the compiler may share a subexpression of P and C that the unfused C
 * contracts, or forward P's stored value), so fusion there is an opt-in.
 *
 * The probe reads [Launch_epilogue.default_backends], not the live state,
 * which SAREK_LAUNCH_EPILOGUE=0 in the environment would empty.
 ******************************************************************************)

module LE = Sarek_execute.Launch_epilogue

let test_default_list () =
  Alcotest.(check (list string))
    "default backends"
    ["HIP"; "Metal"]
    (List.sort compare LE.default_backends)

(* Opt-in backends: they contract inside kernels today (or are unconstrained). *)
let opt_in = ["CUDA/PTX"; "CUDA/C"; "OpenCL"; "Vulkan"; "WebGPU"; "Native"]

let test_opt_in_not_default () =
  List.iter
    (fun b ->
      Alcotest.(check bool)
        (b ^ " not in the default list")
        false
        (List.mem b LE.default_backends))
    opt_in

let () =
  Alcotest.run
    "epilogue defaults"
    [
      ( "default backends",
        [
          Alcotest.test_case "HIP and Metal" `Quick test_default_list;
          Alcotest.test_case "others opt-in" `Quick test_opt_in_not_default;
        ] );
    ]
