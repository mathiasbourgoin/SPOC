(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The header rewrite [load_from_ptx] applies before [cuModuleLoadData].

    [.target] is rewritten to the device SM, so [.version] must be at least the
    PTX ISA that admits that SM: PTX ISA 8.0 does not know sm_120 (issue #506).
    Pure string tests, no device and no libcuda. *)

open Alcotest

let generated =
  ".version 8.0\n\
   .target sm_86\n\
   .address_size 64\n\
   .visible .entry k() { ret; }\n"

let header ptx =
  String.split_on_char '\n' ptx
  |> List.filter (fun l -> String.length l > 0 && (l.[1] = 'v' || l.[1] = 't'))

let rewrite ~major ~minor ptx = Cuda_api.Kernel.with_sm_target ~major ~minor ptx

let test_blackwell () =
  check
    (list string)
    "sm_120 needs ISA 8.7"
    [".version 8.7"; ".target sm_120"]
    (header (rewrite ~major:12 ~minor:0 generated))

let test_pascal () =
  check
    (list string)
    "sm_61 keeps the generated 8.0, never lowered"
    [".version 8.0"; ".target sm_61"]
    (header (rewrite ~major:6 ~minor:1 generated))

let () =
  run
    "cuda_ptx_version"
    [
      ( "with_sm_target",
        [
          test_case "sm_120 raises .version" `Quick test_blackwell;
          test_case "sm_61 keeps .version" `Quick test_pascal;
        ] );
    ]
