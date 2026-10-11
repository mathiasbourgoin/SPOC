(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The Metal cooperative-matrix probe (specs/coopmat-metal-simdgroup.md). *)

module C = Sarek_coopmat

let probe_source = ""

let support ~simd_width ~(probe : (unit, string) result) =
  ignore probe ;
  {
    C.ds_configs = [];
    ds_robust_buffer_access = false;
    ds_subgroup_size = simd_width;
    ds_advertised_count = 0;
    ds_evidence = "Metal: cooperative matrices are not probed";
  }
