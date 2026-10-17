(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The OpenCL claims about sub-groups, on faked probe results (no device): the
   warp-shuffle feature is granted by the MEASURED lane mapping and never by the
   extension strings, [warp_size] is the measured sub-group size (it was the
   constant 32), and a sub-group kernel is built at the OpenCL C level the device
   implements. *)

open Alcotest
module Probe = Sarek_opencl.Opencl_warp_probe

let measured width ~ok =
  Some {Probe.width; lane_mapping_ok = ok; failing_shape = None}

let test_feature_needs_measured_mapping () =
  check
    bool
    "mapping verified: granted"
    true
    (Probe.features_of_probe (measured 16 ~ok:true)
    = [Sarek_ir_analysis.Warp_shuffle]) ;
  check
    bool
    "mapping broken: refused"
    true
    (Probe.features_of_probe (measured 16 ~ok:false) = []) ;
  check
    bool
    "no sub-group size: refused"
    true
    (Probe.features_of_probe (measured 0 ~ok:true) = []) ;
  check bool "probe failed: refused" true (Probe.features_of_probe None = [])

let test_warp_size_is_measured () =
  List.iter
    (fun w ->
      check
        int
        (Printf.sprintf "width %d" w)
        w
        (Probe.warp_size_of_probe (measured w ~ok:true)))
    [8; 16; 32; 64] ;
  check
    int
    "a width is reported even when the mapping is broken"
    16
    (Probe.warp_size_of_probe (measured 16 ~ok:false)) ;
  check int "no sub-groups: no warp structure" 1 (Probe.warp_size_of_probe None)

let test_source_declares_extensions () =
  let k =
    let open Sarek_ir_types in
    let dst =
      {var_name = "d"; var_id = 0; var_type = TVec TInt32; var_mutable = false}
    in
    {
      default_kernel with
      kern_name = "k";
      kern_params =
        [DParam (dst, Some {arr_elttype = TInt32; arr_memspace = Global})];
      kern_body =
        SAssign
          ( LArrayElem ("d", EConst (CInt32 0l)),
            EIntrinsic
              ( [],
                Sarek_warp.ir_name Sarek_warp.Down Sarek_warp.I32,
                [EConst (CInt32 1l); EIntrinsic ([], "lane_id", [])] ) );
    }
  in
  let src = Sarek_opencl.Sarek_ir_opencl.generate k in
  let has needle =
    let n = String.length needle and m = String.length src in
    let rec go i = i + n <= m && (String.sub src i n = needle || go (i + 1)) in
    go 0
  in
  check bool "shuffle extension" true (has "cl_khr_subgroup_shuffle") ;
  check bool "sub-group extension" true (has "cl_khr_subgroups") ;
  check bool "uses the runtime sub-group size" true (has "get_sub_group_size()") ;
  check bool "no CUDA-style warp built-in" false (has "warpSize")

let () =
  run
    "opencl warp capability"
    [
      ( "capability",
        [
          test_case
            "the feature needs a measured lane mapping"
            `Quick
            test_feature_needs_measured_mapping;
          test_case "warp_size is measured" `Quick test_warp_size_is_measured;
          test_case
            "source declares the extensions"
            `Quick
            test_source_declares_extensions;
        ] );
    ]
