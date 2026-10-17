(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Capability gate and launch rule of the warp operations
 * (docs/plans/warp-shuffle-intake-2026-10-17.md, AC-6 and AC-7), on fake
 * devices: no hardware.
 *
 *  - a device without the Warp_shuffle feature REFUSES a kernel that uses a
 *    warp operation, with a located message naming the feature and the device;
 *    with the feature, the same kernel launches;
 *  - a kernel that shuffles needs a block that is a whole number of warps, on
 *    the width the device reports (Native and the interpreter: the simulated
 *    width); a kernel without a shuffle is unaffected.
 ******************************************************************************)

open Spoc_framework
open Alcotest
module Exec = Sarek.Execute

let contains_sub haystack needle =
  let hl = String.length haystack and nl = String.length needle in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  go 0

let caps ?(warp_size = 32) features : Framework_sig.capabilities =
  {
    Framework_sig.max_threads_per_block = 1024;
    max_block_dims = (1024, 1024, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    device_features = features;
    coopmat = None;
    supports_atomics = true;
    warp_size;
    max_registers_per_block = 16384;
    clock_rate_khz = 1000000;
    multiprocessor_count = 4;
    is_cpu = false;
  }

let device ?(framework = "OpenCL") ?warp_size features : Spoc_core.Device.t =
  {
    id = 0;
    backend_id = 0;
    name = "Fake Warp Device";
    framework;
    capabilities = caps ?warp_size features;
  }

open Sarek_ir_types

let kernel_with body =
  let dst =
    {var_name = "dst"; var_id = 0; var_type = TVec TInt32; var_mutable = false}
  in
  {
    default_kernel with
    kern_name = "warp_cap_probe";
    kern_params =
      [DParam (dst, Some {arr_elttype = TInt32; arr_memspace = Global})];
    kern_body = body;
  }

let store e = SAssign (LArrayElem ("dst", EConst (CInt32 0l)), e)

let shuffle_kernel =
  kernel_with
    (store
       (EIntrinsic
          ( [],
            Sarek_warp.ir_name Sarek_warp.Xor Sarek_warp.I32,
            [EConst (CInt32 1l); EConst (CInt32 1l)] )))

let lane_id_kernel = kernel_with (store (EIntrinsic ([], "lane_id", [])))

let plain_kernel = kernel_with (store (EConst (CInt32 7l)))

let refusal ~device ir =
  match Exec.check_device_capabilities ~device ir with
  | () -> None
  | exception e -> Some (Printexc.to_string e)

let test_refused_without_feature () =
  List.iter
    (fun (label, ir) ->
      match refusal ~device:(device []) ir with
      | None -> failf "%s must be refused on a device without the feature" label
      | Some msg ->
          check
            bool
            (label ^ ": names the feature")
            true
            (contains_sub msg "warp-shuffle") ;
          check
            bool
            (label ^ ": names the device")
            true
            (contains_sub msg "Fake Warp Device"))
    [("a shuffle kernel", shuffle_kernel); ("a lane_id kernel", lane_id_kernel)]

let test_allowed_with_feature () =
  let device = device [Sarek_ir_analysis.Warp_shuffle] in
  List.iter
    (fun ir ->
      match refusal ~device ir with
      | None -> ()
      | Some msg -> failf "must launch with the feature, got: %s" msg)
    [shuffle_kernel; lane_id_kernel]

let test_plain_kernel_unaffected () =
  match refusal ~device:(device []) plain_kernel with
  | None -> ()
  | Some msg -> failf "a kernel without a warp operation is never gated: %s" msg

(* Every claim of the feature is by backend: Native and the Interpreter list it. *)
let test_host_backends_list_the_feature () =
  check
    bool
    "interpreter"
    true
    (List.mem
       Sarek_ir_analysis.Warp_shuffle
       Sarek_interp.Sarek_interp_capability.device_features)

let block n = Framework_sig.{x = n; y = 1; z = 1}

let launch_refusal ~device ir ~block:b =
  match Exec.check_launch_warp ~device ~kernel:"k" ir ~block:b with
  | () -> None
  | exception
      Sarek.Execute_error.Execution_error (Invalid_dimensions {reason; _}) ->
      Some reason

let test_block_multiple_of_warp () =
  let d32 = device ~warp_size:32 [Sarek_ir_analysis.Warp_shuffle] in
  let d64 = device ~warp_size:64 [Sarek_ir_analysis.Warp_shuffle] in
  check
    bool
    "128 on W=32"
    true
    (launch_refusal ~device:d32 shuffle_kernel ~block:(block 128) = None) ;
  check
    bool
    "128 on W=64"
    true
    (launch_refusal ~device:d64 shuffle_kernel ~block:(block 128) = None) ;
  (match launch_refusal ~device:d32 shuffle_kernel ~block:(block 100) with
  | None -> fail "a block of 100 on W=32 must be refused"
  | Some reason ->
      check bool "names the width" true (contains_sub reason "warp width 32") ;
      check bool "names the block size" true (contains_sub reason "100 threads") ;
      check bool "names the kernel" true (contains_sub reason "\"k\"")) ;
  check
    bool
    "96 on W=64 is refused"
    true
    (launch_refusal ~device:d64 shuffle_kernel ~block:(block 96) <> None) ;
  (* 2-D and 3-D blocks count their total size *)
  check
    bool
    "(8, 4, 1) = 32 on W=32"
    true
    (launch_refusal
       ~device:d32
       shuffle_kernel
       ~block:Framework_sig.{x = 8; y = 4; z = 1}
    = None) ;
  check
    bool
    "(8, 5, 1) = 40 on W=32 is refused"
    true
    (launch_refusal
       ~device:d32
       shuffle_kernel
       ~block:Framework_sig.{x = 8; y = 5; z = 1}
    <> None)

let test_unaffected_without_shuffle () =
  let d32 = device ~warp_size:32 [Sarek_ir_analysis.Warp_shuffle] in
  check
    bool
    "a kernel without a shuffle accepts any block"
    true
    (launch_refusal ~device:d32 plain_kernel ~block:(block 100) = None) ;
  (* lane_id and warp_size read the warp but are not collective *)
  check
    bool
    "lane_id alone accepts any block"
    true
    (launch_refusal ~device:d32 lane_id_kernel ~block:(block 100) = None)

(* Native and the interpreter simulate a warp whose width is read at launch. *)
let test_simulated_width () =
  let native = device ~framework:"Native" ~warp_size:1 [] in
  check
    bool
    "default 32: 128 ok"
    true
    (launch_refusal ~device:native shuffle_kernel ~block:(block 128) = None) ;
  check
    bool
    "default 32: 100 refused"
    true
    (launch_refusal ~device:native shuffle_kernel ~block:(block 100) <> None) ;
  Sarek_warp.with_width 8 (fun () ->
      check
        bool
        "W=8: 40 ok"
        true
        (launch_refusal ~device:native shuffle_kernel ~block:(block 40) = None)) ;
  Sarek_warp.with_width 64 (fun () ->
      check
        bool
        "W=64: 96 refused"
        true
        (launch_refusal ~device:native shuffle_kernel ~block:(block 96) <> None)) ;
  check int "restored" 32 (Sarek_warp.simulated_width ())

let test_with_width_rejects_non_powers () =
  List.iter
    (fun w ->
      match Sarek_warp.with_width w (fun () -> ()) with
      | () -> failf "width %d must be rejected" w
      | exception Invalid_argument _ -> ())
    [0; 3; 48; -2; 2048]

(* The textual refusal of a backend that cannot spell the feature: Metal has no
   float64, and its helpers cannot read the lane. *)
let test_metal_refusals () =
  let f64_kernel =
    let dst =
      {
        var_name = "d";
        var_id = 0;
        var_type = TVec TFloat64;
        var_mutable = false;
      }
    in
    {
      default_kernel with
      kern_name = "f64";
      kern_params =
        [DParam (dst, Some {arr_elttype = TFloat64; arr_memspace = Global})];
      kern_body =
        SAssign
          ( LArrayElem ("d", EConst (CInt32 0l)),
            EIntrinsic
              ( [],
                Sarek_warp.ir_name Sarek_warp.Idx Sarek_warp.F64,
                [EConst (CFloat64 1.0); EConst (CInt32 0l)] ) );
    }
  in
  (match Sarek_codegen.Sarek_ir_metal.generate f64_kernel with
  | (_ : string) -> fail "Metal must refuse a float64 shuffle"
  | exception _ -> ()) ;
  let in_helper =
    let x =
      {var_name = "x"; var_id = 1; var_type = TInt32; var_mutable = false}
    in
    let helper =
      {
        hf_name = "h";
        hf_params = [x];
        hf_ret_type = TInt32;
        hf_body =
          SReturn
            (EIntrinsic
               ( [],
                 Sarek_warp.ir_name Sarek_warp.Xor Sarek_warp.I32,
                 [EVar x; EConst (CInt32 1l)] ));
      }
    in
    {shuffle_kernel with kern_funcs = [helper]}
  in
  match Sarek_codegen.Sarek_ir_metal.generate in_helper with
  | (_ : string) -> fail "Metal must refuse a shuffle inside a helper function"
  | exception _ -> ()

let () =
  run
    "warp capability and launch rule"
    [
      ( "capability gate",
        [
          test_case
            "refused without the feature"
            `Quick
            test_refused_without_feature;
          test_case "allowed with the feature" `Quick test_allowed_with_feature;
          test_case
            "plain kernel unaffected"
            `Quick
            test_plain_kernel_unaffected;
          test_case
            "host backends list it"
            `Quick
            test_host_backends_list_the_feature;
        ] );
      ( "launch rule",
        [
          test_case
            "block a multiple of the warp"
            `Quick
            test_block_multiple_of_warp;
          test_case
            "no shuffle: unaffected"
            `Quick
            test_unaffected_without_shuffle;
          test_case "simulated width" `Quick test_simulated_width;
          test_case
            "width must be a power of two"
            `Quick
            test_with_width_rejects_non_powers;
        ] );
      ("metal", [test_case "refusals" `Quick test_metal_refusals]);
    ]
