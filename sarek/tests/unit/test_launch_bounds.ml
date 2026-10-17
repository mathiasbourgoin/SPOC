(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * [pragma ["sarek.max_threads N"]] (launch bounds), without a device.
 * specs/launch-bounds-hint.md.
 *
 * 1. The pragma reaches the kernel IR ([kern_launch_bounds]) and is stripped
 *    from the body.
 * 2. Each backend's source: HIP/CUDA-C [__launch_bounds__], Metal
 *    [max_total_threads_per_threadgroup], OpenCL [reqd_work_group_size] and PTX
 *    [.maxntid] from the launch block, GLSL/WGSL unchanged, and a kernel
 *    without the hint byte-identical to the pre-feature output (no new token).
 * 3. The portable launch check, on the Interpreter path, with its negatives.
 * 4. Fusion combines bounds by minimum.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen
open Sarek
open Spoc_framework

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

let ir_of kern =
  let _, kirc = kern in
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let hinted_kernel =
  [%kernel
    fun (out : int32 vector) (n : int32) ->
      pragma
        ["sarek.max_threads 256"]
        (let tid = thread_idx_x + (block_dim_x * block_idx_x) in
         if tid < n then out.(tid) <- tid)]

let split_spelling_kernel =
  [%kernel
    fun (out : int32 vector) (n : int32) ->
      pragma
        ["sarek.max_threads"; "128"]
        (let tid = thread_idx_x + (block_dim_x * block_idx_x) in
         if tid < n then out.(tid) <- tid)]

let plain_kernel =
  [%kernel
    fun (out : int32 vector) (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then out.(tid) <- tid]

let hinted = ir_of hinted_kernel

let plain = ir_of plain_kernel

let no_hint = {hinted with kern_launch_bounds = None}

(** {1 IR} *)

let test_ir_carries_bound () =
  Alcotest.(check (option int))
    "256"
    (Option.map (fun b -> b.max_threads) hinted.kern_launch_bounds)
    (Some 256) ;
  Alcotest.(check (option int))
    "split spelling"
    (Option.map
       (fun b -> b.max_threads)
       (ir_of split_spelling_kernel).kern_launch_bounds)
    (Some 128) ;
  Alcotest.(check bool) "plain has none" true (plain.kern_launch_bounds = None)

let test_pragma_is_stripped () =
  let rec has_pragma = function
    | SPragma _ -> true
    | SSeq l -> List.exists has_pragma l
    | _ -> false
  in
  Alcotest.(check bool) "no SPragma left" false (has_pragma hinted.kern_body)

(** {1 Generated source} *)

let test_cuda_hip () =
  let src = Sarek_ir_cuda.generate hinted in
  Alcotest.(check bool)
    "launch_bounds"
    true
    (contains src "__global__ __launch_bounds__(256) void") ;
  let base = Sarek_ir_cuda.generate no_hint in
  Alcotest.(check bool)
    "no hint, no token"
    false
    (contains base "__launch_bounds__") ;
  Alcotest.(check bool)
    "no hint, plain header"
    true
    (contains base "__global__ void")

let test_metal () =
  let src = Sarek_ir_metal.generate hinted in
  Alcotest.(check bool)
    "attribute"
    true
    (contains src "[[max_total_threads_per_threadgroup(256)]] kernel void") ;
  Alcotest.(check bool)
    "no hint, no token"
    false
    (contains (Sarek_ir_metal.generate no_hint) "max_total_threads")

let test_opencl () =
  let with_block b =
    Sarek_ir_opencl.generate_with_types ~block:b ~types:[] hinted
  in
  Alcotest.(check bool)
    "reqd_work_group_size from the block"
    true
    (contains (with_block (16, 8, 1)) "reqd_work_group_size(16, 8, 1)") ;
  Alcotest.(check bool)
    "block over the bound: attribute not emitted"
    false
    (contains (with_block (32, 32, 1)) "reqd_work_group_size") ;
  Alcotest.(check bool)
    "no block known: nothing emitted"
    false
    (contains (Sarek_ir_opencl.generate hinted) "reqd_work_group_size") ;
  Alcotest.(check bool)
    "no hint: nothing emitted"
    false
    (contains
       (Sarek_ir_opencl.generate_with_types ~block:(16, 8, 1) ~types:[] no_hint)
       "reqd_work_group_size")

let test_ptx () =
  let gen ?block k = Sarek_ir_ptx.generate ?block k in
  Alcotest.(check bool)
    "maxntid from the block"
    true
    (contains (gen ~block:(16, 8, 1) hinted) ".maxntid 16, 8, 1") ;
  Alcotest.(check bool)
    "block over the bound"
    false
    (contains (gen ~block:(32, 32, 1) hinted) ".maxntid") ;
  Alcotest.(check bool) "no block" false (contains (gen hinted) ".maxntid") ;
  Alcotest.(check string)
    "no hint: block changes nothing"
    (gen no_hint)
    (gen ~block:(16, 8, 1) no_hint)

let test_vulkan_webgpu_unchanged () =
  Alcotest.(check string)
    "GLSL"
    (Sarek_ir_glsl.generate no_hint)
    (Sarek_ir_glsl.generate hinted) ;
  Alcotest.(check string)
    "WGSL"
    (Sarek_ir_wgsl.generate no_hint)
    (Sarek_ir_wgsl.generate hinted)

(** {1 Portable launch check} *)

let dims x y z = Framework_sig.{x; y; z}

let refused ~block ir =
  match Execute.check_launch_bounds ~kernel:"k" ir ~block with
  | () -> None
  | exception Execute_error.Execution_error (Invalid_dimensions {reason; _}) ->
      Some reason

let test_check_launch_bounds () =
  Alcotest.(check bool)
    "inside"
    true
    (refused ~block:(dims 16 16 1) hinted = None) ;
  Alcotest.(check bool)
    "exactly the bound"
    true
    (refused ~block:(dims 256 1 1) hinted = None) ;
  (match refused ~block:(dims 32 16 1) hinted with
  | Some reason ->
      Alcotest.(check bool) "names the bound" true (contains reason "256") ;
      Alcotest.(check bool) "names the block size" true (contains reason "512") ;
      Alcotest.(check bool) "names the kernel" true (contains reason "\"k\"")
  | None -> Alcotest.fail "512 threads over a 256 bound must be refused") ;
  Alcotest.(check bool)
    "no hint: never refused"
    true
    (refused ~block:(dims 1024 1 1) no_hint = None)

let test_interpreter_refuses_before_running () =
  let k =
    {
      default_kernel with
      kern_name = "empty";
      kern_launch_bounds = Some {max_threads = 64};
    }
  in
  (match
     Execute.run_interpreter_vectors
       ~ir:k
       ~args:[]
       ~block:(dims 128 1 1)
       ~grid:(dims 1 1 1)
       ~parallel:false
   with
  | () -> Alcotest.fail "launch over the bound must raise"
  | exception Execute_error.Execution_error (Invalid_dimensions _) -> ()) ;
  Execute.run_interpreter_vectors
    ~ir:k
    ~args:[]
    ~block:(dims 64 1 1)
    ~grid:(dims 1 1 1)
    ~parallel:false

(** {1 Fusion} *)

let bound n = Some {max_threads = n}

let test_combine () =
  let c l = Option.map (fun b -> b.max_threads) (combine_launch_bounds l) in
  Alcotest.(check (option int)) "none" None (c [None; None]) ;
  Alcotest.(check (option int)) "one" (Some 64) (c [None; bound 64]) ;
  Alcotest.(check (option int)) "minimum" (Some 64) (c [bound 256; bound 64]) ;
  Alcotest.(check (option int)) "empty" None (c [])

let test_horizontal_fusion_takes_minimum () =
  let member name b = {hinted with kern_name = name; kern_launch_bounds = b} in
  let fused =
    Sarek_execute.Fusion_horizontal.fuse
      ~name:"h"
      [member "a" (bound 256); member "b" (bound 64); member "c" None]
  in
  Alcotest.(check (option int))
    "minimum of the members"
    (Some 64)
    (Option.map (fun b -> b.max_threads) fused.kern_launch_bounds) ;
  let none =
    Sarek_execute.Fusion_horizontal.fuse
      ~name:"h"
      [member "a" None; member "b" None]
  in
  Alcotest.(check bool)
    "no member declares one"
    true
    (none.kern_launch_bounds = None)

let () =
  Alcotest.run
    "launch_bounds"
    [
      ( "ir",
        [
          Alcotest.test_case "bound reaches the IR" `Quick test_ir_carries_bound;
          Alcotest.test_case "pragma stripped" `Quick test_pragma_is_stripped;
        ] );
      ( "codegen",
        [
          Alcotest.test_case "HIP / CUDA-C" `Quick test_cuda_hip;
          Alcotest.test_case "Metal" `Quick test_metal;
          Alcotest.test_case "OpenCL" `Quick test_opencl;
          Alcotest.test_case "PTX" `Quick test_ptx;
          Alcotest.test_case
            "Vulkan / WebGPU unchanged"
            `Quick
            test_vulkan_webgpu_unchanged;
        ] );
      ( "launch check",
        [
          Alcotest.test_case "block vs bound" `Quick test_check_launch_bounds;
          Alcotest.test_case
            "interpreter"
            `Quick
            test_interpreter_refuses_before_running;
        ] );
      ( "fusion",
        [
          Alcotest.test_case "combine" `Quick test_combine;
          Alcotest.test_case
            "horizontal"
            `Quick
            test_horizontal_fusion_takes_minimum;
        ] );
    ]
