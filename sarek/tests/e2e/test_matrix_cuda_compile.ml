(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The int32 byte-plane rewrite on CUDA/C is compiler-output evidence only
 * (specs/matrix-pattern-rewrite.md): no NVIDIA hardware is available, and
 * ZLUDA cannot load wmma.* or unsigned 8-bit mma. This test generates the
 * rewritten kernel's CUDA C, compiles it with NVRTC for sm_80 and checks that
 * the PTX reaches the u8 tensor-core instruction. It skips (loudly) when NVRTC
 * or a CUDA include directory is missing.
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
open Matrix_rewrite_support
module P = Sarek_matrix_pattern

let contains haystack needle =
  let n = String.length needle and h = String.length haystack in
  let rec at i =
    i + n <= h && (String.sub haystack i n = needle || at (i + 1))
  in
  at 0

let () =
  let ir = ir_of k_2d in
  let gemm =
    match P.analyze ir with
    | [P.Gemm g] -> g
    | _ -> failwith "k_2d is not recognised as a product"
  in
  let rewritten =
    match
      Sarek_matrix_rewrite.kernel
        ~subgroup:32
        ir
        gemm
        Sarek_matrix_rewrite.Int32_byte_planes
    with
    | Ok k -> k
    | Error why -> failwith ("rewrite declined: " ^ why)
  in
  let source = Backend_cuda_c.source_of rewritten in
  if source <> "" && not (contains source "nvcuda::wmma") then
    fail "the rewritten CUDA source has no nvcuda::wmma" ;
  (match
     Backend_cuda_c.ptx_of ~name:rewritten.Sarek_ir_types.kern_name source
   with
  | None -> print_endline "  SKIP no libnvrtc or no CUDA include directory"
  | Some ptx ->
      if not (contains ptx "wmma.mma.sync") then
        fail "no wmma.mma.sync in the sm_80 PTX" ;
      if not (contains ptx ".u8.u8.s32") then
        fail "no u8 x u8 -> s32 wmma in the sm_80 PTX" ;
      if !failures = 0 then
        ok "rewritten int32 kernel: NVRTC sm_80 PTX has the u8 wmma") ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
