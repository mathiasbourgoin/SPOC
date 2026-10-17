(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Kernels that must PASS the front end: every warp shuffle is in converged
   control flow, in every spelling. The assertion is successful compilation. *)

[@@@warning "-39"]

module Std = Sarek_stdlib.Std
module Gpu = Sarek_stdlib.Gpu

(* OK: unconditional, all four primitives, three value types *)
let kernel_all_primitives =
  [%kernel
    fun (v : int32 vector) (f : float32 vector) (d : float64 vector) ->
      let t = thread_idx_x in
      v.(t) <- warp_shuffle v.(t) 3l ;
      v.(t) <- warp_shuffle_down v.(t) 1l ;
      v.(t) <- warp_shuffle_up v.(t) 1l ;
      v.(t) <- warp_shuffle_xor v.(t) 4l ;
      f.(t) <- warp_shuffle_xor f.(t) 2l ;
      d.(t) <- warp_shuffle_down d.(t) 1l]

(* OK: after a thread-varying if, back in converged flow *)
let kernel_after_varying_if =
  [%kernel
    fun (v : int32 vector) ->
      let x = if thread_idx_x > 16 then 1l else 2l in
      v.(thread_idx_x) <- warp_shuffle_xor x 1l]

(* OK: under a block-uniform condition and a uniform loop bound *)
let kernel_uniform_control =
  [%kernel
    fun (v : float32 vector) (n : int32) ->
      if block_idx_x > 0l then
        for _i = 0 to 3 do
          v.(thread_idx_x) <-
            v.(thread_idx_x) +. warp_shuffle_xor v.(thread_idx_x) 1l
        done]

(* OK: the qualified spellings, and warp_size / lane_id as values *)
let kernel_qualified =
  [%kernel
    fun (v : int32 vector) ->
      let w = Gpu.warp_size in
      let l = Std.lane_id in
      v.(thread_idx_x) <- Gpu.warp_shuffle_xor (l + w) 1l]

(* OK: a helper containing a shuffle, called from converged flow *)
let kernel_helper =
  [%kernel
    fun (v : int32 vector) ->
      let reduce (x : int32) = warp_shuffle_xor x 1l in
      v.(thread_idx_x) <- reduce v.(thread_idx_x)]

(* OK: the shuffle result is thread-varying, and a varying if AFTER it is fine *)
let kernel_result_varying =
  [%kernel
    fun (v : int32 vector) ->
      let x = warp_shuffle v.(thread_idx_x) 0l in
      if x > 0l then v.(thread_idx_x) <- x]

(* OK: a later helper of the same name shadows the one that shuffles *)
let kernel_shadowed_helper =
  [%kernel
    fun (v : int32 vector) ->
      let reduce (x : int32) = warp_shuffle_xor x 1l in
      v.(thread_idx_x) <- reduce v.(thread_idx_x) ;
      let reduce (y : int32) = y + 1l in
      if thread_idx_x > 16 then v.(thread_idx_x) <- reduce v.(thread_idx_x)]

let () = print_endline "All warp shuffle convergence tests passed (compilation)"
