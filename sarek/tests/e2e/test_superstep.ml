(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test let%shared and let%superstep constructs.
   These should all compile successfully.

   Syntax:
   - let%shared (name : elem_type) = () in body      -- shared array, default size
   - let%shared (name : elem_type) = size in body    -- shared array, explicit size
   - let%superstep name = body in cont               -- superstep with implicit barrier
   - let%superstep [@divergent] name = body in cont  -- accepted, no effect

   Rules: specs/convergence-semantics.md section 2. Structured control flow
   inside a superstep body reconverges before the implicit barrier, so a
   thread-varying `if` without a barrier in its branches is fine with or
   without [@divergent]. A superstep reached under a thread-varying condition
   is rejected with or without it (sarek/tests/negative/test_superstep_*.ml).
*)

(* Test 1: Basic shared memory with default size *)
let kernel_shared_basic =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) ->
      let%shared (tile : float32) = () in
      tile.(thread_idx_x) <- input.(global_thread_id) ;
      block_barrier () ;
      output.(global_thread_id) <- tile.(thread_idx_x)]

(* Test 2: Shared memory with explicit size *)
let kernel_shared_sized =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) ->
      let%shared (tile : float32) = 64l in
      tile.(thread_idx_x) <- input.(global_thread_id) ;
      block_barrier () ;
      output.(global_thread_id) <- tile.(thread_idx_x)]

(* Test 3: Basic superstep with implicit barrier *)
let kernel_superstep_basic =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) ->
      let%shared (tile : float32) = () in
      let%superstep load = tile.(thread_idx_x) <- input.(global_thread_id) in
      output.(global_thread_id) <- tile.(thread_idx_x)]

(* Test 4: Multiple supersteps in sequence *)
let kernel_superstep_chain =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) ->
      let%shared (tile : float32) = () in
      let%superstep load = tile.(thread_idx_x) <- input.(global_thread_id) in
      let%superstep process =
        let v = tile.(thread_idx_x) +. 1.0 in
        tile.(thread_idx_x) <- v
      in
      output.(global_thread_id) <- tile.(thread_idx_x)]

(* Test 5: [@divergent] superstep (the attribute is accepted and inert) *)
let kernel_superstep_divergent =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) ->
      let%shared (tile : float32) = () in
      let%superstep load = tile.(thread_idx_x) <- input.(global_thread_id) in
      let%superstep[@divergent] finalize =
        if thread_idx_x = 0l then output.(0l) <- tile.(0l)
      in
      ()]

(* Test 5b: thread-varying condition inside a superstep body, no barrier in
   the branch, no [@divergent] (SS-2). This kernel was the body of the old
   negative case test_superstep_diverged.ml, which expected a rejection and
   printed a non-blocking KNOWN-ISSUE when it compiled (D-38). It is legal:
   the threads that skip the branch and those that take it all reach the
   implicit end-of-step barrier. *)
let kernel_superstep_varying_if =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) ->
      let%shared (tile : float32) = () in
      let%superstep step =
        if thread_idx_x > 16l then
          tile.(thread_idx_x) <- input.(global_thread_id)
        else tile.(thread_idx_x) <- 0.0
      in
      output.(global_thread_id) <- tile.(thread_idx_x)]

(* Test 6: Uniform condition inside non-divergent superstep (OK) *)
let kernel_superstep_uniform_condition =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) ->
      let%shared (tile : float32) = () in
      let%superstep load =
        if block_idx_x > 0l then tile.(thread_idx_x) <- input.(global_thread_id)
        else tile.(thread_idx_x) <- 0.0
      in
      output.(global_thread_id) <- tile.(thread_idx_x)]

(* Test 7: Multiple shared arrays *)
let kernel_multi_shared =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) ->
      let%shared (tile1 : float32) = () in
      let%shared (tile2 : float32) = 32l in
      let%superstep load =
        tile1.(thread_idx_x) <- input.(global_thread_id) ;
        tile2.(thread_idx_x) <- input.(global_thread_id) +. 1.0
      in
      output.(global_thread_id) <- tile1.(thread_idx_x) +. tile2.(thread_idx_x)]

let () =
  print_endline
    "All let%shared and let%superstep tests passed (compilation success)"
