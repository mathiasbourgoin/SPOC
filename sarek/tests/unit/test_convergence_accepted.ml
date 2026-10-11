(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Kernels the convergence check must ACCEPT (specs/convergence-semantics.md).
   They pin the precision of the variance analysis: each one is safe, and a
   coarser analysis would reject it. The test is that this file compiles
   through the PPX; the cases below only check that each kernel has IR. *)

(* G-11: a helper whose barrier depends on its parameter, called only with a
   kernel parameter (uniform). *)
let helper_uniform_arg =
  [%kernel
    fun (v : int32 vector) (n : int32) ->
      let f (m : int32) = if m > 0l then block_barrier () in
      f n ;
      v.(thread_idx_x) <- n]

(* Helper result summaries are per call: [idx thread_idx_x] varies,
   [idx n] does not. *)
let helper_result_per_call =
  [%kernel
    fun (v : int32 vector) (n : int32) ->
      let idx (m : int32) = m * 2l in
      v.(thread_idx_x) <- idx thread_idx_x ;
      if idx n > 4l then block_barrier ()]

(* G-14: broadcast through shared memory. One thread writes, a barrier
   publishes, every thread reads the same location (race freedom). *)
let shared_broadcast =
  [%kernel
    fun (v : int32 vector) (n : int32) ->
      let%shared (s : int32) = 4l in
      if thread_idx_x = 0l then s.(0l) <- v.(0l) ;
      block_barrier () ;
      if s.(0l) > 0l then block_barrier () ;
      v.(thread_idx_x) <- s.(0l)]

(* A uniform loop counter around a barrier. *)
let uniform_counter =
  [%kernel
    fun (v : int32 vector) (n : int32) ->
      let i = mut 0l in
      while i < n do
        block_barrier () ;
        i := i + 1l
      done ;
      v.(thread_idx_x) <- i]

(* Control dependence is relative to the declaration: [y] is declared and
   assigned inside the same varying branch, so it does not make anything
   after the branch varying; the barrier is after the branch. *)
let assignment_inside_region =
  [%kernel
    fun (v : int32 vector) (n : int32) ->
      if thread_idx_x > 16l then begin
        let y = mut 0l in
        y := 1l ;
        v.(thread_idx_x) <- y
      end ;
      block_barrier ()]

let has_ir (_, kirc) = Option.is_some kirc.Sarek.Kirc_types.body_ir

let case name k =
  Alcotest.test_case name `Quick (fun () ->
      Alcotest.(check bool) (name ^ " has IR") true (has_ir k))

let () =
  Alcotest.run
    "convergence_accepted"
    [
      ( "accepted",
        [
          case "G-11 helper with uniform argument" helper_uniform_arg;
          case "helper result per call" helper_result_per_call;
          case "G-14 shared broadcast" shared_broadcast;
          case "uniform loop counter" uniform_counter;
          case "assignment inside its region" assignment_inside_region;
        ] );
    ]
