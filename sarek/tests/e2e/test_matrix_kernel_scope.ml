(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A Kernel-scoped accuracy opt-in rewrites that kernel only
 * (specs/optimisation-policy.md, M2), on the Interpreter (no GPU).
 *
 * The rewrite renames the kernel it produces (<name>_sarek_mm): the capability
 * gate of the launch, and the interpreter, must read the budget of the kernel
 * the rewrite came from, not the process scope. Kernel "probe_a" holds
 * Accumulation_order; the same GEMM named probe_b holds nothing.
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
open Matrix_rewrite_support
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

let k_f16 =
  [%kernel
    fun (a : float16 vector)
        (b : float16 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum :=
            sum
            +. float32_of_float16 a.((row * k) + i)
               *. float32_of_float16 b.((i * n) + col)
        done ;
        c.((row * n) + col) <- sum
      end]

let size = 32

let named name = {(ir_of k_f16) with Sarek_ir_types.kern_name = name}

let run dev ir =
  let i = Execute.Int32 (Int32.of_int size) in
  let c = vec_f32_zero (size * size) in
  let block, grid = geo_2d ~m:size ~n:size ~nb:1 in
  let outcome =
    Sarek_matrix.run
      ~device:dev
      ~ir
      ~args:
        [
          Execute.Vec (vec_f16_of (Array.make (size * size) 0.5));
          Execute.Vec (vec_f16_of (Array.make (size * size) 0.25));
          Execute.Vec c;
          i;
          i;
          i;
        ]
      ~block
      ~grid
      ()
  in
  Transfer.flush dev ;
  (outcome, Spoc_core.Vector.get c 0)

let () =
  Sarek_execute.Matrix_specialise.(set_mode Auto) ;
  let dev =
    match
      List.find_opt
        (fun d -> d.Device.framework = "Interpreter")
        (Array.to_list (Device.init ()))
    with
    | Some d -> d
    | None ->
        prerr_endline "no Interpreter device" ;
        exit 2
  in
  (match
     Sarek_policy.grant
       (Sarek_policy.Kernel "probe_a")
       (Sarek_policy.Deviation Sarek_policy.Accumulation_order)
       ~reason:"test_matrix_kernel_scope"
   with
  | Ok () -> ()
  | Error _ -> fail "grant refused") ;
  (match run dev (named "probe_a") with
  | Sarek_matrix.Rewritten _, c0 ->
      ok "probe_a (Kernel grant): rewritten" ;
      if c0 = 4.0 then ok "probe_a: c.(0) = 4"
      else fail "probe_a: c.(0) = %g" c0
  | Sarek_matrix.Original why, _ -> fail "probe_a: not rewritten (%s)" why
  | exception e -> fail "probe_a: raised %s" (Printexc.to_string e)) ;
  (match run dev (named "probe_b") with
  | Sarek_matrix.Original why, _ -> ok "probe_b (no grant): original (%s)" why
  | Sarek_matrix.Rewritten _, _ -> fail "probe_b: rewritten without a grant"
  | exception e -> fail "probe_b: raised %s" (Printexc.to_string e)) ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
