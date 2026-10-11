(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * Convergence analysis for barrier safety.
 *
 * GPU barriers require ALL threads in a workgroup to reach the barrier.
 * If threads diverge (take different branches) and only some reach the
 * barrier, the GPU deadlocks.
 *
 * The rules are specified in specs/convergence-semantics.md:
 * - §3 variance: computed by Sarek_variance, a whole-kernel fixpoint over the
 *   analysis IR of Sarek_variance_ir (assignments with control dependence,
 *   if/match values, declared intrinsic variance, private arrays, helper
 *   summaries);
 * - §2 supersteps: a superstep reached in diverged flow is an error with or
 *   without [@divergent], which is inert; body and continuation are checked in
 *   the outer mode;
 * - §4 helpers: a helper body is checked from Converged with its parameter
 *   variance; a call in diverged flow to a helper that contains a
 *   synchronisation point is rejected at the call;
 * - §5: a synchronisation point in diverged flow is an error; the right
 *   operand of [&&]/[||] runs under the left one (C-24); after a varying
 *   early return the rest of a sequence is diverged (F-04).
 *
 * Dimension usage and the native execution strategy live in Sarek_dim_usage,
 * included at the end.
 ******************************************************************************)

open Sarek_typed_ast
open Sarek_variance_ir
module StringSet = Set.Make (String)

(** Execution mode for convergence analysis *)
type exec_mode =
  | Converged  (** All threads in workgroup execute this code *)
  | Diverged  (** Threads may have taken different branches *)

(** Analysis context for {!check_expr}. [varying_vars] names free variables the
    caller knows to be thread-varying (a kernel needs none). *)
type ctx = {mode : exec_mode; varying_vars : StringSet.t}

let init_ctx = {mode = Converged; varying_vars = StringSet.empty}

type env = {
  st : Sarek_variance.state;
  fsync : (fkey, sync) Hashtbl.t;  (** helpers containing a sync point *)
}

let varies env n = Sarek_variance.node_may_vary env.st n

let inner env mode cond = if varies env cond then Diverged else mode

let sync_error loc = function
  | Barrier -> Sarek_error.Barrier_in_diverged_flow loc
  | Warp name -> Sarek_error.Warp_collective_in_diverged_flow (name, loc)

(* ------------------------------------------------------------------------- *)
(* Synchronisation summaries                                                  *)
(* ------------------------------------------------------------------------- *)

let callee_sync fsync = function
  | Fun f -> Hashtbl.find_opt fsync f
  | Prim name -> prim_sync name
  | Unknown -> Some Barrier

(** First synchronisation point [n] may execute, in evaluation order. *)
let rec node_sync fsync (n : node) : sync option =
  let own =
    match n.d with
    | Prim_call (_, s, _) -> s
    | Superstep _ -> Some Barrier
    | Call (_, callees, _) -> List.find_map (callee_sync fsync) callees
    | _ -> None
  in
  match own with
  | Some _ -> own
  | None -> List.find_map (node_sync fsync) (children n)

let compute_fsync (prog : program) =
  let fsync = Hashtbl.create 8 in
  let changed = ref true in
  while !changed do
    changed := false ;
    Hashtbl.iter
      (fun f (fn : fn) ->
        if not (Hashtbl.mem fsync f) then
          match node_sync fsync fn.fbody with
          | Some s ->
              Hashtbl.replace fsync f s ;
              changed := true
          | None -> ())
      prog.funs
  done ;
  fsync

(* ------------------------------------------------------------------------- *)
(* Early returns (F-04), mirroring ConvergenceSpec.has_varying_return         *)
(* ------------------------------------------------------------------------- *)

let rec has_reachable_return (n : node) =
  match n.d with
  | Return _ -> true
  | If (_, t, e) ->
      has_reachable_return t
      || Option.fold ~none:false ~some:has_reachable_return e
  | Match (_, cases) -> List.exists (fun (_, b) -> has_reachable_return b) cases
  | Seq ns -> List.exists has_reachable_return ns
  | Let (_, _, v, b) -> has_reachable_return v || has_reachable_return b
  | While (_, b) | For (_, _, _, b) -> has_reachable_return b
  | Superstep (b, k) -> has_reachable_return b || has_reachable_return k
  | Let_rec (_, k) -> has_reachable_return k
  | _ -> false

let rec has_varying_return env (n : node) =
  let hvr = has_varying_return env in
  match n.d with
  | If (c, t, e) ->
      if varies env c then
        has_reachable_return t
        || Option.fold ~none:false ~some:has_reachable_return e
      else hvr t || Option.fold ~none:false ~some:hvr e
  | Match (s, cases) ->
      if varies env s then
        List.exists (fun (_, b) -> has_reachable_return b) cases
      else List.exists (fun (_, b) -> hvr b) cases
  | Seq ns -> List.exists hvr ns
  | Let (_, _, v, b) -> hvr v || hvr b
  | While (c, b) -> hvr c || hvr b
  | For (_, lo, hi, b) -> hvr lo || hvr hi || hvr b
  | _ -> false

(* ------------------------------------------------------------------------- *)
(* Checking                                                                   *)
(* ------------------------------------------------------------------------- *)

let rec check env mode (n : node) : Sarek_error.error list =
  let here = check env mode in
  match n.d with
  | Prim_call (_, sync, args) ->
      List.concat_map here args @ sync_here mode n sync
  | If (c, t, e) ->
      let m = inner env mode c in
      here c @ check env m t @ Option.fold ~none:[] ~some:(check env m) e
  | Short (a, b) -> here a @ check env (inner env mode a) b
  | While (c, b) -> here c @ check env (inner env mode c) b
  | For (_, lo, hi, b) ->
      let m = if varies env lo || varies env hi then Diverged else mode in
      here lo @ here hi @ check env m b
  | Match (s, cases) ->
      let m = inner env mode s in
      here s @ List.concat_map (fun (_, b) -> check env m b) cases
  | Seq ns -> check_seq env mode ns
  | Call (sel, callees, args) -> check_call env mode n sel callees args
  | Superstep (b, k) ->
      (* SS-1 (any flag), SS-2/SS-3: body and continuation in the outer mode *)
      let entry =
        if mode = Diverged then [Sarek_error.Barrier_in_diverged_flow n.loc]
        else []
      in
      entry @ here b @ here k
  | _ -> List.concat_map here (children n)

and sync_here mode (n : node) sync =
  match (mode, sync) with Diverged, Some s -> [sync_error n.loc s] | _ -> []

and check_seq env mode = function
  | [] -> []
  | e :: rest ->
      let errors = check env mode e in
      let mode' = if has_varying_return env e then Diverged else mode in
      errors @ check_seq env mode' rest

(* H-2: a call in diverged flow (or through a varying selector) to a helper
   that contains a synchronisation point is rejected at the call. *)
and check_call env mode n sel callees args =
  let errors = check env mode sel @ List.concat_map (check env mode) args in
  match inner env mode sel with
  | Converged -> errors
  | Diverged -> (
      match List.find_map (callee_sync env.fsync) callees with
      | Some s -> errors @ [sync_error n.loc s]
      | None -> errors)

(** H-1: every helper body from Converged, then the main body in [mode]. *)
let check_program ?(seed = []) mode (prog : program) =
  let st = Sarek_variance.analyse ~seed prog in
  let env = {st; fsync = compute_fsync prog} in
  let fkeys =
    List.sort compare (Hashtbl.fold (fun f _ acc -> f :: acc) prog.funs [])
  in
  let fun_errors =
    List.concat_map
      (fun f -> check env Converged (Hashtbl.find prog.funs f).fbody)
      fkeys
  in
  fun_errors @ check env mode prog.body

(** Check one expression (used by tests and by the formal production link). *)
let check_expr ctx (te : texpr) : Sarek_error.error list =
  let prog, seed = of_expr ~varying:(StringSet.elements ctx.varying_vars) te in
  check_program ~seed ctx.mode prog

(** Is the value of [te] possibly thread-varying (block scope)? *)
let is_thread_varying (te : texpr) : bool =
  let prog, _ = of_expr te in
  let st = Sarek_variance.analyse prog in
  Sarek_variance.node_may_vary st prog.body

(** Check a kernel for convergence safety *)
let check_kernel (kernel : tkernel) : (unit, Sarek_error.error list) result =
  match check_program Converged (of_kernel kernel) with
  | [] -> Ok ()
  | errors -> Error errors

include Sarek_dim_usage
