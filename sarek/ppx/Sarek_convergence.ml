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
 * This pass tracks execution mode (Converged vs Diverged) and rejects
 * barrier calls when mode is Diverged.
 *
 * Example of rejected code:
 *   if thread_idx_x > 16 then
 *     block_barrier ()   (* Error: only threads 17-31 reach this! *)
 *
 * This minimal implementation catches ~80% of real bugs:
 * - Direct thread_idx_* comparisons in if/while conditions
 * - Array accesses with thread-varying indices in conditions
 *
 * Future extensions:
 * - Dataflow analysis: track thread-varying through variable assignments
 * - Path-sensitive: allow `if c then barrier() else barrier()`
 * - Inter-procedural: analyze called functions
 ******************************************************************************)

open Sarek_typed_ast
open Sarek_env

(** Execution mode for convergence analysis *)
type exec_mode =
  | Converged  (** All threads in workgroup execute this code *)
  | Diverged  (** Threads may have taken different branches *)

(** Analysis context *)
type ctx = {
  mode : exec_mode;
      (* Future: varying_vars : StringSet.t for dataflow analysis *)
}

(** Initial context - start converged *)
let init_ctx = {mode = Converged}

(** Enter diverged mode *)
let diverge _ctx = {mode = Diverged}

(** Check if an intrinsic ref is thread-varying using core primitives *)
let is_thread_varying_intrinsic (ref : intrinsic_ref) : bool =
  match ref with
  | IntrinsicRef (_, base_name) ->
      Sarek_core_primitives.is_thread_varying base_name
  | CorePrimitiveRef name -> Sarek_core_primitives.is_thread_varying name

(** Check if an intrinsic ref is a barrier/convergence point *)
let is_barrier_ref (ref : intrinsic_ref) : bool =
  match ref with
  | IntrinsicRef (_, base_name) ->
      Sarek_core_primitives.is_convergence_point base_name
  | CorePrimitiveRef name -> Sarek_core_primitives.is_convergence_point name

(** Check if an expression's value varies per-thread.

    Thread-varying (different per thread):
    - thread_idx_x, thread_idx_y, thread_idx_z
    - global_thread_id
    - Expressions derived from above
    - Array accesses with thread-varying index

    Uniform (same for all threads in workgroup):
    - Constants and literals
    - block_idx_x/y/z (same for whole block)
    - block_dim_x/y/z, grid_dim_x/y/z
    - Expressions derived only from uniform values *)
let rec is_thread_varying (te : texpr) : bool =
  match te.te with
  (* Literals are uniform *)
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
    ->
      false
  (* Check if variable is a thread-varying intrinsic *)
  | TEVar (name, _) -> Sarek_core_primitives.is_thread_varying name
  (* Intrinsic constants - check if thread-varying *)
  | TEIntrinsicConst ref -> is_thread_varying_intrinsic ref
  (* Binary ops: varying if either operand varies *)
  | TEBinop (_, e1, e2) -> is_thread_varying e1 || is_thread_varying e2
  (* Unary ops: varying if operand varies *)
  | TEUnop (_, e) -> is_thread_varying e
  (* Array/vector access: varying if index varies *)
  | TEVecGet (_, idx) | TEArrGet (_, idx) -> is_thread_varying idx
  (* Field access: varying if record varies *)
  | TEFieldGet (e, _, _) -> is_thread_varying e
  (* Application: varying if any argument varies *)
  | TEApp (_, args) -> List.exists is_thread_varying args
  (* Intrinsic function: varying if any argument varies *)
  | TEIntrinsicFun (_, _, args) -> List.exists is_thread_varying args
  (* Tuple: varying if any element varies *)
  | TETuple es -> List.exists is_thread_varying es
  (* Let binding: check the bound expression and body *)
  | TELet (_, _, value, body) | TELetMut (_, _, value, body) ->
      is_thread_varying value || is_thread_varying body
  (* Typed expression: shouldn't appear but check inner *)
  | TESeq es -> List.exists is_thread_varying es
  (* BSP constructs: check body/continuation *)
  | TELetShared (_, _, _, size_opt, body) ->
      (match size_opt with
        | Some size -> is_thread_varying size
        | None -> false)
      || is_thread_varying body
  | TESuperstep (_, _, step_body, cont) ->
      is_thread_varying step_body || is_thread_varying cont
  | TEOpen (_, body) -> is_thread_varying body
  (* Default: assume uniform (minimize false positives) *)
  | TEIf _ | TEFor _ | TEWhile _ | TEMatch _ | TERecord _ | TEConstr _
  | TEReturn _ | TECreateArray _ | TEGlobalRef _ | TENative _ | TEPragma _
  | TEVecSet _ | TEArrSet _ | TEFieldSet _ | TEAssign _ | TELetRec _ ->
      false

(** Check if an intrinsic reference is a barrier *)
let is_barrier_intrinsic (ref : intrinsic_ref) : bool = is_barrier_ref ref

(** Check if an intrinsic ref requires warp convergence *)
let is_warp_convergence_ref (ref : intrinsic_ref) : bool =
  match ref with
  | IntrinsicRef (_, base_name) ->
      Sarek_core_primitives.is_warp_convergence_point base_name
  | CorePrimitiveRef name ->
      Sarek_core_primitives.is_warp_convergence_point name

(** Collect errors from convergence analysis *)
let rec check_expr ctx (te : texpr) : Sarek_error.error list =
  match te.te with
  (* Intrinsic function calls - check convergence requirement from AST *)
  | TEIntrinsicFun (ref, convergence, args) ->
      let arg_errors = List.concat_map (check_expr ctx) args in
      if ctx.mode = Diverged then
        match convergence with
        | Some Sarek_core_primitives.ConvergencePoint ->
            Sarek_error.Barrier_in_diverged_flow te.te_loc :: arg_errors
        | Some Sarek_core_primitives.WarpConvergence ->
            let name = Sarek_env.intrinsic_ref_display_name ref in
            Sarek_error.Warp_collective_in_diverged_flow (name, te.te_loc)
            :: arg_errors
        | Some Sarek_core_primitives.NoEffect | None -> arg_errors
      else arg_errors
  (* If expression - diverge if condition is thread-varying *)
  | TEIf (cond, then_e, else_opt) ->
      let cond_errors = check_expr ctx cond in
      let inner_ctx = if is_thread_varying cond then diverge ctx else ctx in
      let then_errors = check_expr inner_ctx then_e in
      let else_errors =
        match else_opt with
        | None -> []
        | Some else_e -> check_expr inner_ctx else_e
      in
      cond_errors @ then_errors @ else_errors
  (* While loop - diverge if condition is thread-varying *)
  | TEWhile (cond, body) ->
      let cond_errors = check_expr ctx cond in
      let inner_ctx = if is_thread_varying cond then diverge ctx else ctx in
      let body_errors = check_expr inner_ctx body in
      cond_errors @ body_errors
  (* For loop - check if bounds are thread-varying *)
  | TEFor (_, _, lo, hi, _, body) ->
      let lo_errors = check_expr ctx lo in
      let hi_errors = check_expr ctx hi in
      let inner_ctx =
        if is_thread_varying lo || is_thread_varying hi then diverge ctx
        else ctx
      in
      let body_errors = check_expr inner_ctx body in
      lo_errors @ hi_errors @ body_errors
  (* Match - diverge if scrutinee is thread-varying *)
  | TEMatch (scrut, cases) ->
      let scrut_errors = check_expr ctx scrut in
      let inner_ctx = if is_thread_varying scrut then diverge ctx else ctx in
      let case_errors =
        List.concat_map (fun (_, e) -> check_expr inner_ctx e) cases
      in
      scrut_errors @ case_errors
  (* Recursively check subexpressions - these don't change mode *)
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ ->
      []
  | TEVecGet (v, i) -> check_expr ctx v @ check_expr ctx i
  | TEVecSet (v, i, x) -> check_expr ctx v @ check_expr ctx i @ check_expr ctx x
  | TEArrGet (a, i) -> check_expr ctx a @ check_expr ctx i
  | TEArrSet (a, i, x) -> check_expr ctx a @ check_expr ctx i @ check_expr ctx x
  | TEFieldGet (e, _, _) -> check_expr ctx e
  | TEFieldSet (e, _, _, x) -> check_expr ctx e @ check_expr ctx x
  | TEBinop (_, a, b) -> check_expr ctx a @ check_expr ctx b
  | TEUnop (_, e) -> check_expr ctx e
  | TEApp (f, args) -> check_expr ctx f @ List.concat_map (check_expr ctx) args
  | TEAssign (_, _, e) -> check_expr ctx e
  | TELet (_, _, value, body) | TELetMut (_, _, value, body) ->
      check_expr ctx value @ check_expr ctx body
  | TESeq es -> List.concat_map (check_expr ctx) es
  | TERecord (_, fields) ->
      List.concat_map (fun (_, e) -> check_expr ctx e) fields
  | TEConstr (_, _, arg) -> (
      match arg with None -> [] | Some e -> check_expr ctx e)
  | TETuple es -> List.concat_map (check_expr ctx) es
  | TEReturn e -> check_expr ctx e
  | TECreateArray (size, _, _) -> check_expr ctx size
  | TEPragma (_, body) -> check_expr ctx body
  (* BSP constructs *)
  | TELetShared (_, _, _, size_opt, body) ->
      let size_errors =
        match size_opt with None -> [] | Some size -> check_expr ctx size
      in
      size_errors @ check_expr ctx body
  | TESuperstep (_, divergent, step_body, cont) ->
      (* For non-divergent supersteps, check that the body doesn't contain
         any diverging control flow (thread-varying conditions).
         The implicit barrier at the end requires all threads to reach it. *)
      let body_errors =
        if divergent then
          (* Divergent flag set: allow any control flow, just check for nested barriers *)
          check_expr ctx step_body
        else
          (* Non-divergent: check for any divergence in the body.
             We do this by checking in Converged mode - if there's an if/while
             with thread-varying condition followed by a barrier, that's caught.
             But we also need to catch the implicit barrier at the end.
             So if the body ends in diverged mode (any thread-varying branch taken),
             that's an error because the implicit barrier would be in diverged flow. *)
          let body_errs = check_expr {mode = Converged} step_body in
          (* Also check: does the body contain any diverging conditions?
             If so, the implicit barrier at the end is in diverged flow. *)
          if contains_diverging_control_flow step_body then
            Sarek_error.Barrier_in_diverged_flow te.te_loc :: body_errs
          else body_errs
      in
      (* After the implicit barrier, we're back to converged *)
      let cont_errors = check_expr {mode = Converged} cont in
      body_errors @ cont_errors
  | TEOpen (_, body) -> check_expr ctx body
  | TELetRec (_, _, _, fn_body, cont) ->
      check_expr ctx fn_body @ check_expr ctx cont

(** Check if an expression contains any control flow with thread-varying
    conditions. This is used for superstep analysis - the implicit barrier at
    the end of a superstep requires that no divergence occurs within the body.
*)
and contains_diverging_control_flow (te : texpr) : bool =
  match te.te with
  | TEIf (cond, then_e, else_opt) -> (
      is_thread_varying cond
      || contains_diverging_control_flow then_e
      ||
      match else_opt with
      | None -> false
      | Some e -> contains_diverging_control_flow e)
  | TEWhile (cond, body) ->
      is_thread_varying cond || contains_diverging_control_flow body
  | TEFor (_, _, lo, hi, _, body) ->
      is_thread_varying lo || is_thread_varying hi
      || contains_diverging_control_flow body
  | TEMatch (scrut, cases) ->
      is_thread_varying scrut
      || List.exists (fun (_, e) -> contains_diverging_control_flow e) cases
  (* Recursively check subexpressions *)
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) ->
      contains_diverging_control_flow v || contains_diverging_control_flow b
  | TESeq es -> List.exists contains_diverging_control_flow es
  | TEBinop (_, a, b) ->
      contains_diverging_control_flow a || contains_diverging_control_flow b
  | TEUnop (_, e) -> contains_diverging_control_flow e
  | TEApp (_, args) -> List.exists contains_diverging_control_flow args
  | TEIntrinsicFun (_, _, args) ->
      List.exists contains_diverging_control_flow args
  | TEVecGet (v, i) ->
      contains_diverging_control_flow v || contains_diverging_control_flow i
  | TEVecSet (v, i, x) ->
      contains_diverging_control_flow v
      || contains_diverging_control_flow i
      || contains_diverging_control_flow x
  | TEArrGet (a, i) ->
      contains_diverging_control_flow a || contains_diverging_control_flow i
  | TEArrSet (a, i, x) ->
      contains_diverging_control_flow a
      || contains_diverging_control_flow i
      || contains_diverging_control_flow x
  | TEFieldGet (e, _, _) -> contains_diverging_control_flow e
  | TEFieldSet (e, _, _, x) ->
      contains_diverging_control_flow e || contains_diverging_control_flow x
  | TERecord (_, fields) ->
      List.exists (fun (_, e) -> contains_diverging_control_flow e) fields
  | TETuple es -> List.exists contains_diverging_control_flow es
  | TEReturn e -> contains_diverging_control_flow e
  | TEPragma (_, body) -> contains_diverging_control_flow body
  | TEAssign (_, _, e) -> contains_diverging_control_flow e
  | TECreateArray (size, _, _) -> contains_diverging_control_flow size
  | TEConstr (_, _, arg) -> (
      match arg with
      | None -> false
      | Some e -> contains_diverging_control_flow e)
  | TELetShared (_, _, _, size_opt, body) ->
      (match size_opt with
        | None -> false
        | Some s -> contains_diverging_control_flow s)
      || contains_diverging_control_flow body
  | TESuperstep (_, _, step_body, cont) ->
      contains_diverging_control_flow step_body
      || contains_diverging_control_flow cont
  | TEOpen (_, body) -> contains_diverging_control_flow body
  | TELetRec (_, _, _, fn_body, cont) ->
      contains_diverging_control_flow fn_body
      || contains_diverging_control_flow cont
  (* Terminal cases - no nested control flow *)
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ ->
      false

(** Check a module item *)
let check_module_item ctx (item : tmodule_item) : Sarek_error.error list =
  match item with
  | TMConst (_, _, _, e) -> check_expr ctx e
  | TMFun (_, _, _, e) -> check_expr ctx e

(** Check a kernel for convergence safety *)
let check_kernel (kernel : tkernel) : (unit, Sarek_error.error list) result =
  let ctx = init_ctx in
  (* Check module items first *)
  let item_errors =
    List.concat_map (check_module_item ctx) kernel.tkern_module_items
  in
  (* Check kernel body *)
  let body_errors = check_expr ctx kernel.tkern_body in
  let all_errors = item_errors @ body_errors in
  if all_errors = [] then Ok () else Error all_errors

(** Check if a kernel uses any barriers (explicit or implicit). Used for
    compile-time optimization of native kernel execution. *)
let rec expr_uses_barriers (te : texpr) : bool =
  match te.te with
  (* Superstep has implicit barrier *)
  | TESuperstep (_, _, step_body, cont) ->
      true || expr_uses_barriers step_body || expr_uses_barriers cont
  (* Intrinsic function - check if it's a convergence point *)
  | TEIntrinsicFun (_, Some Sarek_core_primitives.ConvergencePoint, _) -> true
  | TEIntrinsicFun (_, _, args) -> List.exists expr_uses_barriers args
  (* Recursively check subexpressions *)
  | TEIf (cond, then_e, else_opt) ->
      expr_uses_barriers cond || expr_uses_barriers then_e
      || Option.is_some
           (Option.bind else_opt (fun e ->
                if expr_uses_barriers e then Some () else None))
  | TEWhile (cond, body) -> expr_uses_barriers cond || expr_uses_barriers body
  | TEFor (_, _, lo, hi, _, body) ->
      expr_uses_barriers lo || expr_uses_barriers hi || expr_uses_barriers body
  | TEMatch (scrut, cases) ->
      expr_uses_barriers scrut
      || List.exists (fun (_, e) -> expr_uses_barriers e) cases
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) ->
      expr_uses_barriers v || expr_uses_barriers b
  | TESeq es -> List.exists expr_uses_barriers es
  | TEBinop (_, a, b) -> expr_uses_barriers a || expr_uses_barriers b
  | TEUnop (_, e) -> expr_uses_barriers e
  | TEApp (f, args) ->
      expr_uses_barriers f || List.exists expr_uses_barriers args
  | TEVecGet (v, i) -> expr_uses_barriers v || expr_uses_barriers i
  | TEVecSet (v, i, x) ->
      expr_uses_barriers v || expr_uses_barriers i || expr_uses_barriers x
  | TEArrGet (a, i) -> expr_uses_barriers a || expr_uses_barriers i
  | TEArrSet (a, i, x) ->
      expr_uses_barriers a || expr_uses_barriers i || expr_uses_barriers x
  | TEFieldGet (e, _, _) -> expr_uses_barriers e
  | TEFieldSet (e, _, _, x) -> expr_uses_barriers e || expr_uses_barriers x
  | TERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_barriers e) fields
  | TETuple es -> List.exists expr_uses_barriers es
  | TEReturn e -> expr_uses_barriers e
  | TEPragma (_, body) -> expr_uses_barriers body
  | TEAssign (_, _, e) -> expr_uses_barriers e
  | TECreateArray (size, _, _) -> expr_uses_barriers size
  | TEConstr (_, _, arg) ->
      Option.is_some
        (Option.bind arg (fun e ->
             if expr_uses_barriers e then Some () else None))
  | TELetShared (_, _, _, size_opt, body) ->
      (match size_opt with None -> false | Some s -> expr_uses_barriers s)
      || expr_uses_barriers body
  | TEOpen (_, body) -> expr_uses_barriers body
  | TELetRec (_, _, _, fn_body, cont) ->
      expr_uses_barriers fn_body || expr_uses_barriers cont
  (* Terminal cases - no barriers *)
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ ->
      false

(** Check if a kernel uses barriers *)
let kernel_uses_barriers (kernel : tkernel) : bool =
  (* Check module items *)
  let item_uses =
    List.exists
      (fun (item : tmodule_item) ->
        match item with
        | TMConst (_, _, _, e) -> expr_uses_barriers e
        | TMFun (_, _, _, e) -> expr_uses_barriers e)
      kernel.tkern_module_items
  in
  item_uses || expr_uses_barriers kernel.tkern_body

(** {1 Kernel Dimensionality Analysis}

    Detect which thread/block dimensions a kernel uses to enable optimized
    native CPU execution. Simple kernels that only use global_idx_x can be run
    with a simple parallel for loop without the thread_state overhead. *)

(** Dimension usage record *)
type dim_usage = {
  uses_x : bool;
  uses_y : bool;
  uses_z : bool;
  uses_block_dim : bool;  (** Uses block_dim_x/y/z *)
  uses_grid_dim : bool;  (** Uses grid_dim_x/y/z *)
  uses_thread_idx : bool;  (** Uses thread_idx_x/y/z directly *)
  uses_block_idx : bool;  (** Uses block_idx_x/y/z directly *)
  uses_shared_mem : bool;  (** Uses shared memory *)
}

let empty_dim_usage =
  {
    uses_x = false;
    uses_y = false;
    uses_z = false;
    uses_block_dim = false;
    uses_grid_dim = false;
    uses_thread_idx = false;
    uses_block_idx = false;
    uses_shared_mem = false;
  }

let merge_dim_usage a b =
  {
    uses_x = a.uses_x || b.uses_x;
    uses_y = a.uses_y || b.uses_y;
    uses_z = a.uses_z || b.uses_z;
    uses_block_dim = a.uses_block_dim || b.uses_block_dim;
    uses_grid_dim = a.uses_grid_dim || b.uses_grid_dim;
    uses_thread_idx = a.uses_thread_idx || b.uses_thread_idx;
    uses_block_idx = a.uses_block_idx || b.uses_block_idx;
    uses_shared_mem = a.uses_shared_mem || b.uses_shared_mem;
  }

(** Check if an intrinsic name affects dimension usage *)
let dim_usage_of_name name =
  match name with
  | "global_idx" | "global_idx_x" | "global_size" | "global_size_x"
  | "global_thread_id" ->
      {empty_dim_usage with uses_x = true}
  | "global_idx_y" | "global_size_y" ->
      {empty_dim_usage with uses_x = true; uses_y = true}
  | "global_idx_z" | "global_size_z" ->
      {empty_dim_usage with uses_x = true; uses_y = true; uses_z = true}
  | "thread_idx_x" ->
      {empty_dim_usage with uses_x = true; uses_thread_idx = true}
  | "thread_idx_y" ->
      {empty_dim_usage with uses_y = true; uses_thread_idx = true}
  | "thread_idx_z" ->
      {empty_dim_usage with uses_z = true; uses_thread_idx = true}
  | "block_idx_x" -> {empty_dim_usage with uses_x = true; uses_block_idx = true}
  | "block_idx_y" -> {empty_dim_usage with uses_y = true; uses_block_idx = true}
  | "block_idx_z" -> {empty_dim_usage with uses_z = true; uses_block_idx = true}
  | "block_dim_x" -> {empty_dim_usage with uses_x = true; uses_block_dim = true}
  | "block_dim_y" -> {empty_dim_usage with uses_y = true; uses_block_dim = true}
  | "block_dim_z" -> {empty_dim_usage with uses_z = true; uses_block_dim = true}
  | "grid_dim_x" -> {empty_dim_usage with uses_x = true; uses_grid_dim = true}
  | "grid_dim_y" -> {empty_dim_usage with uses_y = true; uses_grid_dim = true}
  | "grid_dim_z" -> {empty_dim_usage with uses_z = true; uses_grid_dim = true}
  | _ -> empty_dim_usage

(** Get dimension usage from an intrinsic reference *)
let dim_usage_of_intrinsic_ref (ref : intrinsic_ref) =
  match ref with
  | IntrinsicRef (_, name) -> dim_usage_of_name name
  | CorePrimitiveRef name -> dim_usage_of_name name

(** Analyze expression for dimension usage *)
let rec expr_dim_usage (te : texpr) : dim_usage =
  match te.te with
  (* Variable - might be a thread index *)
  | TEVar (name, _) -> dim_usage_of_name name
  (* Intrinsic constant - check for thread/block/grid *)
  | TEIntrinsicConst ref -> dim_usage_of_intrinsic_ref ref
  (* Intrinsic function - check name and args *)
  | TEIntrinsicFun (ref, _, args) ->
      let ref_usage = dim_usage_of_intrinsic_ref ref in
      let args_usage =
        List.fold_left
          merge_dim_usage
          empty_dim_usage
          (List.map expr_dim_usage args)
      in
      merge_dim_usage ref_usage args_usage
  (* Shared memory allocation *)
  | TELetShared (_, _, _, size_opt, body) ->
      let size_usage =
        match size_opt with
        | None -> empty_dim_usage
        | Some e -> expr_dim_usage e
      in
      let body_usage = expr_dim_usage body in
      merge_dim_usage {size_usage with uses_shared_mem = true} body_usage
  (* Recursive cases *)
  | TEBinop (_, a, b) -> merge_dim_usage (expr_dim_usage a) (expr_dim_usage b)
  | TEUnop (_, e) -> expr_dim_usage e
  | TEIf (c, t, e_opt) -> (
      let usage = merge_dim_usage (expr_dim_usage c) (expr_dim_usage t) in
      match e_opt with
      | None -> usage
      | Some e -> merge_dim_usage usage (expr_dim_usage e))
  | TEWhile (c, b) -> merge_dim_usage (expr_dim_usage c) (expr_dim_usage b)
  | TEFor (_, _, lo, hi, _, body) ->
      merge_dim_usage
        (expr_dim_usage lo)
        (merge_dim_usage (expr_dim_usage hi) (expr_dim_usage body))
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) ->
      merge_dim_usage (expr_dim_usage v) (expr_dim_usage b)
  | TESeq es ->
      List.fold_left
        merge_dim_usage
        empty_dim_usage
        (List.map expr_dim_usage es)
  | TEApp (f, args) ->
      List.fold_left
        merge_dim_usage
        (expr_dim_usage f)
        (List.map expr_dim_usage args)
  | TEVecGet (v, i) | TEArrGet (v, i) ->
      merge_dim_usage (expr_dim_usage v) (expr_dim_usage i)
  | TEVecSet (v, i, x) | TEArrSet (v, i, x) ->
      merge_dim_usage
        (expr_dim_usage v)
        (merge_dim_usage (expr_dim_usage i) (expr_dim_usage x))
  | TEFieldGet (e, _, _) -> expr_dim_usage e
  | TEFieldSet (e, _, _, x) ->
      merge_dim_usage (expr_dim_usage e) (expr_dim_usage x)
  | TEAssign (_, _, e) -> expr_dim_usage e
  | TERecord (_, fields) ->
      List.fold_left
        merge_dim_usage
        empty_dim_usage
        (List.map (fun (_, e) -> expr_dim_usage e) fields)
  | TETuple es ->
      List.fold_left
        merge_dim_usage
        empty_dim_usage
        (List.map expr_dim_usage es)
  | TEMatch (s, cases) ->
      List.fold_left
        merge_dim_usage
        (expr_dim_usage s)
        (List.map (fun (_, e) -> expr_dim_usage e) cases)
  | TEConstr (_, _, arg) -> (
      match arg with None -> empty_dim_usage | Some e -> expr_dim_usage e)
  | TEReturn e -> expr_dim_usage e
  | TECreateArray (size, _, _) -> expr_dim_usage size
  | TEPragma (_, body) -> expr_dim_usage body
  | TESuperstep (_, _, step, cont) ->
      merge_dim_usage (expr_dim_usage step) (expr_dim_usage cont)
  | TEOpen (_, body) -> expr_dim_usage body
  | TELetRec (_, _, _, fn_body, cont) ->
      merge_dim_usage (expr_dim_usage fn_body) (expr_dim_usage cont)
  (* Terminal cases - no dimension usage *)
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEGlobalRef _ | TENative _ ->
      empty_dim_usage

(** Kernel execution strategy based on dimension analysis *)
type exec_strategy =
  | Simple1D  (** Only uses global_idx_x - can use simple parallel for *)
  | Simple2D  (** Only uses global_idx_x/y - can use nested loops *)
  | Simple3D  (** Uses all three dimensions with simple loops *)
  | FullState
      (** Uses block/thread indices or shared memory - needs full state *)

(** Determine the optimal execution strategy for a kernel *)
let kernel_exec_strategy (kernel : tkernel) : exec_strategy =
  (* First check if kernel uses barriers - if so, need full state for barrier function *)
  if kernel_uses_barriers kernel then FullState
  else begin
    (* Analyze module items *)
    let items_usage =
      List.fold_left
        merge_dim_usage
        empty_dim_usage
        (List.map
           (fun (item : tmodule_item) ->
             match item with
             | TMConst (_, _, _, e) -> expr_dim_usage e
             | TMFun (_, _, _, e) -> expr_dim_usage e)
           kernel.tkern_module_items)
    in
    (* Analyze kernel body *)
    let body_usage = expr_dim_usage kernel.tkern_body in
    let usage = merge_dim_usage items_usage body_usage in

    (* If uses block/thread indices directly, or shared memory, need full state *)
    if
      usage.uses_thread_idx || usage.uses_block_idx || usage.uses_block_dim
      || usage.uses_grid_dim || usage.uses_shared_mem
    then FullState
      (* Otherwise, can use simplified loops based on dimensions used *)
    else if usage.uses_z then Simple3D
    else if usage.uses_y then Simple2D
    else if usage.uses_x then Simple1D
    else
      (* No dimensions used at all - still need to run at least once per global thread *)
      Simple1D
  end
