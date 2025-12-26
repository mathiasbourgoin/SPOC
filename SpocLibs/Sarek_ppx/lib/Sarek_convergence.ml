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
  | TEIntrinsicFun (_, args) -> List.exists is_thread_varying args
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
  (* Default: assume uniform (minimize false positives) *)
  | TEIf _ | TEFor _ | TEWhile _ | TEMatch _ | TERecord _ | TEConstr _
  | TEReturn _ | TECreateArray _ | TEGlobalRef _ | TENative _ | TENativeFun _
  | TEPragma _ | TEVecSet _ | TEArrSet _ | TEFieldSet _ | TEAssign _ ->
      false

(** Check if an intrinsic reference is a barrier *)
let is_barrier_intrinsic (ref : intrinsic_ref) : bool = is_barrier_ref ref

(** Collect errors from convergence analysis *)
let rec check_expr ctx (te : texpr) : Sarek_error.error list =
  match te.te with
  (* Barrier calls - the key check *)
  | TEIntrinsicFun (ref, args) when is_barrier_intrinsic ref ->
      let arg_errors = List.concat_map (check_expr ctx) args in
      if ctx.mode = Diverged then
        Sarek_error.Barrier_in_diverged_flow te.te_loc :: arg_errors
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
  | TEVar _ | TEGlobalRef _ | TENative _ | TENativeFun _ | TEIntrinsicConst _ ->
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
  | TEIntrinsicFun (_, args) -> List.concat_map (check_expr ctx) args
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
  | TEIntrinsicFun (_, args) -> List.exists contains_diverging_control_flow args
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
  (* Terminal cases - no nested control flow *)
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEGlobalRef _ | TENative _ | TENativeFun _ | TEIntrinsicConst _ ->
      false

(** Check a module item *)
let check_module_item ctx (item : tmodule_item) : Sarek_error.error list =
  match item with
  | TMConst (_, _, _, e) -> check_expr ctx e
  | TMFun (_, _, e) -> check_expr ctx e

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
