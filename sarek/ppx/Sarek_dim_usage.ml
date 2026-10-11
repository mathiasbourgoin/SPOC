(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - barrier and dimension usage, native execution strategy
 *
 * Split out of Sarek_convergence (which includes this module, so the
 * [Sarek_convergence.*] names used by native code generation and the tests
 * are unchanged).
 ******************************************************************************)

open Sarek_typed_ast
open Sarek_env

(** Check if a kernel uses any barriers (explicit or implicit). Used for
    compile-time optimization of native kernel execution. *)
let rec expr_uses_barriers (te : texpr) : bool =
  match te.te with
  (* Superstep has implicit barrier *)
  | TESuperstep _ -> true
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
