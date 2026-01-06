(******************************************************************************
 * Sarek PPX - Tail Recursion Analysis and Elimination
 *
 * This module detects tail-recursive functions and transforms them into
 * loops for GPU execution. Non-tail recursion can be inlined using
 * pragma ["sarek.inline N"].
 *
 * Pragmas:
 * - pragma ["unroll N"] body: For tail recursion transformed to loops.
 *   Passes #pragma unroll N to the GPU compiler.
 * - pragma ["sarek.inline N"] body: For non-tail recursion.
 *   PPX mechanically inlines N times, then verifies no calls remain.
 ******************************************************************************)

open Sarek_typed_ast
open Sarek_types

(** {1 Recursion Analysis} *)

(** Information about a recursive function *)
type recursion_info = {
  ri_name : string;  (** Function name *)
  ri_is_tail : bool;  (** All recursive calls are in tail position *)
  ri_max_depth : int option;  (** Bounded depth if detectable *)
  ri_call_count : int;  (** Number of recursive call sites *)
  ri_in_loop : bool;  (** Has recursive calls inside loops *)
}

(** Check if an expression is a call to the named function *)
let is_self_call (fname : string) (expr : texpr) : bool =
  match expr.te with TEVar (name, _) -> name = fname | _ -> false

(** Check if a function application is a recursive call *)
let is_recursive_call (fname : string) (expr : texpr) : bool =
  match expr.te with TEApp (fn, _) -> is_self_call fname fn | _ -> false

(** Count recursive calls in an expression *)
let rec count_recursive_calls (fname : string) (expr : texpr) : int =
  match expr.te with
  | TEApp (fn, args) ->
      let self = if is_self_call fname fn then 1 else 0 in
      let in_fn = count_recursive_calls fname fn in
      let in_args =
        List.fold_left (fun acc a -> acc + count_recursive_calls fname a) 0 args
      in
      self + in_fn + in_args
  | TELet (_, _, value, body) | TELetMut (_, _, value, body) ->
      count_recursive_calls fname value + count_recursive_calls fname body
  | TELetShared (_, _, _, _, body) -> count_recursive_calls fname body
  | TEIf (cond, then_, else_opt) -> (
      count_recursive_calls fname cond
      + count_recursive_calls fname then_
      +
      match else_opt with
      | Some e -> count_recursive_calls fname e
      | None -> 0)
  | TEFor (_, _, lo, hi, _, body) ->
      count_recursive_calls fname lo
      + count_recursive_calls fname hi
      + count_recursive_calls fname body
  | TEWhile (cond, body) ->
      count_recursive_calls fname cond + count_recursive_calls fname body
  | TESeq exprs ->
      List.fold_left (fun acc e -> acc + count_recursive_calls fname e) 0 exprs
  | TEBinop (_, a, b) ->
      count_recursive_calls fname a + count_recursive_calls fname b
  | TEUnop (_, e) -> count_recursive_calls fname e
  | TEVecGet (v, i) | TEArrGet (v, i) ->
      count_recursive_calls fname v + count_recursive_calls fname i
  | TEVecSet (v, i, x) | TEArrSet (v, i, x) ->
      count_recursive_calls fname v
      + count_recursive_calls fname i
      + count_recursive_calls fname x
  | TEFieldGet (r, _, _) -> count_recursive_calls fname r
  | TEFieldSet (r, _, _, v) ->
      count_recursive_calls fname r + count_recursive_calls fname v
  | TEMatch (scrut, cases) ->
      count_recursive_calls fname scrut
      + List.fold_left
          (fun acc (_, body) -> acc + count_recursive_calls fname body)
          0
          cases
  | TETuple es ->
      List.fold_left (fun acc e -> acc + count_recursive_calls fname e) 0 es
  | TERecord (_, fields) ->
      List.fold_left
        (fun acc (_, e) -> acc + count_recursive_calls fname e)
        0
        fields
  | TEConstr (_, _, arg) -> (
      match arg with Some e -> count_recursive_calls fname e | None -> 0)
  | TEAssign (_, _, v) -> count_recursive_calls fname v
  | TEReturn e -> count_recursive_calls fname e
  | TESuperstep (_, _, step, cont) ->
      count_recursive_calls fname step + count_recursive_calls fname cont
  | TEPragma (_, body) -> count_recursive_calls fname body
  | TEOpen (_, body) -> count_recursive_calls fname body
  | TECreateArray (size, _, _) -> count_recursive_calls fname size
  | TEIntrinsicFun (_, _, args) ->
      List.fold_left (fun acc a -> acc + count_recursive_calls fname a) 0 args
  | TELetRec (_, _, _params, fn_body, cont) ->
      (* Count calls in function body and continuation *)
      let in_body = count_recursive_calls fname fn_body in
      let in_cont = count_recursive_calls fname cont in
      in_body + in_cont
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ ->
      0

(** Check if all recursive calls are in tail position *)
let is_tail_recursive (fname : string) (expr : texpr) : bool =
  (* Helper: check if expression is in tail position *)
  let rec check_tail expr =
    match expr.te with
    | TEApp (fn, args) when is_self_call fname fn ->
        (* Recursive call in tail position - but args must not contain recursive calls! *)
        let in_args =
          List.fold_left
            (fun acc a -> acc + count_recursive_calls fname a)
            0
            args
        in
        in_args = 0
    | TEApp (fn, args) ->
        (* Non-self call: recursive calls in args are NOT tail *)
        let in_fn = count_recursive_calls fname fn in
        let in_args =
          List.fold_left
            (fun acc a -> acc + count_recursive_calls fname a)
            0
            args
        in
        in_fn = 0 && in_args = 0
    | TEIf (cond, then_, else_opt) -> (
        (* Cond is not tail, branches are tail *)
        count_recursive_calls fname cond = 0
        && check_tail then_
        && match else_opt with Some e -> check_tail e | None -> true)
    | TELet (_, _, value, body) | TELetMut (_, _, value, body) ->
        (* Value is not tail, body is tail *)
        count_recursive_calls fname value = 0 && check_tail body
    | TELetShared (_, _, _, _, body) -> check_tail body
    | TESeq exprs -> (
        match List.rev exprs with
        | [] -> true
        | last :: rest ->
            (* Only last expression is in tail position *)
            List.for_all (fun e -> count_recursive_calls fname e = 0) rest
            && check_tail last)
    | TEFor (_, _, lo, hi, _, body) ->
        (* Recursive calls in loops are never tail *)
        count_recursive_calls fname lo = 0
        && count_recursive_calls fname hi = 0
        && count_recursive_calls fname body = 0
    | TEWhile (cond, body) ->
        count_recursive_calls fname cond = 0
        && count_recursive_calls fname body = 0
    | TEMatch (scrut, cases) ->
        count_recursive_calls fname scrut = 0
        && List.for_all (fun (_, body) -> check_tail body) cases
    | TEReturn e -> check_tail e
    | TESuperstep (_, _, step, cont) ->
        count_recursive_calls fname step = 0 && check_tail cont
    | TEPragma (_, body) -> check_tail body
    | TEOpen (_, body) -> check_tail body
    | _ ->
        (* Other expressions: no recursive calls allowed for tail recursion *)
        count_recursive_calls fname expr = 0
  in
  check_tail expr

(** Check if recursive calls occur inside loops *)
let rec has_recursion_in_loops (fname : string) (expr : texpr) : bool =
  match expr.te with
  | TEFor (_, _, lo, hi, _, body) ->
      count_recursive_calls fname body > 0
      || has_recursion_in_loops fname lo
      || has_recursion_in_loops fname hi
      || has_recursion_in_loops fname body
  | TEWhile (cond, body) ->
      count_recursive_calls fname body > 0
      || has_recursion_in_loops fname cond
      || has_recursion_in_loops fname body
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) ->
      has_recursion_in_loops fname v || has_recursion_in_loops fname b
  | TELetShared (_, _, _, _, b) -> has_recursion_in_loops fname b
  | TEIf (c, t, e) -> (
      has_recursion_in_loops fname c
      || has_recursion_in_loops fname t
      || match e with Some x -> has_recursion_in_loops fname x | None -> false)
  | TESeq es -> List.exists (has_recursion_in_loops fname) es
  | TEApp (fn, args) ->
      has_recursion_in_loops fname fn
      || List.exists (has_recursion_in_loops fname) args
  | TEBinop (_, a, b) ->
      has_recursion_in_loops fname a || has_recursion_in_loops fname b
  | TEUnop (_, e) -> has_recursion_in_loops fname e
  | TEMatch (s, cases) ->
      has_recursion_in_loops fname s
      || List.exists (fun (_, b) -> has_recursion_in_loops fname b) cases
  | TESuperstep (_, _, step, cont) ->
      has_recursion_in_loops fname step || has_recursion_in_loops fname cont
  | _ -> false

(** Detect bounded recursion depth from the code structure.

    This is a conservative analysis that only returns Some when we can prove the
    recursion is bounded. Currently disabled as the analysis is too imprecise.

    TODO: Implement proper termination analysis that tracks:
    - Which parameter is the "depth" parameter (must increase towards bound)
    - That recursive calls increment that parameter
    - That the bound is reached before max_inline_limit

    For now, we only support tail recursion which is transformed to loops. *)
let detect_bounded_depth (_fname : string) (_body : texpr) : int option =
  (* Disabled - the simple heuristic was too imprecise and caused infinite
     inlining for patterns like fibonacci where n decreases rather than
     a depth parameter that increases. *)
  None

(** Analyze a function for recursion patterns *)
let analyze_recursion (fname : string) (body : texpr) : recursion_info =
  let call_count = count_recursive_calls fname body in
  let is_tail = call_count > 0 && is_tail_recursive fname body in
  let in_loop = has_recursion_in_loops fname body in
  let max_depth =
    if call_count > 0 && not is_tail then detect_bounded_depth fname body
    else None
  in
  {
    ri_name = fname;
    ri_is_tail = is_tail;
    ri_max_depth = max_depth;
    ri_call_count = call_count;
    ri_in_loop = in_loop;
  }

(** {1 Tail Recursion Elimination} *)

(** Fresh variable ID generator for transformation (thread-safe) *)
let transform_var_counter = Atomic.make 0

let fresh_transform_id () = Atomic.fetch_and_add transform_var_counter 1

(** Transform a tail-recursive function into a loop.

    Uses a simple continue-flag approach suitable for GPU code:
    - _continue: bool mutable, starts true
    - _result: mutable of result type, stores the return value
    - Loop while _continue
    - Base case: _result := value; _continue := false
    - Recursive call: update loop vars (continue stays true) *)
let eliminate_tail_recursion (fname : string) (params : tparam list)
    (body : texpr) (loc : Sarek_ast.loc) : texpr =
  (* Create loop variable IDs for each parameter - use __name to avoid C redeclaration *)
  let loop_vars =
    List.map
      (fun p ->
        (p.tparam_name, p.tparam_id, fresh_transform_id (), p.tparam_type))
      params
  in

  (* Create result and continue variables *)
  let result_id = fresh_transform_id () in
  let continue_id = fresh_transform_id () in
  let result_ty = body.ty in

  (* Helper: substitute original param references with loop var references in an expression *)
  let rec substitute_params expr =
    match expr.te with
    | TEVar (name, id) -> (
        (* Check if this is a reference to one of our function parameters *)
        match
          List.find_opt
            (fun (n, orig_id, _, _) -> n = name && orig_id = id)
            loop_vars
        with
        | Some (n, _, loop_id, _) ->
            (* Replace with reference to loop variable *)
            {expr with te = TEVar ("__" ^ n, loop_id)}
        | None -> expr)
    | TELet (n, id, v, b) ->
        {expr with te = TELet (n, id, substitute_params v, substitute_params b)}
    | TELetMut (n, id, v, b) ->
        {
          expr with
          te = TELetMut (n, id, substitute_params v, substitute_params b);
        }
    | TEIf (c, t, e) ->
        {
          expr with
          te =
            TEIf
              ( substitute_params c,
                substitute_params t,
                Option.map substitute_params e );
        }
    | TESeq es -> {expr with te = TESeq (List.map substitute_params es)}
    | TEApp (fn, args) ->
        {
          expr with
          te = TEApp (substitute_params fn, List.map substitute_params args);
        }
    | TEBinop (op, a, b) ->
        {expr with te = TEBinop (op, substitute_params a, substitute_params b)}
    | TEUnop (op, e) -> {expr with te = TEUnop (op, substitute_params e)}
    | TEWhile (c, b) ->
        {expr with te = TEWhile (substitute_params c, substitute_params b)}
    | TEFor (v, id, lo, hi, dir, b) ->
        {
          expr with
          te =
            TEFor
              ( v,
                id,
                substitute_params lo,
                substitute_params hi,
                dir,
                substitute_params b );
        }
    | TEVecGet (v, i) ->
        {expr with te = TEVecGet (substitute_params v, substitute_params i)}
    | TEVecSet (v, i, x) ->
        {
          expr with
          te =
            TEVecSet
              (substitute_params v, substitute_params i, substitute_params x);
        }
    | TEArrGet (v, i) ->
        {expr with te = TEArrGet (substitute_params v, substitute_params i)}
    | TEArrSet (v, i, x) ->
        {
          expr with
          te =
            TEArrSet
              (substitute_params v, substitute_params i, substitute_params x);
        }
    | TEMatch (s, cases) ->
        {
          expr with
          te =
            TEMatch
              ( substitute_params s,
                List.map (fun (p, b) -> (p, substitute_params b)) cases );
        }
    | TEReturn e -> {expr with te = TEReturn (substitute_params e)}
    | TEAssign (n, id, v) ->
        {expr with te = TEAssign (n, id, substitute_params v)}
    | TETuple es -> {expr with te = TETuple (List.map substitute_params es)}
    | TERecord (n, fs) ->
        {
          expr with
          te = TERecord (n, List.map (fun (f, e) -> (f, substitute_params e)) fs);
        }
    | TEConstr (t, c, arg) ->
        {expr with te = TEConstr (t, c, Option.map substitute_params arg)}
    | TEFieldGet (r, t, f) ->
        {expr with te = TEFieldGet (substitute_params r, t, f)}
    | TEFieldSet (r, t, f, v) ->
        {
          expr with
          te = TEFieldSet (substitute_params r, t, f, substitute_params v);
        }
    | TEOpen (m, b) -> {expr with te = TEOpen (m, substitute_params b)}
    | TEPragma (p, b) -> {expr with te = TEPragma (p, substitute_params b)}
    | TECreateArray (sz, ty, mem) ->
        {expr with te = TECreateArray (substitute_params sz, ty, mem)}
    | TEIntrinsicFun (r, c, args) ->
        {expr with te = TEIntrinsicFun (r, c, List.map substitute_params args)}
    | TELetShared (n, id, sz, ty, b) ->
        {expr with te = TELetShared (n, id, sz, ty, substitute_params b)}
    | TESuperstep (s, n, step, cont) ->
        {
          expr with
          te = TESuperstep (s, n, substitute_params step, substitute_params cont);
        }
    | TELetRec (n, id, ps, fb, c) ->
        {
          expr with
          te = TELetRec (n, id, ps, substitute_params fb, substitute_params c);
        }
    | _ -> expr
  in

  (* Helper: create assignment to loop variable *)
  let mk_loop_assign name id value =
    {te = TEAssign ("__" ^ name, id, value); ty = t_unit; te_loc = loc}
  in

  (* Helper: create assignment to _continue := false (to break from loop) *)
  let mk_break () =
    {
      te =
        TEAssign
          ( "_continue",
            continue_id,
            {te = TEBool false; ty = t_bool; te_loc = loc} );
      ty = t_unit;
      te_loc = loc;
    }
  in

  (* Helper: create assignment to _result *)
  let mk_result_assign value =
    {te = TEAssign ("_result", result_id, value); ty = t_unit; te_loc = loc}
  in

  (* Transform the body: replace recursive calls with assignments *)
  let rec transform expr =
    match expr.te with
    | TEApp (fn, args) when is_self_call fname fn ->
        (* Replace recursive call with assignment to loop variables.
           _continue stays true, so loop continues.

           IMPORTANT: We must evaluate all arguments BEFORE assigning them,
           otherwise `gcd b (a mod b)` would become:
             __a := __b;
             __b := __a mod __b;  // Wrong! Uses new __a, not old
           Instead we generate:
             let _tmp_0 = __b in
             let _tmp_1 = __a mod __b in
             __a := _tmp_0;
             __b := _tmp_1;
        *)
        (* Create temporary variables for each argument *)
        let temps =
          List.mapi
            (fun i arg ->
              let tmp_id = fresh_transform_id () in
              let tmp_name = "_tmp_" ^ string_of_int i in
              (tmp_name, tmp_id, arg))
            args
        in
        (* Create assignments from temps to loop vars *)
        let assigns =
          List.map2
            (fun (name, _orig_id, loop_id, _ty) (tmp_name, tmp_id, arg) ->
              let tmp_ref =
                {te = TEVar (tmp_name, tmp_id); ty = arg.ty; te_loc = loc}
              in
              mk_loop_assign name loop_id tmp_ref)
            loop_vars
            temps
        in
        (* Wrap assigns in let bindings for temps *)
        let body = {te = TESeq assigns; ty = t_unit; te_loc = expr.te_loc} in
        List.fold_right
          (fun (tmp_name, tmp_id, arg) acc ->
            {te = TELet (tmp_name, tmp_id, arg, acc); ty = t_unit; te_loc = loc})
          temps
          body
    | TEIf (cond, then_, else_opt) ->
        (* Only transform branches - condition is NOT in tail position *)
        {
          expr with
          te =
            TEIf
              ( cond,
                (* Keep condition as-is *)
                transform then_,
                Option.map transform else_opt );
        }
    | TELet (name, id, value, body_expr) ->
        (* Only transform body - value binding is NOT in tail position *)
        {expr with te = TELet (name, id, value, transform body_expr)}
    | TELetMut (name, id, value, body_expr) ->
        {expr with te = TELetMut (name, id, value, transform body_expr)}
    | TESeq exprs -> (
        (* Only the last expression is in tail position *)
        match List.rev exprs with
        | [] -> expr
        | last :: rest ->
            let transformed_last = transform last in
            {expr with te = TESeq (List.rev (transformed_last :: rest))})
    | TEMatch (scrut, cases) ->
        (* Only transform branch bodies - scrutinee is NOT in tail position *)
        {
          expr with
          te =
            TEMatch
              ( scrut,
                (* Keep scrutinee as-is *)
                List.map
                  (fun (pat, body_expr) -> (pat, transform body_expr))
                  cases );
        }
    | TEReturn e ->
        (* Return becomes: _result := e; _continue := false *)
        {
          te = TESeq [mk_result_assign (transform e); mk_break ()];
          ty = t_unit;
          te_loc = loc;
        }
    | _ ->
        (* For non-tail expressions that return a value, wrap as result assignment *)
        if count_recursive_calls fname expr = 0 then
          (* Base case: _result := expr; _continue := false *)
          {
            te = TESeq [mk_result_assign expr; mk_break ()];
            ty = t_unit;
            te_loc = loc;
          }
        else expr (* Keep as-is, will be handled by recursion *)
  in

  (* Build the loop structure *)

  (* 1. Initialize loop variables from parameters - use __name prefix to avoid C redeclaration *)
  let init_loop_vars =
    List.map
      (fun (name, orig_id, loop_id, ty) ->
        let param_ref = {te = TEVar (name, orig_id); ty; te_loc = loc} in
        ("__" ^ name, loop_id, param_ref, ty))
      loop_vars
  in

  (* 2. Initialize _continue to true *)
  let init_continue =
    ( "_continue",
      continue_id,
      {te = TEBool true; ty = t_bool; te_loc = loc},
      t_bool )
  in

  (* 3. Initialize _result to a default value of the appropriate type.
     This value will be overwritten before use, but we need a valid initializer. *)
  let init_result =
    let rec default_for_type ty =
      match ty with
      | TPrim TInt32 ->
          (* int/int32 use TEInt with t_int32 type *)
          {te = TEInt 0; ty = result_ty; te_loc = loc}
      | TReg Int -> {te = TEInt 0; ty = result_ty; te_loc = loc}
      | TReg Int64 -> {te = TEInt64 0L; ty = result_ty; te_loc = loc}
      | TReg Float32 -> {te = TEFloat 0.0; ty = result_ty; te_loc = loc}
      | TReg Float64 -> {te = TEDouble 0.0; ty = result_ty; te_loc = loc}
      | TReg Char -> {te = TEInt 0; ty = result_ty; te_loc = loc}
      | TReg (Custom _) -> {te = TEInt 0; ty = result_ty; te_loc = loc}
      | TPrim TBool -> {te = TEBool false; ty = result_ty; te_loc = loc}
      | TPrim TUnit -> {te = TEUnit; ty = result_ty; te_loc = loc}
      | TVar {contents = Link t} -> default_for_type t
      | _ ->
          (* For other types, use TEInt 0 as a fallback - it's just a placeholder *)
          {te = TEInt 0; ty = result_ty; te_loc = loc}
    in
    let init_val = default_for_type result_ty in
    ("_result", result_id, init_val, result_ty)
  in

  (* 4. Transform body - first substitute param refs, then transform recursive calls *)
  let transformed_body = substitute_params (transform body) in

  (* 5. While loop: while _continue do transformed_body done *)
  let loop_cond =
    {te = TEVar ("_continue", continue_id); ty = t_bool; te_loc = loc}
  in

  let while_loop =
    {te = TEWhile (loop_cond, transformed_body); ty = t_unit; te_loc = loc}
  in

  (* 6. Return _result *)
  let return_result =
    {te = TEVar ("_result", result_id); ty = result_ty; te_loc = loc}
  in

  (* 7. Wrap everything in let bindings *)
  let wrap_lets body_expr bindings =
    List.fold_right
      (fun (name, id, init, _ty) acc ->
        {te = TELetMut (name, id, init, acc); ty = acc.ty; te_loc = loc})
      bindings
      body_expr
  in

  (* Also need spoc_prof_cond for Kirc's while loop generation *)
  let spoc_prof_cond_id = fresh_transform_id () in
  let init_spoc_prof_cond =
    ( "spoc_prof_cond",
      spoc_prof_cond_id,
      {te = TEBool true; ty = t_bool; te_loc = loc},
      t_bool )
  in

  let final_body =
    {te = TESeq [while_loop; return_result]; ty = result_ty; te_loc = loc}
  in
  (* Order: loop vars first, then continue, then result, then spoc_prof_cond (innermost) *)
  wrap_lets
    final_body
    (init_spoc_prof_cond :: init_result :: init_continue :: init_loop_vars)

(** {1 Bounded Recursion Inlining} *)

(** Inline bounded recursion up to a maximum depth. This is used for
    non-tail-recursive functions with known bounds.

    Note: max_depth is the value from the termination condition (e.g., if depth
    >= 4). We inline one extra level to ensure the base case is properly
    expanded. *)
let inline_bounded_recursion (fname : string) (params : tparam list)
    (body : texpr) (max_depth : int) (_loc : Sarek_ast.loc) : texpr =
  (* Add some buffer to ensure base cases are fully expanded *)
  let effective_depth = max_depth + 2 in
  let rec inline depth expr =
    if depth > effective_depth then
      (* Exceeded depth - this should not happen if depth detection is correct.
         Return the expression unchanged as a safety fallback. *)
      expr
    else
      match expr.te with
      | TEApp (fn, args) when is_self_call fname fn ->
          (* Inline: substitute params with args and recurse *)
          let subst = List.map2 (fun p a -> (p.tparam_name, a)) params args in
          let inlined = substitute subst body in
          inline (depth + 1) inlined
      | TEIf (c, t, e) ->
          {
            expr with
            te =
              TEIf (inline depth c, inline depth t, Option.map (inline depth) e);
          }
      | TELet (n, id, v, b) ->
          {expr with te = TELet (n, id, inline depth v, inline depth b)}
      | TELetMut (n, id, v, b) ->
          {expr with te = TELetMut (n, id, inline depth v, inline depth b)}
      | TESeq es -> {expr with te = TESeq (List.map (inline depth) es)}
      | TEApp (fn, args) ->
          {expr with te = TEApp (inline depth fn, List.map (inline depth) args)}
      | TEBinop (op, a, b) ->
          {expr with te = TEBinop (op, inline depth a, inline depth b)}
      | TEUnop (op, e) -> {expr with te = TEUnop (op, inline depth e)}
      | _ -> expr
  and substitute subst expr =
    match expr.te with
    | TEVar (name, _id) -> (
        match List.assoc_opt name subst with
        | Some replacement -> replacement
        | None -> expr)
    | TELet (n, id, v, b) ->
        let subst' = List.filter (fun (name, _) -> name <> n) subst in
        {expr with te = TELet (n, id, substitute subst v, substitute subst' b)}
    | TELetMut (n, id, v, b) ->
        let subst' = List.filter (fun (name, _) -> name <> n) subst in
        {
          expr with
          te = TELetMut (n, id, substitute subst v, substitute subst' b);
        }
    | TEIf (c, t, e) ->
        {
          expr with
          te =
            TEIf
              ( substitute subst c,
                substitute subst t,
                Option.map (substitute subst) e );
        }
    | TESeq es -> {expr with te = TESeq (List.map (substitute subst) es)}
    | TEApp (fn, args) ->
        {
          expr with
          te = TEApp (substitute subst fn, List.map (substitute subst) args);
        }
    | TEBinop (op, a, b) ->
        {expr with te = TEBinop (op, substitute subst a, substitute subst b)}
    | TEUnop (op, e) -> {expr with te = TEUnop (op, substitute subst e)}
    | TEFor (v, id, lo, hi, dir, body_expr) ->
        let subst' = List.filter (fun (name, _) -> name <> v) subst in
        {
          expr with
          te =
            TEFor
              ( v,
                id,
                substitute subst lo,
                substitute subst hi,
                dir,
                substitute subst' body_expr );
        }
    | _ -> expr
  in

  inline 0 body

(** {1 Pragma-Based Inlining} *)

(** Maximum number of AST nodes allowed after inlining. Prevents code explosion
    from exponential recursion like fib. *)
let max_inlined_nodes = 10000

(** Count AST nodes in an expression *)
let rec count_nodes (expr : texpr) : int =
  1
  +
  match expr.te with
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ ->
      0
  | TEApp (fn, args) ->
      count_nodes fn + List.fold_left (fun acc a -> acc + count_nodes a) 0 args
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) -> count_nodes v + count_nodes b
  | TELetShared (_, _, _, _, b) -> count_nodes b
  | TEIf (c, t, e) -> (
      count_nodes c + count_nodes t
      + match e with Some x -> count_nodes x | None -> 0)
  | TEFor (_, _, lo, hi, _, body) ->
      count_nodes lo + count_nodes hi + count_nodes body
  | TEWhile (c, b) -> count_nodes c + count_nodes b
  | TESeq es -> List.fold_left (fun acc e -> acc + count_nodes e) 0 es
  | TEBinop (_, a, b) -> count_nodes a + count_nodes b
  | TEUnop (_, e) -> count_nodes e
  | TEVecGet (v, i) | TEArrGet (v, i) -> count_nodes v + count_nodes i
  | TEVecSet (v, i, x) | TEArrSet (v, i, x) ->
      count_nodes v + count_nodes i + count_nodes x
  | TEFieldGet (r, _, _) -> count_nodes r
  | TEFieldSet (r, _, _, v) -> count_nodes r + count_nodes v
  | TEMatch (s, cases) ->
      count_nodes s
      + List.fold_left (fun acc (_, b) -> acc + count_nodes b) 0 cases
  | TETuple es -> List.fold_left (fun acc e -> acc + count_nodes e) 0 es
  | TERecord (_, fs) ->
      List.fold_left (fun acc (_, e) -> acc + count_nodes e) 0 fs
  | TEConstr (_, _, arg) -> (
      match arg with Some e -> count_nodes e | None -> 0)
  | TEAssign (_, _, v) -> count_nodes v
  | TEReturn e -> count_nodes e
  | TESuperstep (_, _, step, cont) -> count_nodes step + count_nodes cont
  | TEPragma (_, body) -> count_nodes body
  | TEOpen (_, body) -> count_nodes body
  | TECreateArray (sz, _, _) -> count_nodes sz
  | TEIntrinsicFun (_, _, args) ->
      List.fold_left (fun acc a -> acc + count_nodes a) 0 args
  | TELetRec (_, _, _, fb, c) -> count_nodes fb + count_nodes c

(** Parse pragma options to extract sarek.inline depth. Returns Some depth for
    "sarek.inline N", None otherwise. *)
let parse_sarek_inline_pragma (opts : string list) : int option =
  match opts with
  | [opt] -> (
      (* Handle "sarek.inline N" as a single string *)
      match String.split_on_char ' ' opt with
      | ["sarek.inline"; n] -> int_of_string_opt n
      | _ -> None)
  | ["sarek.inline"; n] -> int_of_string_opt n
  | _ -> None

(** Check if pragma is an unroll pragma *)
let is_unroll_pragma (opts : string list) : bool =
  match opts with
  | [opt] -> String.length opt >= 6 && String.sub opt 0 6 = "unroll"
  | "unroll" :: _ -> true
  | _ -> false

(** Substitute a variable in an expression with another expression *)
let rec subst_var (name : string) (id : int) (replacement : texpr) (e : texpr) :
    texpr =
  match e.te with
  | TEVar (n, i) when n = name && i = id -> replacement
  | TEVar _ -> e
  | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _ | TEBool _ | TEUnit
    ->
      e
  | TEApp (fn, args) ->
      {
        e with
        te =
          TEApp
            ( subst_var name id replacement fn,
              List.map (subst_var name id replacement) args );
      }
  | TELet (n, i, v, b) ->
      (* Don't substitute in body if this binding shadows our variable *)
      let v' = subst_var name id replacement v in
      let b' =
        if n = name && i = id then b else subst_var name id replacement b
      in
      {e with te = TELet (n, i, v', b')}
  | TELetMut (n, i, v, b) ->
      let v' = subst_var name id replacement v in
      let b' =
        if n = name && i = id then b else subst_var name id replacement b
      in
      {e with te = TELetMut (n, i, v', b')}
  | TEIf (c, t, el) ->
      {
        e with
        te =
          TEIf
            ( subst_var name id replacement c,
              subst_var name id replacement t,
              Option.map (subst_var name id replacement) el );
      }
  | TESeq es ->
      {e with te = TESeq (List.map (subst_var name id replacement) es)}
  | TEFor (v, vid, lo, hi, dir, b) ->
      let lo' = subst_var name id replacement lo in
      let hi' = subst_var name id replacement hi in
      let b' =
        if v = name && vid = id then b else subst_var name id replacement b
      in
      {e with te = TEFor (v, vid, lo', hi', dir, b')}
  | TEWhile (c, b) ->
      {
        e with
        te =
          TEWhile
            (subst_var name id replacement c, subst_var name id replacement b);
      }
  | TEBinop (op, a, b) ->
      {
        e with
        te =
          TEBinop
            ( op,
              subst_var name id replacement a,
              subst_var name id replacement b );
      }
  | TEUnop (op, x) -> {e with te = TEUnop (op, subst_var name id replacement x)}
  | TEVecGet (v, i) ->
      {
        e with
        te =
          TEVecGet
            (subst_var name id replacement v, subst_var name id replacement i);
      }
  | TEVecSet (v, i, x) ->
      {
        e with
        te =
          TEVecSet
            ( subst_var name id replacement v,
              subst_var name id replacement i,
              subst_var name id replacement x );
      }
  | TEArrGet (v, i) ->
      {
        e with
        te =
          TEArrGet
            (subst_var name id replacement v, subst_var name id replacement i);
      }
  | TEArrSet (v, i, x) ->
      {
        e with
        te =
          TEArrSet
            ( subst_var name id replacement v,
              subst_var name id replacement i,
              subst_var name id replacement x );
      }
  | TEFieldGet (r, t, f) ->
      {e with te = TEFieldGet (subst_var name id replacement r, t, f)}
  | TEFieldSet (r, t, f, v) ->
      {
        e with
        te =
          TEFieldSet
            ( subst_var name id replacement r,
              t,
              f,
              subst_var name id replacement v );
      }
  | TEMatch (s, cases) ->
      {
        e with
        te =
          TEMatch
            ( subst_var name id replacement s,
              List.map
                (fun (p, b) -> (p, subst_var name id replacement b))
                cases );
      }
  | TETuple es ->
      {e with te = TETuple (List.map (subst_var name id replacement) es)}
  | TERecord (n, fs) ->
      {
        e with
        te =
          TERecord
            (n, List.map (fun (f, x) -> (f, subst_var name id replacement x)) fs);
      }
  | TEConstr (t, c, arg) ->
      {
        e with
        te = TEConstr (t, c, Option.map (subst_var name id replacement) arg);
      }
  | TEAssign (n, i, v) ->
      {e with te = TEAssign (n, i, subst_var name id replacement v)}
  | TEReturn x -> {e with te = TEReturn (subst_var name id replacement x)}
  | TESuperstep (s, n, step, cont) ->
      {
        e with
        te =
          TESuperstep
            ( s,
              n,
              subst_var name id replacement step,
              subst_var name id replacement cont );
      }
  | TEPragma (p, b) ->
      {e with te = TEPragma (p, subst_var name id replacement b)}
  | TEOpen (m, b) -> {e with te = TEOpen (m, subst_var name id replacement b)}
  | TECreateArray (sz, ty, mem) ->
      {e with te = TECreateArray (subst_var name id replacement sz, ty, mem)}
  | TEIntrinsicFun (r, c, args) ->
      {
        e with
        te = TEIntrinsicFun (r, c, List.map (subst_var name id replacement) args);
      }
  | TELetShared (n, i, sz, ty, b) ->
      let b' =
        if n = name && i = id then b else subst_var name id replacement b
      in
      {e with te = TELetShared (n, i, sz, ty, b')}
  | TELetRec (n, i, ps, fb, c) ->
      let fb' =
        if n = name && i = id then fb else subst_var name id replacement fb
      in
      let c' =
        if n = name && i = id then c else subst_var name id replacement c
      in
      {e with te = TELetRec (n, i, ps, fb', c')}
  (* These don't contain variable references to substitute *)
  | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ -> e

(** Substitute all occurrences of a recursive call with the function body. This
    is one step of mechanical inlining.

    IMPORTANT: We do NOT recurse into the substituted body - that happens in the
    next inlining round. This ensures we do exactly N rounds of inlining.

    Instead of creating let-bindings (which generate statement expressions that
    OpenCL doesn't support), we directly substitute parameter values into the
    body. *)
let substitute_recursive_calls (fname : string) (params : tparam list)
    (body : texpr) (expr : texpr) : texpr =
  let rec subst e =
    match e.te with
    | TEApp (fn, args) when is_self_call fname fn ->
        (* Direct substitution: replace params with args in body.
           Note: we substitute in args (they're from the current expression)
           but then substitute those into body. *)
        let substituted_args = List.map subst args in
        (* Substitute each parameter with its corresponding argument *)
        List.fold_left2
          (fun acc param arg ->
            subst_var param.tparam_name param.tparam_id arg acc)
          body
          params
          substituted_args
    | TEApp (fn, args) -> {e with te = TEApp (subst fn, List.map subst args)}
    | TELet (n, id, v, b) -> {e with te = TELet (n, id, subst v, subst b)}
    | TELetMut (n, id, v, b) -> {e with te = TELetMut (n, id, subst v, subst b)}
    | TEIf (c, t, el) ->
        {e with te = TEIf (subst c, subst t, Option.map subst el)}
    | TESeq es -> {e with te = TESeq (List.map subst es)}
    | TEFor (v, id, lo, hi, dir, b) ->
        {e with te = TEFor (v, id, subst lo, subst hi, dir, subst b)}
    | TEWhile (c, b) -> {e with te = TEWhile (subst c, subst b)}
    | TEBinop (op, a, b) -> {e with te = TEBinop (op, subst a, subst b)}
    | TEUnop (op, x) -> {e with te = TEUnop (op, subst x)}
    | TEVecGet (v, i) -> {e with te = TEVecGet (subst v, subst i)}
    | TEVecSet (v, i, x) -> {e with te = TEVecSet (subst v, subst i, subst x)}
    | TEArrGet (v, i) -> {e with te = TEArrGet (subst v, subst i)}
    | TEArrSet (v, i, x) -> {e with te = TEArrSet (subst v, subst i, subst x)}
    | TEFieldGet (r, t, f) -> {e with te = TEFieldGet (subst r, t, f)}
    | TEFieldSet (r, t, f, v) ->
        {e with te = TEFieldSet (subst r, t, f, subst v)}
    | TEMatch (s, cases) ->
        {
          e with
          te = TEMatch (subst s, List.map (fun (p, b) -> (p, subst b)) cases);
        }
    | TETuple es -> {e with te = TETuple (List.map subst es)}
    | TERecord (n, fs) ->
        {e with te = TERecord (n, List.map (fun (f, x) -> (f, subst x)) fs)}
    | TEConstr (t, c, arg) ->
        {e with te = TEConstr (t, c, Option.map subst arg)}
    | TEAssign (n, id, v) -> {e with te = TEAssign (n, id, subst v)}
    | TEReturn x -> {e with te = TEReturn (subst x)}
    | TESuperstep (s, n, step, cont) ->
        {e with te = TESuperstep (s, n, subst step, subst cont)}
    | TEPragma (p, b) -> {e with te = TEPragma (p, subst b)}
    | TEOpen (m, b) -> {e with te = TEOpen (m, subst b)}
    | TECreateArray (sz, ty, mem) ->
        {e with te = TECreateArray (subst sz, ty, mem)}
    | TEIntrinsicFun (r, c, args) ->
        {e with te = TEIntrinsicFun (r, c, List.map subst args)}
    | TELetShared (n, id, sz, ty, b) ->
        {e with te = TELetShared (n, id, sz, ty, subst b)}
    | TELetRec (n, id, ps, fb, c) ->
        {e with te = TELetRec (n, id, ps, subst fb, subst c)}
    | _ -> e
  in
  subst expr

(** Inline a recursive function N times using pragma ["sarek.inline N"]. Returns
    Ok new_expr if successful, Error message if failed.

    Note: After inlining, some calls may remain syntactically present but they
    will be in dead code branches (e.g., after the base case condition). We emit
    a warning but allow the code to proceed, trusting the GPU compiler to
    eliminate dead code. *)
let inline_with_pragma (fname : string) (params : tparam list) (body : texpr)
    (depth : int) : (texpr, string) result =
  Sarek_debug.log "inline_with_pragma '%s' depth=%d" fname depth ;
  let rec inline_n n expr =
    Sarek_debug.log "inline_n n=%d" n ;
    if n = 0 then (
      Sarek_debug.log "inline_n done (n=0)" ;
      expr)
    else (
      (* Check node count BEFORE substitution to avoid explosion *)
      Sarek_debug.log "counting nodes..." ;
      let nodes = count_nodes expr in
      Sarek_debug.log "nodes=%d" nodes ;
      if nodes > max_inlined_nodes then
        raise
          (Failure
             (Printf.sprintf
                "Inlining produced %d nodes (limit: %d). Reduce inline depth \
                 or use tail recursion."
                nodes
                max_inlined_nodes)) ;
      Sarek_debug.log "substituting..." ;
      let expr' = substitute_recursive_calls fname params body expr in
      Sarek_debug.log "substitution done, recursing" ;
      inline_n (n - 1) expr')
  in
  try
    Sarek_debug.log "starting inline_n" ;
    let result = inline_n depth body in
    Sarek_debug.log "inline_n finished" ;
    (* Check for remaining recursive calls *)
    let remaining = count_recursive_calls fname result in
    if remaining > 0 && Sarek_debug.enabled then (
      (* Warn but allow - these should be in dead code branches *)
      Format.eprintf
        "Sarek warning: %d recursive calls remain after %d inlines.@."
        remaining
        depth ;
      Format.eprintf
        "  These should be in unreachable branches (GPU will eliminate).@." ;
      Format.eprintf "  If you get runtime errors, increase the inline depth.@.") ;
    Ok result
  with Failure msg -> Error msg

(** {1 Kernel-Level Pass} *)

(** Maximum inline depth for bounded recursion. Beyond this, we refuse to
    compile as it would generate too much code. *)
let max_inline_limit = 16

(** Extract pragma options and inner body from a function body. Returns (Some
    opts, inner_body) if body is TEPragma, (None, body) otherwise. *)
let extract_pragma (body : texpr) : string list option * texpr =
  match body.te with
  | TEPragma (opts, inner) -> (Some opts, inner)
  | _ -> (None, body)

(** Transform all recursive module functions in a kernel:
    - Tail-recursive functions are transformed to loops
    - Non-tail recursion with pragma ["sarek.inline N"] is inlined N times
    - Validates pragma usage: unroll for tail, sarek.inline for non-tail

    This pass is run after type checking and before lowering to Kirc. *)
let transform_kernel (kernel : tkernel) : tkernel =
  Sarek_debug.log_enter "transform_kernel" ;
  let new_items =
    List.map
      (function
        | TMFun (name, is_rec, params, body) as orig -> (
            Sarek_debug.log "processing TMFun '%s'" name ;
            let pragma_opts, inner_body = extract_pragma body in
            let info = analyze_recursion name inner_body in
            if info.ri_call_count = 0 then
              (* Not recursive, leave as-is *)
              orig
            else if info.ri_is_tail then (
              (* Tail recursive - transform to loop *)
              (* Check for invalid pragma *)
              (match pragma_opts with
              | Some opts when Option.is_some (parse_sarek_inline_pragma opts)
                ->
                  Format.eprintf
                    "Sarek error: function '%s' is tail-recursive.@."
                    name ;
                  Format.eprintf
                    "  Use 'pragma [\"unroll N\"]' instead of 'sarek.inline'.@." ;
                  failwith "Cannot use sarek.inline on tail-recursive function"
              | _ -> ()) ;
              if Sarek_debug.enabled then
                Format.eprintf
                  "Sarek: transforming tail-recursive function '%s' to loop@."
                  name ;
              let new_body =
                eliminate_tail_recursion
                  name
                  params
                  inner_body
                  inner_body.te_loc
              in
              (* Preserve pragma wrapper if present (for unroll) *)
              let final_body =
                match pragma_opts with
                | Some opts -> {body with te = TEPragma (opts, new_body)}
                | None -> new_body
              in
              TMFun (name, is_rec, params, final_body))
            else
              (* Non-tail recursion *)
              (* Check for pragma *)
              match pragma_opts with
              | Some opts when is_unroll_pragma opts ->
                  Format.eprintf
                    "Sarek error: function '%s' is NOT tail-recursive.@."
                    name ;
                  Format.eprintf
                    "  Cannot use 'unroll' on non-tail recursion.@." ;
                  Format.eprintf
                    "  Use 'pragma [\"sarek.inline N\"]' to inline, or rewrite \
                     as tail-recursive.@." ;
                  failwith "Cannot use unroll on non-tail-recursive function"
              | Some opts -> (
                  match parse_sarek_inline_pragma opts with
                  | Some depth -> (
                      if Sarek_debug.enabled then
                        Format.eprintf
                          "Sarek: inlining recursive function '%s' %d times@."
                          name
                          depth ;
                      match inline_with_pragma name params inner_body depth with
                      | Ok new_body ->
                          if Sarek_debug.enabled then
                            Format.eprintf
                              "Sarek: successfully inlined '%s' (%d nodes)@."
                              name
                              (count_nodes new_body) ;
                          TMFun (name, is_rec, params, new_body)
                      | Error msg ->
                          Format.eprintf "Sarek error in '%s': %s@." name msg ;
                          failwith msg)
                  | None ->
                      (* Unknown pragma, warn and leave as-is *)
                      Format.eprintf
                        "Sarek warning: function '%s' is recursive but not \
                         tail-recursive@."
                        name ;
                      Format.eprintf
                        "  Use 'pragma [\"sarek.inline N\"]' to inline.@." ;
                      orig)
              | None ->
                  (* No pragma, unbounded non-tail recursion - error *)
                  Format.eprintf
                    "Sarek error: function '%s' is recursive but not \
                     tail-recursive@."
                    name ;
                  Format.eprintf
                    "  (recursive calls: %d, in loops: %b)@."
                    info.ri_call_count
                    info.ri_in_loop ;
                  Format.eprintf
                    "  Use 'pragma [\"sarek.inline N\"]' to inline, or rewrite \
                     as tail-recursive.@." ;
                  failwith
                    (Printf.sprintf
                       "Non-tail recursion in '%s' requires pragma \
                        [\"sarek.inline N\"]"
                       name))
        | item -> item)
      kernel.tkern_module_items
  in
  {kernel with tkern_module_items = new_items}
