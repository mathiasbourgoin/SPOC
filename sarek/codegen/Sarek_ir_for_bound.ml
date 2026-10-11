(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Evaluate a [for] upper bound once. See the .mli for the rule. *)

open Sarek_ir_types

(* [EVar] occurrences are lowered with [var_mutable = false] even when they name
   a [mut] variable, so mutability cannot be read off the reference. A variable
   is stable for a loop exactly when the loop body never assigns it. *)
let rec lvalue_root = function
  | LVar v -> Some v.var_name
  | LRecordField (lv, _) -> lvalue_root lv
  | LArrayElem _ | LArrayElemExpr _ -> None

let rec assigned_names (s : stmt) : string list =
  match s with
  | SAssign (lv, _) -> Option.to_list (lvalue_root lv)
  | SSeq stmts -> List.concat_map assigned_names stmts
  | SIf (_, t, e) ->
      assigned_names t @ Option.fold ~none:[] ~some:assigned_names e
  | SWhile (_, b)
  | SFor (_, _, _, _, b)
  | SLet (_, _, b)
  | SLetMut (_, _, b)
  | SPragma (_, b)
  | SBlock b ->
      assigned_names b
  | SMatch (_, cases) -> List.concat_map (fun (_, b) -> assigned_names b) cases
  | SReturn _ | SBarrier | SWarpBarrier | SExpr _ | SEmpty | SMemFence
  | SNative _ | SCoopmat _ ->
      []

(* The rule is syntactic: a variable is stable unless the loop body assigns it
   by name. That is sound only because a local function cannot capture a [mut]
   variable by reference (a helper that tries fails to compile, loudly), so
   nothing but an assignment in the body can change one. Revisit this rule if
   closures ever may mutate captured variables. *)
let rec stable ~assigned (e : expr) : bool =
  let go = stable ~assigned in
  match e with
  | EConst _ | EArrayLen _ -> true
  | EVar v -> not (List.mem v.var_name assigned)
  | EBinop (_, a, b) -> go a && go b
  | EUnop (_, a) | ECast (_, a) -> go a
  | EIntrinsic (_, name, []) ->
      List.mem name Sarek_ir_intrinsic_dispatch.thread_intrinsic_names
  | EIf (c, t, f) -> go c && go t && go f
  | _ -> false

let fresh_temp counter ~prefix ty =
  incr counter ;
  {
    var_name = Printf.sprintf "sarek_for_%s_%d" prefix !counter;
    var_id = 0;
    var_type = ty;
    var_mutable = false;
  }

(* The temporaries [walk] introduces in front of a loop. *)
let is_loop_temp v =
  let has p =
    String.length v.var_name >= String.length p
    && String.sub v.var_name 0 (String.length p) = p
  in
  has "sarek_for_hi_" || has "sarek_for_lo_"

let rec under_temps f = function
  | SLet (t, e, b) when is_loop_temp t -> SLet (t, e, under_temps f b)
  | s -> f s

(* OCaml evaluates the start, then the bound. Hoisting the bound in front of
   the loop must not move it ahead of a start that has effects or that the
   bound depends on, so such a start is hoisted first. *)
let hoist_bound counter v lo hi dir body =
  let ty = v.var_type in
  let lo_tmp, lo_expr =
    if stable ~assigned:[] lo then ([], lo)
    else
      let t = fresh_temp counter ~prefix:"lo" ty in
      ([(t, lo)], EVar t)
  in
  let hi_tmp = fresh_temp counter ~prefix:"hi" ty in
  List.fold_right
    (fun (t, e) acc -> SLet (t, e, acc))
    (lo_tmp @ [(hi_tmp, hi)])
    (SFor (v, lo_expr, EVar hi_tmp, dir, body))

let rec walk counter (s : stmt) : stmt =
  let go = walk counter in
  match s with
  | SFor (v, lo, hi, dir, body) ->
      let assigned = assigned_names body in
      let body = go body in
      if stable ~assigned hi then SFor (v, lo, hi, dir, body)
      else hoist_bound counter v lo hi dir body
  | SPragma (hints, (SFor _ as loop)) ->
      (* Keep the pragma attached to the loop it governs. *)
      under_temps (fun loop -> SPragma (hints, loop)) (go loop)
  | SPragma (hints, body) -> SPragma (hints, go body)
  | SSeq stmts -> SSeq (List.map go stmts)
  | SIf (c, t, e) -> SIf (c, go t, Option.map go e)
  | SWhile (c, body) -> SWhile (c, go body)
  | SMatch (scrut, cases) ->
      SMatch (scrut, List.map (fun (p, b) -> (p, go b)) cases)
  | SLet (v, e, body) -> SLet (v, e, go body)
  | SLetMut (v, e, body) -> SLetMut (v, e, go body)
  | SBlock body -> SBlock (go body)
  | SAssign _ | SReturn _ | SBarrier | SWarpBarrier | SExpr _ | SEmpty
  | SMemFence | SNative _ | SCoopmat _ ->
      s

let hoist_stmt s = walk (ref 0) s

let hoist (k : kernel) : kernel =
  let counter = ref 0 in
  {
    k with
    kern_body = walk counter k.kern_body;
    kern_funcs =
      List.map
        (fun hf -> {hf with hf_body = walk counter hf.hf_body})
        k.kern_funcs;
  }
