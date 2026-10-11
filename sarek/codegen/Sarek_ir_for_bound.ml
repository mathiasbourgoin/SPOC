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

let fresh_bound_var counter =
  incr counter ;
  {
    var_name = Printf.sprintf "sarek_for_hi_%d" !counter;
    var_id = 0;
    var_type = TInt32;
    var_mutable = false;
  }

let rec walk counter (s : stmt) : stmt =
  let go = walk counter in
  match s with
  | SFor (v, lo, hi, dir, body) ->
      let assigned = assigned_names body in
      let body = go body in
      if stable ~assigned hi then SFor (v, lo, hi, dir, body)
      else
        let tmp = fresh_bound_var counter in
        SLet (tmp, hi, SFor (v, lo, EVar tmp, dir, body))
  | SPragma (hints, (SFor _ as loop)) -> (
      (* Keep the pragma attached to the loop it governs. *)
      match go loop with
      | SLet (tmp, hi, loop) -> SLet (tmp, hi, SPragma (hints, loop))
      | loop -> SPragma (hints, loop))
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
