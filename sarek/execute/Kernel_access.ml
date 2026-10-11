(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** How a kernel uses its vector parameters, and whether it can be fused
    horizontally (specs/launch-window.md).

    Both answers are read from the IR only and are conservative: a use this
    module cannot classify counts as a read AND a write, and a construct it does
    not understand makes the kernel non-fusable. Wrong in the other direction
    would let the launch window reorder or merge launches that conflict. *)

open Sarek_ir_types

(** {1 Read/write sets} *)

type access = {read : bool; write : bool}

let no_access = {read = false; write = false}

let both = {read = true; write = true}

let join a b = {read = a.read || b.read; write = a.write || b.write}

(** Vector parameter names of [k], in declaration order. *)
let vector_params (k : kernel) : string list =
  List.filter_map
    (function DParam (v, Some _) -> Some v.var_name | _ -> None)
    k.kern_params

(* The accumulator maps a parameter name to its access so far. Only names in
   the initial table are recorded: shared arrays and locals are not launch
   arguments. *)
type acc = (string, access) Hashtbl.t

let note (tbl : acc) name a =
  match Hashtbl.find_opt tbl name with
  | Some prev -> Hashtbl.replace tbl name (join prev a)
  | None -> ()

let read_ = {read = true; write = false}

let write_ = {read = false; write = true}

(* An array base given as an expression: an [EVar] naming a parameter is an
   access of that kind; anything else is visited as an ordinary expression,
   where a bare parameter counts as [both]. *)
let rec base_access tbl a = function
  | EVar v -> note tbl v.var_name a
  | e -> expr tbl e

and expr tbl e =
  match e with
  | EConst _ | EArrayLen _ -> ()
  | EVar v ->
      (* A vector passed whole (to an intrinsic, a helper, a record...) may be
         read or written by the callee. *)
      note tbl v.var_name both
  | EArrayRead (name, idx) ->
      note tbl name read_ ;
      expr tbl idx
  | EArrayReadExpr (base, idx) ->
      base_access tbl read_ base ;
      expr tbl idx
  | EBinop (_, a, b) ->
      expr tbl a ;
      expr tbl b
  | EUnop (_, a) | ECast (_, a) | ERecordField (a, _) -> expr tbl a
  | EIntrinsic (_, _, args) | ETuple args | EVariant (_, _, args) ->
      List.iter (expr tbl) args
  | EApp (f, args) -> List.iter (expr tbl) (f :: args)
  | ERecord (_, fields) -> List.iter (fun (_, e) -> expr tbl e) fields
  | EArrayCreate (_, size, _) -> expr tbl size
  | EIf (c, t, f) -> List.iter (expr tbl) [c; t; f]
  | EMatch (s, cases) ->
      expr tbl s ;
      List.iter (fun (_, e) -> expr tbl e) cases

let rec lvalue tbl = function
  | LVar _ -> ()
  | LArrayElem (name, idx) ->
      note tbl name write_ ;
      expr tbl idx
  | LArrayElemExpr (base, idx) ->
      base_access tbl write_ base ;
      expr tbl idx
  | LRecordField (lv, _) -> lvalue_inner tbl lv

(* [a.(i).f <- v] writes element [i] of [a]. *)
and lvalue_inner tbl lv = lvalue tbl lv

let coopmat tbl = function
  | CM_decl _ | CM_muladd _ -> ()
  | CM_load {src; index; stride; _} ->
      note tbl src read_ ;
      expr tbl index ;
      expr tbl stride
  | CM_store {dst; index; stride; _} ->
      note tbl dst write_ ;
      expr tbl index ;
      expr tbl stride

let rec stmt tbl s =
  match s with
  | SAssign (lv, e) ->
      lvalue tbl lv ;
      expr tbl e
  | SSeq ss -> List.iter (stmt tbl) ss
  | SIf (c, t, f) ->
      expr tbl c ;
      stmt tbl t ;
      Option.iter (stmt tbl) f
  | SWhile (c, b) ->
      expr tbl c ;
      stmt tbl b
  | SFor (_, lo, hi, _, b) ->
      expr tbl lo ;
      expr tbl hi ;
      stmt tbl b
  | SMatch (e, cases) ->
      expr tbl e ;
      List.iter (fun (_, b) -> stmt tbl b) cases
  | SReturn e | SExpr e -> expr tbl e
  | SLet (_, e, b) | SLetMut (_, e, b) ->
      expr tbl e ;
      stmt tbl b
  | SPragma (_, b) | SBlock b -> stmt tbl b
  | SCoopmat op -> coopmat tbl op
  | SNative _ ->
      (* Inline backend code may touch any parameter. *)
      Hashtbl.filter_map_inplace (fun _ _ -> Some both) tbl
  | SBarrier | SWarpBarrier | SMemFence | SEmpty -> ()

(** Access of each vector parameter of [k], in declaration order. Helper
    functions receive vectors only as arguments, which {!expr} already counts as
    [both]. *)
let param_accesses (k : kernel) : (string * access) list =
  let names = vector_params k in
  let tbl : acc = Hashtbl.create 8 in
  List.iter (fun n -> Hashtbl.replace tbl n no_access) names ;
  stmt tbl k.kern_body ;
  List.iter
    (function
      | DLocal (_, Some e) -> expr tbl e | DLocal _ | DParam _ | DShared _ -> ())
    k.kern_locals ;
  List.map (fun n -> (n, Hashtbl.find tbl n)) names

(** {1 Horizontal fusability} *)

(** Intrinsics whose value depends on the work-group's position in the grid.
    Horizontal fusion rewrites exactly these. *)
let position_intrinsics =
  [
    "block_idx_x";
    "grid_dim_x";
    "global_thread_id";
    "global_idx";
    "global_idx_x";
    "global_size";
    "global_size_x";
  ]

let is_stdlib_path = function "Sarek_stdlib" :: _ -> true | _ -> false

(* The leading [let a = create_array ... Shared in] bindings of a body, and the
   rest. Shared arrays must be declared at kernel scope on OpenCL and Metal, so
   only this prefix can be hoisted above the dispatch branch. *)
let rec shared_prefix = function
  | SLet (v, (EArrayCreate (_, _, Shared) as e), rest) ->
      let binds, body = shared_prefix rest in
      ((v, e) :: binds, body)
  | s -> ([], s)

exception Refuse of string

let refuse fmt = Printf.ksprintf (fun s -> raise (Refuse s)) fmt

let rec check_expr ~in_helper e =
  let go = check_expr ~in_helper in
  match e with
  | EIntrinsic (path, name, args) ->
      if not (is_stdlib_path path) then
        refuse
          "intrinsic %s.%s is not from Sarek_stdlib"
          (String.concat "." path)
          name ;
      if in_helper && List.mem name position_intrinsics then
        refuse "helper function reads %s" name ;
      List.iter go args
  | EArrayCreate (_, _, Shared) ->
      refuse "shared array not in the body's leading let prefix"
  | EArrayCreate (_, size, _) -> go size
  | EConst _ | EVar _ | EArrayLen _ -> ()
  | EArrayRead (_, i) -> go i
  | EArrayReadExpr (a, b) | EBinop (_, a, b) ->
      go a ;
      go b
  | EUnop (_, a) | ECast (_, a) | ERecordField (a, _) -> go a
  | ETuple es | EVariant (_, _, es) -> List.iter go es
  | EApp (f, es) -> List.iter go (f :: es)
  | ERecord (_, fs) -> List.iter (fun (_, e) -> go e) fs
  | EIf (a, b, c) -> List.iter go [a; b; c]
  | EMatch (s, cs) ->
      go s ;
      List.iter (fun (_, e) -> go e) cs

let rec check_lvalue ~in_helper = function
  | LVar _ -> ()
  | LArrayElem (_, i) -> check_expr ~in_helper i
  | LArrayElemExpr (a, i) ->
      check_expr ~in_helper a ;
      check_expr ~in_helper i
  | LRecordField (lv, _) -> check_lvalue ~in_helper lv

let rec check_stmt ~in_helper s =
  let e = check_expr ~in_helper and st = check_stmt ~in_helper in
  match s with
  | SAssign (lv, x) ->
      check_lvalue ~in_helper lv ;
      e x
  | SSeq ss -> List.iter st ss
  | SIf (c, t, f) ->
      e c ;
      st t ;
      Option.iter st f
  | SWhile (c, b) ->
      e c ;
      st b
  | SFor (_, lo, hi, _, b) ->
      e lo ;
      e hi ;
      st b
  | SMatch (x, cs) ->
      e x ;
      List.iter (fun (_, b) -> st b) cs
  | SReturn x | SExpr x -> e x
  | SLet (_, x, b) | SLetMut (_, x, b) ->
      e x ;
      st b
  | SPragma (_, b) | SBlock b -> st b
  | SNative _ -> refuse "inline native code"
  | SCoopmat _ -> refuse "cooperative-matrix statement"
  | SBarrier | SWarpBarrier | SMemFence | SEmpty -> ()

(** [Ok ()] when [k] can be a member of a horizontally fused kernel, otherwise
    the reason it cannot. *)
let horizontal_verdict (k : kernel) : (unit, string) result =
  match
    let binds, body = shared_prefix k.kern_body in
    List.iter
      (fun (_, x) ->
        match x with
        | EArrayCreate (_, size, _) -> check_expr ~in_helper:false size
        | _ -> ())
      binds ;
    check_stmt ~in_helper:false body ;
    List.iter
      (function
        | DLocal (_, Some x) -> check_expr ~in_helper:false x
        | DShared (_, _, Some x) -> check_expr ~in_helper:false x
        | DLocal _ | DShared _ | DParam _ -> ())
      k.kern_locals ;
    List.iter (fun hf -> check_stmt ~in_helper:true hf.hf_body) k.kern_funcs
  with
  | () -> Ok ()
  | exception Refuse why -> Error why
