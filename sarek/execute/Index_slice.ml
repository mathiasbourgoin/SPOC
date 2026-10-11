(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Host evaluation of a kernel's store indices for one launch
    (specs/epilogue-fusion.md).

    [stores] answers "which elements of array [a] does this launch write, and
    how often", by running, for every thread of the launch, only the integer
    computations that the stores to [a] depend on: the index expressions, the
    conditions and loop bounds around the stores, and, transitively, the
    variables those read. These may use thread and work-group indices, launch
    dimensions, int32 scalar parameters and int32 constants. Anything else (an
    element read, a float, a [match], an int64) makes the evaluation fail, which
    callers must treat as "unknown": never as "no store".

    {b The slice.} A variable is relevant when a store index, a condition or
    loop bound enclosing a store or a [return], or an assignment to a relevant
    variable reads it (fixpoint). Statements that neither contain a store to
    [a], a [return], nor assign a relevant variable are skipped: they cannot
    change which stores execute. A [return] stops the thread, so the conditions
    around it are relevant too.

    Values follow int32 two's-complement arithmetic, as the device does.
    Division or remainder by zero fails the evaluation. *)

open Sarek_ir_types

exception Unknown of string

let unknown fmt = Printf.ksprintf (fun s -> raise (Unknown s)) fmt

(** {1 Relevance} *)

module Ids = Set.Make (Int)

let rec expr_vars acc = function
  | EVar v -> Ids.add v.var_id acc
  | EConst _ | EArrayLen _ -> acc
  | EBinop (_, a, b) | EArrayReadExpr (a, b) -> expr_vars (expr_vars acc a) b
  | EUnop (_, a) | ECast (_, a) | ERecordField (a, _) -> expr_vars acc a
  | EArrayRead (_, i) -> expr_vars acc i
  | EIntrinsic (_, _, es) | ETuple es | EVariant (_, _, es) ->
      List.fold_left expr_vars acc es
  | EApp (f, es) -> List.fold_left expr_vars acc (f :: es)
  | ERecord (_, fs) -> List.fold_left (fun a (_, e) -> expr_vars a e) acc fs
  | EArrayCreate (_, s, _) -> expr_vars acc s
  | EIf (a, b, c) -> List.fold_left expr_vars acc [a; b; c]
  | EMatch (s, cs) ->
      List.fold_left (fun a (_, e) -> expr_vars a e) (expr_vars acc s) cs

(* Does [s] contain a store to [arr] (or, with [arr = None], to any array
   parameter in [params]) or a [return]? *)
let rec has_target ~is_target s =
  let go = has_target ~is_target in
  match s with
  | SAssign (LArrayElem (a, _), _) -> is_target a
  | SAssign (LArrayElemExpr (EVar v, _), _) -> is_target v.var_name
  | SAssign (LRecordField (lv, _), x) -> go (SAssign (lv, x))
  | SCoopmat (CM_store {dst; _}) -> is_target dst
  | SReturn _ -> true
  | SSeq ss -> List.exists go ss
  | SIf (_, t, f) -> go t || Option.fold ~none:false ~some:go f
  | SWhile (_, b) | SFor (_, _, _, _, b) | SPragma (_, b) | SBlock b -> go b
  | SLet (_, _, b) | SLetMut (_, _, b) -> go b
  | SMatch (_, cs) -> List.exists (fun (_, b) -> go b) cs
  | SAssign _ | SExpr _ | SBarrier | SWarpBarrier | SMemFence | SEmpty
  | SNative _ | SCoopmat _ ->
      false

(* Variables a statement assigns (let, let mut, loop variable, :=). *)
let rec assigned acc = function
  | SAssign (LVar v, _) -> Ids.add v.var_id acc
  | SLet (v, _, b) | SLetMut (v, _, b) -> assigned (Ids.add v.var_id acc) b
  | SFor (v, _, _, _, b) -> assigned (Ids.add v.var_id acc) b
  | SSeq ss -> List.fold_left assigned acc ss
  | SIf (_, t, f) ->
      Option.fold ~none:(assigned acc t) ~some:(assigned (assigned acc t)) f
  | SWhile (_, b) | SPragma (_, b) | SBlock b -> assigned acc b
  | SMatch (_, cs) -> List.fold_left (fun a (_, b) -> assigned a b) acc cs
  | SAssign _ | SExpr _ | SReturn _ | SBarrier | SWarpBarrier | SMemFence
  | SEmpty | SNative _ | SCoopmat _ ->
      acc

(* A statement matters when it contains a target store/return or assigns a
   relevant variable. *)
let matters ~is_target rel s =
  has_target ~is_target s
  || not (Ids.is_empty (Ids.inter rel (assigned Ids.empty s)))

(* One pass adding to [rel] the variables read by everything that matters. *)
let rec grow ~is_target rel s =
  let m = matters ~is_target rel in
  let g = grow ~is_target in
  match s with
  | SAssign (LArrayElem (a, i), _) when is_target a -> expr_vars rel i
  | SAssign (LVar v, e) when Ids.mem v.var_id rel -> expr_vars rel e
  | SLet (v, e, b) | SLetMut (v, e, b) ->
      let rel = if Ids.mem v.var_id rel then expr_vars rel e else rel in
      g rel b
  | SSeq ss -> List.fold_left g rel ss
  | SIf (c, t, f) ->
      let rel = if m s then expr_vars rel c else rel in
      let rel = g rel t in
      Option.fold ~none:rel ~some:(g rel) f
  | SWhile (c, b) -> g (if m s then expr_vars rel c else rel) b
  | SFor (v, lo, hi, _, b) ->
      let rel =
        if m s || Ids.mem v.var_id rel then expr_vars (expr_vars rel lo) hi
        else rel
      in
      g rel b
  | SPragma (_, b) | SBlock b -> g rel b
  | SMatch (e, cs) ->
      let rel = if m s then expr_vars rel e else rel in
      List.fold_left (fun r (_, b) -> g r b) rel cs
  | SReturn _ | SAssign _ | SExpr _ | SBarrier | SWarpBarrier | SMemFence
  | SEmpty | SNative _ | SCoopmat _ ->
      rel

let relevant ~is_target body =
  let rec fix rel =
    let rel' = grow ~is_target rel body in
    if Ids.equal rel rel' then rel else fix rel'
  in
  fix Ids.empty

(** {1 Compilation to closures} *)

(* Thread state: position, and one int32 slot per relevant variable. *)
type state = {
  slots : int32 array;
  mutable tx : int;
  mutable ty : int;
  mutable tz : int;
  mutable bx : int;
  mutable by : int;
  mutable bz : int;
  dims : int array;  (** block x y z, grid x y z *)
  mutable stop : bool;  (** the thread executed [return] *)
  mutable steps : int;  (** loop iterations and threads so far, this run *)
}

type cenv = {
  slot_of : (int, int) Hashtbl.t;  (** var_id -> slot *)
  scalars : (string, int32) Hashtbl.t;  (** int32 scalar parameters *)
  mutable used : string list;  (** scalar parameters the slice read *)
}

let b2i b = if b then 1l else 0l

let i32 n = Int32.of_int n

let position name : state -> int32 =
  match name with
  | "thread_idx_x" | "thread_id_x" -> fun s -> i32 s.tx
  | "thread_idx_y" | "thread_id_y" -> fun s -> i32 s.ty
  | "thread_idx_z" | "thread_id_z" -> fun s -> i32 s.tz
  | "block_idx_x" | "block_id_x" -> fun s -> i32 s.bx
  | "block_idx_y" | "block_id_y" -> fun s -> i32 s.by
  | "block_idx_z" | "block_id_z" -> fun s -> i32 s.bz
  | "block_dim_x" -> fun s -> i32 s.dims.(0)
  | "block_dim_y" -> fun s -> i32 s.dims.(1)
  | "block_dim_z" -> fun s -> i32 s.dims.(2)
  | "grid_dim_x" -> fun s -> i32 s.dims.(3)
  | "grid_dim_y" -> fun s -> i32 s.dims.(4)
  | "grid_dim_z" -> fun s -> i32 s.dims.(5)
  | "global_thread_id" | "global_idx" | "global_idx_x" ->
      fun s -> i32 ((s.bx * s.dims.(0)) + s.tx)
  | "global_idx_y" -> fun s -> i32 ((s.by * s.dims.(1)) + s.ty)
  | "global_idx_z" -> fun s -> i32 ((s.bz * s.dims.(2)) + s.tz)
  | "global_size" | "global_size_x" -> fun s -> i32 (s.dims.(0) * s.dims.(3))
  | n -> unknown "intrinsic %s" n

let check_div d = if d = 0l then raise (Unknown "division by zero") else d

let binop op (a : state -> int32) (b : state -> int32) : state -> int32 =
  match op with
  | Add -> fun s -> Int32.add (a s) (b s)
  | Sub -> fun s -> Int32.sub (a s) (b s)
  | Mul -> fun s -> Int32.mul (a s) (b s)
  | Div ->
      fun s ->
        let x = a s in
        Int32.div x (check_div (b s))
  | Mod ->
      fun s ->
        let x = a s in
        Int32.rem x (check_div (b s))
  | Eq -> fun s -> b2i (a s = b s)
  | Ne -> fun s -> b2i (a s <> b s)
  | Lt -> fun s -> b2i (a s < b s)
  | Le -> fun s -> b2i (a s <= b s)
  | Gt -> fun s -> b2i (a s > b s)
  | Ge -> fun s -> b2i (a s >= b s)
  | And -> fun s -> if a s <> 0l then b2i (b s <> 0l) else 0l
  | Or -> fun s -> if a s <> 0l then 1l else b2i (b s <> 0l)
  | Shl ->
      fun s ->
        let x = a s in
        Int32.shift_left x (Int32.to_int (b s) land 31)
  | Shr ->
      fun s ->
        let x = a s in
        Int32.shift_right x (Int32.to_int (b s) land 31)
  | Lshr32 ->
      fun s ->
        let x = a s in
        Int32.shift_right_logical x (Int32.to_int (b s) land 31)
  | BitAnd -> fun s -> Int32.logand (a s) (b s)
  | BitOr -> fun s -> Int32.logor (a s) (b s)
  | BitXor -> fun s -> Int32.logxor (a s) (b s)
  | Lshr64 -> unknown "int64 shift"

let rec cexpr env e : state -> int32 =
  match e with
  | EConst (CInt32 n) -> fun _ -> n
  | EConst (CBool b) ->
      let v = b2i b in
      fun _ -> v
  | EVar v -> (
      match Hashtbl.find_opt env.slot_of v.var_id with
      | Some k -> fun s -> s.slots.(k)
      | None -> (
          match Hashtbl.find_opt env.scalars v.var_name with
          | Some n ->
              if not (List.mem v.var_name env.used) then
                env.used <- v.var_name :: env.used ;
              fun _ -> n
          | None -> unknown "variable %s" v.var_name))
  | EBinop (op, a, b) -> binop op (cexpr env a) (cexpr env b)
  | EUnop (Neg, a) ->
      let a = cexpr env a in
      fun s -> Int32.neg (a s)
  | EUnop (Not, a) ->
      let a = cexpr env a in
      fun s -> b2i (a s = 0l)
  | EUnop (BitNot, a) ->
      let a = cexpr env a in
      fun s -> Int32.lognot (a s)
  | EIntrinsic (_, name, []) -> position name
  | ECast ((TInt32 | TBool), a) -> cexpr env a
  | EIf (c, a, b) ->
      let c = cexpr env c and a = cexpr env a and b = cexpr env b in
      fun s -> if c s <> 0l then a s else b s
  | _ -> unknown "non-integer or memory-dependent expression"

type cstmt = state -> unit

let seq (fs : cstmt list) : cstmt =
 fun s -> List.iter (fun f -> if not s.stop then f s) fs

(* Work bound for one [run] (loop iterations plus threads): the evaluation
   happens while the launch window is locked. Beyond it, the answer is
   unknown, so nothing is merged. *)
let max_steps = 1 lsl 26

let tick s =
  s.steps <- s.steps + 1 ;
  if s.steps > max_steps then unknown "evaluation budget exceeded"

let rec cstmt ~is_target ~record env rel st : cstmt =
  let c = cstmt ~is_target ~record env rel in
  let slot (v : var) =
    match Hashtbl.find_opt env.slot_of v.var_id with
    | Some k -> k
    | None ->
        let k = Hashtbl.length env.slot_of in
        Hashtbl.replace env.slot_of v.var_id k ;
        k
  in
  let skip : cstmt = fun _ -> () in
  if not (matters ~is_target rel st) then skip
  else
    match st with
    | SAssign (LArrayElem (a, i), _) when is_target a ->
        let i = cexpr env i in
        fun s -> record (i s)
    | SAssign (LArrayElemExpr (EVar v, _), _) when is_target v.var_name ->
        unknown "store through an array expression"
    | SAssign (LRecordField _, _) | SCoopmat (CM_store _) ->
        (* Only plain element stores to the target are counted; any other
           store form to it makes the evaluation unknown. *)
        unknown "store to a field or a cooperative-matrix tile"
    | SAssign (LVar v, e) ->
        let k = slot v and e = cexpr env e in
        fun s -> s.slots.(k) <- e s
    | SLet (v, e, b) | SLetMut (v, e, b) ->
        if Ids.mem v.var_id rel then begin
          let k = slot v in
          let e = cexpr env e in
          let b = c b in
          fun s ->
            s.slots.(k) <- e s ;
            b s
        end
        else c b
    | SSeq ss -> seq (List.map c ss)
    | SIf (cond, t, f) ->
        let cond = cexpr env cond and t = c t in
        let f = Option.fold ~none:skip ~some:c f in
        fun s -> if cond s <> 0l then t s else f s
    | SWhile (cond, b) ->
        let cond = cexpr env cond and b = c b in
        fun s ->
          while (not s.stop) && cond s <> 0l do
            b s ;
            tick s
          done
    | SFor (v, lo, hi, dir, b) ->
        let k = slot v
        and lo = cexpr env lo
        and hi = cexpr env hi
        and b = c b in
        fun s ->
          let lo = Int32.to_int (lo s) and hi = Int32.to_int (hi s) in
          let step = match dir with Upto -> 1 | Downto -> -1 in
          let i = ref lo in
          while
            (not s.stop)
            && match dir with Upto -> !i <= hi | Downto -> !i >= hi
          do
            s.slots.(k) <- Int32.of_int !i ;
            b s ;
            tick s ;
            i := !i + step
          done
    | SPragma (_, b) | SBlock b -> c b
    | SReturn _ -> fun s -> s.stop <- true
    | SMatch _ -> unknown "match around a store"
    | SNative _ | SCoopmat _ -> unknown "native or cooperative-matrix code"
    | SAssign _ | SExpr _ | SBarrier | SWarpBarrier | SMemFence | SEmpty -> skip

(** {1 Running a launch} *)

type dims = {block : int * int * int; grid : int * int * int}

(** Run [body] for every thread of a launch of [dims]. [is_target] selects the
    arrays whose stores are recorded; [record] receives each stored index (and,
    for stores in a thread, in program order). [scalars] gives the int32 scalar
    parameters by name; [bind] pre-binds variables (var, value) before each
    thread. Returns the scalar parameters the evaluation read. Raises [Unknown]
    when a relevant computation is not an integer function of the launch. *)
let run ?(bind = []) ~is_target ~record ~scalars ~dims body : string list =
  let rel = relevant ~is_target body in
  let rel =
    List.fold_left (fun r ((v : var), _) -> Ids.add v.var_id r) rel bind
  in
  let env =
    {slot_of = Hashtbl.create 16; scalars = Hashtbl.create 8; used = []}
  in
  List.iter (fun (n, v) -> Hashtbl.replace env.scalars n v) scalars ;
  let bound =
    List.map
      (fun ((v : var), value) ->
        let k = Hashtbl.length env.slot_of in
        Hashtbl.replace env.slot_of v.var_id k ;
        (k, value))
      bind
  in
  let code = cstmt ~is_target ~record env rel body in
  let bx, by, bz = dims.block and gx, gy, gz = dims.grid in
  let s =
    {
      slots = Array.make (max 1 (Hashtbl.length env.slot_of)) 0l;
      tx = 0;
      ty = 0;
      tz = 0;
      bx = 0;
      by = 0;
      bz = 0;
      dims = [|bx; by; bz; gx; gy; gz|];
      stop = false;
      steps = 0;
    }
  in
  for gk = 0 to gz - 1 do
    for gj = 0 to gy - 1 do
      for gi = 0 to gx - 1 do
        for tk = 0 to bz - 1 do
          for tj = 0 to by - 1 do
            for ti = 0 to bx - 1 do
              s.bx <- gi ;
              s.by <- gj ;
              s.bz <- gk ;
              s.tx <- ti ;
              s.ty <- tj ;
              s.tz <- tk ;
              s.stop <- false ;
              tick s ;
              List.iter (fun (k, v) -> s.slots.(k) <- v) bound ;
              code s
            done
          done
        done
      done
    done
  done ;
  env.used
