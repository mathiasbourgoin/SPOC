(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Automatic unroll hints for small constant-trip [for] loops. See the .mli
   for the rule and specs/codegen-lsr-unroll.md for why it is a default. *)

open Sarek_ir_types

let max_trip = 16

let max_expansion = 64

let hint = ["unroll"]

let enabled () =
  (not (Sarek_policy.restricted Sarek_policy.Auto_unroll))
  &&
  match Sys.getenv_opt "SAREK_AUTO_UNROLL" with
  | Some "0" -> false
  | _ -> true

(* Integer value of a constant bound: literals and [+ - *] of literals. Any
   value outside the int32 range (where a device's int32 arithmetic would
   wrap) gives up, so a wrapped bound is never mis-counted. *)
let rec const_int (e : expr) : int64 option =
  let in_range n =
    if
      Int64.compare n (Int64.of_int32 Int32.min_int) >= 0
      && Int64.compare n (Int64.of_int32 Int32.max_int) <= 0
    then Some n
    else None
  in
  let both a b f =
    match (const_int a, const_int b) with
    | Some x, Some y -> in_range (f x y)
    | _ -> None
  in
  match e with
  | EConst (CInt32 n) -> Some (Int64.of_int32 n)
  | EConst (CInt64 n) -> in_range n
  | ECast ((TInt32 | TInt64), e) -> const_int e
  | EBinop (Add, a, b) -> both a b Int64.add
  | EBinop (Sub, a, b) -> both a b Int64.sub
  | EBinop (Mul, a, b) -> both a b Int64.mul
  | _ -> None

let trip_count start stop dir =
  match (const_int start, const_int stop) with
  | Some lo, Some hi ->
      let n =
        match dir with Upto -> Int64.sub hi lo | Downto -> Int64.sub lo hi
      in
      (* Bounds are inclusive. A negative difference means zero iterations. *)
      if Int64.compare n 0L < 0 then Some 0
      else if Int64.compare n (Int64.of_int max_trip) >= 0 then None
      else Some (Int64.to_int n + 1)
  | _ -> None

(* Sarek's own directives ([sarek.inline N], also written as two strings
   ["sarek.inline"; "N"]) are not loop pragmas. *)
let is_sarek_directive = function
  | h :: _ -> String.length h >= 6 && String.sub h 0 6 = "sarek."
  | [] -> false

(* [walk ~under_pragma s] returns the annotated statement and the largest
   expansion (product of trip counts of hinted loops along one nesting path)
   inside it. [under_pragma] is true between an explicit [SPragma] and the
   first loop it governs, which is left as the author wrote it. *)
let rec walk ~under_pragma (s : stmt) : stmt * int =
  let go = walk ~under_pragma in
  match s with
  | SFor (v, start, stop, dir, body) ->
      let body, inner = walk ~under_pragma:false body in
      let loop = SFor (v, start, stop, dir, body) in
      if under_pragma then
        (* The author's pragma governs this loop. Count it as fully unrolled
           when its trip count is known, so an automatic hint around it still
           respects max_expansion. *)
        match trip_count start stop dir with
        | Some n when n >= 1 -> (loop, n * inner)
        | _ -> (loop, inner)
      else begin
        match trip_count start stop dir with
        | Some n when n >= 1 && n * inner <= max_expansion ->
            (SPragma (hint, loop), n * inner)
        | _ -> (loop, inner)
      end
  | SPragma (h, body) ->
      let explicit = not (is_sarek_directive h) in
      let body, e = walk ~under_pragma:(under_pragma || explicit) body in
      (SPragma (h, body), e)
  | SSeq stmts ->
      let pairs = List.map go stmts in
      ( SSeq (List.map fst pairs),
        List.fold_left (fun m (_, e) -> max m e) 1 pairs )
  | SIf (c, t, e) -> (
      let t, et = go t in
      match e with
      | None -> (SIf (c, t, None), et)
      | Some e ->
          let e, ee = go e in
          (SIf (c, t, Some e), max et ee))
  | SWhile (c, body) ->
      let body, e = walk ~under_pragma:false body in
      (SWhile (c, body), e)
  | SMatch (scrut, cases) ->
      let pairs = List.map (fun (p, b) -> (p, go b)) cases in
      ( SMatch (scrut, List.map (fun (p, (b, _)) -> (p, b)) pairs),
        List.fold_left (fun m (_, (_, e)) -> max m e) 1 pairs )
  | SLet (v, e, body) ->
      let body, x = go body in
      (SLet (v, e, body), x)
  | SLetMut (v, e, body) ->
      let body, x = go body in
      (SLetMut (v, e, body), x)
  | SBlock body ->
      let body, x = go body in
      (SBlock body, x)
  | SAssign _ | SReturn _ | SBarrier | SWarpBarrier | SExpr _ | SEmpty
  | SMemFence | SNative _ | SCoopmat _ ->
      (s, 1)

let annotate_stmt s = if enabled () then fst (walk ~under_pragma:false s) else s

let annotate (k : kernel) : kernel =
  if not (enabled ()) then k
  else
    {
      k with
      kern_body = annotate_stmt k.kern_body;
      kern_funcs =
        List.map
          (fun hf -> {hf with hf_body = annotate_stmt hf.hf_body})
          k.kern_funcs;
    }
