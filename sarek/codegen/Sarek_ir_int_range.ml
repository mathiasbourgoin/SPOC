(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Interval arithmetic on int32 values, for Sarek_ir_f16_narrowing: a range
   always contains every value the expression can take, wrapping included. *)

open Sarek_ir_types

let int32_min = -0x8000_0000

let int32_max = 0x7fff_ffff

type range = {lo : int; hi : int}

let full = {lo = int32_min; hi = int32_max}

let point n = {lo = n; hi = n}

let join_range a b = {lo = min a.lo b.lo; hi = max a.hi b.hi}

(* A range outside int32 means the operation may wrap: any value. *)
let fit r = if r.lo >= int32_min && r.hi <= int32_max then r else full

let nonneg r = r.lo >= 0

(* In float: a product of two int32 bounds can exceed OCaml's 63-bit int. *)
let mul_range a b =
  let corners =
    [(a.lo, b.lo); (a.lo, b.hi); (a.hi, b.lo); (a.hi, b.hi)]
    |> List.map (fun (x, y) -> float_of_int x *. float_of_int y)
  in
  let lo = List.fold_left min infinity corners
  and hi = List.fold_left max neg_infinity corners in
  if lo >= float_of_int int32_min && hi <= float_of_int int32_max then
    {lo = int_of_float lo; hi = int_of_float hi}
  else full

let div_range a b =
  if b.lo = b.hi && b.lo > 0 then {lo = a.lo / b.lo; hi = a.hi / b.lo} else full

let mod_range a b =
  if b.lo <> b.hi || b.lo = 0 then full
  else
    let m = abs b.lo - 1 in
    if nonneg a then {lo = 0; hi = min a.hi m}
    else {lo = -m; hi = (if a.hi <= 0 then 0 else m)}

let and_range a b =
  match (nonneg a, nonneg b) with
  | true, true -> {lo = 0; hi = min a.hi b.hi}
  | true, false -> {lo = 0; hi = a.hi}
  | false, true -> {lo = 0; hi = b.hi}
  | false, false -> full

(* The smallest 2^k - 1 >= n. *)
let rec all_ones n = if n <= 0 then 0 else (all_ones (n lsr 1) lsl 1) lor 1

let or_range a b =
  if nonneg a && nonneg b then {lo = 0; hi = all_ones (max a.hi b.hi)} else full

let shift_ok r = r.lo >= 0 && r.hi <= 31

let asr_range a b =
  if not (shift_ok b) then full
  else
    let corners =
      [a.lo asr b.lo; a.lo asr b.hi; a.hi asr b.lo; a.hi asr b.hi]
    in
    {
      lo = List.fold_left min max_int corners;
      hi = List.fold_left max min_int corners;
    }

let lsr_range a b =
  if not (shift_ok b) then full
  else if nonneg a then {lo = a.lo lsr b.hi; hi = a.hi lsr b.lo}
  else if b.lo >= 1 then {lo = 0; hi = (1 lsl (32 - b.lo)) - 1}
  else full

let shl_range a b =
  if b.lo = b.hi && shift_ok b then mul_range a (point (1 lsl b.lo)) else full

let binop_range op a b =
  match op with
  | Add -> fit {lo = a.lo + b.lo; hi = a.hi + b.hi}
  | Sub -> fit {lo = a.lo - b.hi; hi = a.hi - b.lo}
  | Mul -> mul_range a b
  | Div -> div_range a b
  | Mod -> mod_range a b
  | BitAnd -> and_range a b
  | BitOr | BitXor -> or_range a b
  | Shl -> shl_range a b
  | Shr -> asr_range a b
  | Lshr32 -> lsr_range a b
  | Eq | Ne | Lt | Le | Gt | Ge | And | Or -> {lo = 0; hi = 1}
  | Lshr64 -> full

let unop_range op a =
  match op with
  | Neg -> fit {lo = -a.hi; hi = -a.lo}
  | BitNot -> {lo = -a.hi - 1; hi = -a.lo - 1}
  | Not -> {lo = 0; hi = 1}
