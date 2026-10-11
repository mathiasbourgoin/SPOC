(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Host-side evaluation of a uniform int32 expression. See the interface. *)

open Sarek_ir_types

type launch = {
  grid : int * int * int;
  block : int * int * int;
  scalar : var -> int32 option;
}

let dim (x, y, z) = function 'x' -> x | 'y' -> y | _ -> z

let axis name = name.[String.length name - 1]

let intrinsic launch name =
  let prefix p =
    String.length name > String.length p
    && String.sub name 0 (String.length p) = p
  in
  if prefix "block_dim_" then Some (Int32.of_int (dim launch.block (axis name)))
  else if prefix "grid_dim_" then
    Some (Int32.of_int (dim launch.grid (axis name)))
  else None

let ( let* ) = Option.bind

let rec eval launch e =
  match e with
  | EConst (CInt32 n) -> Some n
  | EConst (CInt64 _) | ECast (TInt64, _) -> None
  | EConst (CBool b) -> Some (if b then 1l else 0l)
  | EVar v -> if v.var_type = TInt64 then None else launch.scalar v
  | ECast (TInt32, a) -> eval launch a
  | EIntrinsic (_, name, []) -> intrinsic launch name
  | EUnop (Neg, a) ->
      let* x = eval launch a in
      Some (Int32.neg x)
  | EBinop (op, a, b) ->
      let* x = eval launch a in
      let* y = eval launch b in
      binop op x y
  | _ -> None

and binop op x y =
  match op with
  | Add -> Some (Int32.add x y)
  | Sub -> Some (Int32.sub x y)
  | Mul -> Some (Int32.mul x y)
  | Div -> if y = 0l then None else Some (Int32.div x y)
  | Mod -> if y = 0l then None else Some (Int32.rem x y)
  | BitAnd -> Some (Int32.logand x y)
  | BitOr -> Some (Int32.logor x y)
  | BitXor -> Some (Int32.logxor x y)
  | Shl -> Some (Int32.shift_left x (Int32.to_int y land 31))
  | Shr -> Some (Int32.shift_right x (Int32.to_int y land 31))
  | Lshr32 -> Some (Int32.shift_right_logical x (Int32.to_int y land 31))
  | Lshr64 | Eq | Ne | Lt | Le | Gt | Ge | And | Or -> None

(* Intrinsics whose value depends on the launch geometry. [global_size] and
   [num_groups] are not Sarek stdlib intrinsics today; they are listed so that
   a library defining them cannot slip a geometry-dependent value past this
   check. *)
let launch_prefixes = ["block_dim_"; "grid_dim_"; "global_size"; "num_groups"]

let rec mentions_launch_dims = function
  | EIntrinsic (_, name, []) ->
      List.exists
        (fun q ->
          String.length name >= String.length q
          && String.sub name 0 (String.length q) = q)
        launch_prefixes
  | EBinop (_, a, b) -> mentions_launch_dims a || mentions_launch_dims b
  | EUnop (_, a) | ECast (_, a) | EArrayRead (_, a) -> mentions_launch_dims a
  | EIntrinsic (_, _, args) -> List.exists mentions_launch_dims args
  | EIf (c, a, b) ->
      mentions_launch_dims c || mentions_launch_dims a || mentions_launch_dims b
  | _ -> false
