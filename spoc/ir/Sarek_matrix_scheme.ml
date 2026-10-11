(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Building blocks of the rewritten GEMM kernel: tiling, IR helpers, operand
   addressing and the per-case element schemes. specs/matrix-pattern-rewrite.md. *)

open Sarek_ir_types
module Affine = Sarek_ir_affine
module P = Sarek_matrix_pattern
module C = Sarek_coopmat_types

type lowering =
  | Int32_byte_planes
  | F16_operands
  | Bf16_operands
  | Bf16x3_operands

let lowering_name = function
  | Int32_byte_planes -> "int32 byte planes (u8*u8+s32->s32, 10 products)"
  | F16_operands -> "f16 operands (f16*f16+f32->f32)"
  | Bf16_operands -> "bf16 operands (bf16*bf16+f32->f32)"
  | Bf16x3_operands -> "bf16x3 split operands (bf16*bf16+f32->f32, 6 products)"

(* ---- Tiling ------------------------------------------------------------ *)

(* A 64 x 64 block tile, 4 waves of 32 as 2 x 2, each wave 2 x 2 fragments. *)
let bm = 64

let bn = 64

(* The reduction step staged per barrier pair: the largest of 64, 32, 16 whose
   staging planes and accumulator tiles fit in 48 KiB of shared memory. *)
let elt_bytes = function TUint8 -> 1 | TFloat16 -> 2 | _ -> 4

let threads = 128

let extra_names =
  ["sarek_mm_me"; "sarek_mm_ne"; "sarek_mm_ze"; "sarek_mm_d"; "sarek_mm_t"]

(* ---- IR building blocks -------------------------------------------------- *)

let i32 n = EConst (CInt32 (Int32.of_int n))

let ( +: ) a b = EBinop (Add, a, b)

let ( -: ) a b = EBinop (Sub, a, b)

let ( *: ) a b = EBinop (Mul, a, b)

let ( /: ) a b = EBinop (Div, a, b)

let ( %: ) a b = EBinop (Mod, a, b)

let ( <: ) a b = EBinop (Lt, a, b)

let ( &&: ) a b = EBinop (And, a, b)

let gpu name = EIntrinsic (["Sarek_stdlib"; "Gpu"], name, [])

type fresh = {mutable next : int}

let new_var fresh name ty =
  fresh.next <- fresh.next + 1 ;
  {
    var_name = "sarek_mm_" ^ name;
    var_id = fresh.next;
    var_type = ty;
    var_mutable = false;
  }

let max_var_id (k : kernel) =
  Sarek_ir_analysis.kernel_fold
    {
      fe = (fun acc e -> match e with EVar v -> max acc v.var_id | _ -> acc);
      ft = (fun acc _ -> acc);
      fs =
        (fun acc s ->
          match s with
          | SLet (v, _, _) | SLetMut (v, _, _) | SFor (v, _, _, _, _) ->
              max acc v.var_id
          | _ -> acc);
      fnative = (fun acc -> acc);
      visit_lvalue = true;
    }
    (List.fold_left
       (fun acc -> function DParam (v, _) -> max acc v.var_id | _ -> acc)
       0
       k.kern_params)
    k

(* [let v = e in body v] *)
let let_ fresh name ty e body =
  let v = new_var fresh name ty in
  SLet (v, e, body (EVar v))

let for_ fresh name ~upto body =
  let v = {(new_var fresh name TInt32) with var_mutable = true} in
  SFor (v, i32 0, i32 upto, Upto, body (EVar v))

(* ---- Operand addressing --------------------------------------------------- *)

(* An operand's index as [offset0 + row * rs + col * cs + z * bs]: the
   constant part and the batch stride are split out of PR 1's offset. *)
type address = {offset0 : expr; rs : expr; cs : expr; bs : expr}

let address env (o : P.operand) batch =
  match Affine.of_expr env o.offset with
  | Error (Affine.Wide_index _) ->
      Error "an operand offset uses int64 arithmetic"
  | Error _ -> Error "an operand offset is not affine"
  | Ok f ->
      let bs =
        match batch with
        | Some b -> Affine.poly_to_expr (Affine.coefficient f b)
        | None -> i32 0
      in
      let rest = Affine.without f (Option.to_list batch) in
      if not (Affine.is_uniform rest) then
        Error "an operand offset varies with something other than the batch"
      else
        let addr =
          {
            offset0 = Affine.to_expr rest;
            rs = o.row_stride;
            cs = o.col_stride;
            bs;
          }
        in
        let uses_dims =
          List.exists
            Sarek_ir_eval_int.mentions_launch_dims
            [addr.offset0; addr.rs; addr.cs; addr.bs]
        in
        if uses_dims then Error "an operand address reads block_dim/grid_dim"
        else Ok addr

let index addr ~row ~col ~z =
  addr.offset0 +: (row *: addr.rs) +: (col *: addr.cs) +: (z *: addr.bs)

(* ---- Element schemes ---------------------------------------------------- *)

type scheme = {
  planes : int;
  shared_elt : elttype;
  operand : C.component_type;
  acc : C.component_type;
  acc_elt : elttype;
  products : (int * int * int) list;
      (** (plane of A, plane of B, accumulator) *)
  accumulators : int;
  stage : expr -> expr list;  (** an operand element -> its plane values *)
}

let int32_intrinsic name args =
  EIntrinsic (["Sarek_stdlib"; "Int32"], name, args)

(* Round a float32 to bfloat16, nearest even, kept as float32. *)
let bf16_round e =
  let bits = int32_intrinsic "bits_of_float" [e] in
  let lsb = EBinop (BitAnd, EBinop (Lshr32, bits, i32 16), i32 1) in
  let rounded = EBinop (Add, bits, EBinop (Add, i32 0x7fff, lsb)) in
  let high x = EBinop (BitAnd, x, EConst (CInt32 0xffff0000l)) in
  (* A NaN stays a (quiet) NaN, as in the HIP load helper: rounding its bits
     could carry into the exponent and give an infinity. *)
  let is_nan =
    EBinop (Gt, EBinop (BitAnd, bits, i32 0x7fffffff), i32 0x7f800000)
  in
  int32_intrinsic
    "float_of_bits"
    [EIf (is_nan, high (EBinop (BitOr, bits, i32 0x400000)), high rounded)]

let pairs ~planes ~max_order =
  List.concat_map
    (fun i ->
      List.filter_map
        (fun j -> if i + j <= max_order then Some (i, j, i + j) else None)
        (List.init planes Fun.id))
    (List.init planes Fun.id)

let fzero = EConst (CFloat32 0.0)

let scheme ~operand_elt = function
  | Int32_byte_planes ->
      {
        planes = 4;
        shared_elt = TUint8;
        operand = C.Uint8;
        acc = C.Sint32;
        acc_elt = TInt32;
        products = pairs ~planes:4 ~max_order:3;
        accumulators = 4;
        stage =
          (fun v ->
            List.init 4 (fun p ->
                ECast (TUint8, EBinop (Lshr32, v, i32 (8 * p)))));
      }
  | F16_operands ->
      {
        planes = 1;
        shared_elt = TFloat16;
        operand = C.Float16;
        acc = C.Float32;
        acc_elt = TFloat32;
        products = [(0, 0, 0)];
        accumulators = 1;
        stage =
          (fun v ->
            [(if operand_elt = TFloat16 then v else ECast (TFloat16, v))]);
      }
  | Bf16_operands ->
      {
        planes = 1;
        shared_elt = TFloat32;
        operand = C.BFloat16;
        acc = C.Float32;
        acc_elt = TFloat32;
        products = [(0, 0, 0)];
        accumulators = 1;
        stage = (fun v -> [v]);
      }
  | Bf16x3_operands ->
      let split v =
        let hi = bf16_round v in
        let r1 = EBinop (Sub, v, hi) in
        let mid = bf16_round r1 in
        [hi; r1; EBinop (Sub, r1, mid)]
      in
      {
        planes = 3;
        shared_elt = TFloat32;
        operand = C.BFloat16;
        acc = C.Float32;
        acc_elt = TFloat32;
        products =
          List.map (fun (i, j, _) -> (i, j, 0)) (pairs ~planes:3 ~max_order:2);
        accumulators = 1;
        stage = split;
      }

(* Shared rows are padded by 16 bytes so that the 16 lanes reading one
   fragment's rows (or columns) do not all hit one LDS bank. *)
let pad s = 16 / elt_bytes s.shared_elt

let choose_bk s =
  let acc_tiles = (if s.accumulators > 1 then 2 else 1) * bm * bn * 4 in
  let fits bk =
    (s.planes * ((bm * bk) + (bk * bn)) * elt_bytes s.shared_elt) + acc_tiles
    <= 48 * 1024
  in
  List.find fits [16]

let config s =
  {
    C.cfg_shape = {C.m = 16; n = 16; k = 16};
    cfg_a = s.operand;
    cfg_b = s.operand;
    cfg_c = s.acc;
    cfg_result = s.acc;
    cfg_saturating = false;
    cfg_scope = C.Subgroup;
  }

let address_env (k : kernel) (g : P.gemm) : Affine.env =
  let scalars =
    List.filter_map
      (function DParam (v, None) -> Some v.var_id | _ -> None)
      k.kern_params
  in
  {
    classify_var =
      (fun v ->
        if List.exists (fun w -> w.var_id = v.var_id) g.enclosing_loops then
          Affine.Atom_leaf
        else if List.mem v.var_id scalars then Affine.Uniform_leaf
        else Affine.Rejected_leaf);
    definition = (fun _ -> None);
  }

(* The region parameters appended to the original's, under their own names
   (not [new_var]'s prefixed ones: callers pass them by position). *)
let extra_param fresh name =
  fresh.next <- fresh.next + 1 ;
  {var_name = name; var_id = fresh.next; var_type = TInt32; var_mutable = false}

(* A bound, initial value or epilogue that reads block_dim/grid_dim changes
   meaning under the rewritten launch's own dimensions. *)
let uses_launch_dims (g : P.gemm) =
  let es =
    [g.k_lo; g.k_hi; g.init]
    @ match g.epilogue with P.Epilogue e -> [e] | P.Plain -> []
  in
  List.exists Sarek_ir_eval_int.mentions_launch_dims es
