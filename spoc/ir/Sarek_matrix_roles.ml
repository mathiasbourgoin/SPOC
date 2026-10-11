(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Index roles of a GEMM nest: which atoms form the row (M), column (N) and
   batch indices, and each operand's strides. specs/matrix-pattern-advice.md. *)

open Sarek_ir_types
open Sarek_matrix_pattern_types
module Affine = Sarek_ir_affine

let ( let* ) = Result.bind

let affine_of env arr idx =
  match Affine.of_expr env idx with
  | Ok f -> Ok f
  | Error err -> Error (Index_not_affine (arr, Affine.error_expr err))

(* A tile is loaded with one base and one leading dimension: one stride must
   be 1 and the rows (or columns) must not overlap. Equal strides ([a.(row + i)],
   [c.(row + col)]: Toeplitz/Hankel views, correlations, races) overlap. A
   symbolic leading dimension is trusted ([ld = k] cannot be compared with the
   bounds without the launch). *)
let layout_of arr ~row_stride ~col_stride ~row_extent ~col_extent =
  (* A constant leading dimension is accepted only when the contiguous extent
     is a known constant that it covers, and it is at least a tile: otherwise
     rows may overlap ([a.(row * 32 + i)] with [i] up to [k - 1]). *)
  let constant_ld_unproven ld extent =
    match Affine.poly_constant ld with
    | Some c -> (
        c < 16
        ||
        match Option.bind extent Affine.poly_constant with
        | Some e -> c < e
        | None -> true)
    | None -> false
  in
  if Affine.poly_equal row_stride col_stride then
    Error (Overlapping_operand arr)
  else if Affine.poly_is_one col_stride then
    if constant_ld_unproven row_stride col_extent then
      Error (Constant_leading_dimension arr)
    else Ok Row_major
  else if Affine.poly_is_one row_stride then
    if constant_ld_unproven col_stride row_extent then
      Error (Constant_leading_dimension arr)
    else Ok Col_major
  else Error (Operand_not_unit_stride arr)

let make_operand ?row_extent ?col_extent (arr, idx, elt) form ~row_stride
    ~col_stride ~excluded =
  let* layout = layout_of arr ~row_stride ~col_stride ~row_extent ~col_extent in
  Ok
    {
      array = arr;
      elt;
      index = idx;
      row_stride = Affine.poly_to_expr row_stride;
      col_stride = Affine.poly_to_expr col_stride;
      layout;
      offset = Affine.to_expr (Affine.without form excluded);
    }

let diff xs ys = List.filter (fun x -> not (List.mem x ys)) xs

let inter xs ys = List.filter (fun x -> List.mem x ys) xs

(* A row (or column) index may be several atoms: [thread_idx_y + block_dim_y *
   block_idx_y] is two. They form ONE index when their coefficients in the
   operand and in C are the same weights scaled by one stride each:
   [coef_X(a) = w_a * s_X] and [coef_C(a) = w_a * s_C]. The reference atom
   [a0] has weight 1; the strides are its coefficients. *)
type group = {
  g_index : Affine.t;
  g_atoms : expr list;
  s_x : Affine.poly;
  s_c : Affine.poly;
}

let weight_of fx fc ~s_x ~s_c a =
  let coef_c = Affine.coefficient fc a in
  let w =
    if Affine.poly_equal coef_c s_c then Some (Affine.poly_const 1)
    else Affine.poly_div coef_c s_c
  in
  match w with
  | Some w
    when Affine.poly_equal (Affine.coefficient fx a) (Affine.poly_mul w s_x) ->
      Some (a, w)
  | _ -> None

(* The largest set of [atoms] whose weights are consistent with one reference
   atom. Atoms left out are not part of this index: they become batch atoms
   (indices over which the other operand is broadcast). *)
let group_of fx fc atoms =
  let attempt a0 =
    let s_x = Affine.coefficient fx a0 and s_c = Affine.coefficient fc a0 in
    let weights = List.filter_map (weight_of fx fc ~s_x ~s_c) atoms in
    {
      g_index = Affine.of_terms weights;
      g_atoms = List.map fst weights;
      s_x;
      s_c;
    }
  in
  List.fold_left
    (fun best a0 ->
      let g = attempt a0 in
      match best with
      | _ when g.g_atoms = [] -> best
      | Some b when List.length b.g_atoms >= List.length g.g_atoms -> best
      | _ -> Some g)
    None
    atoms

(* The thread-varying leaves of an atom: intrinsics without arguments and
   loop variables, not the uniform ones. *)
let rec varying_leaves env e =
  match e with
  | EIntrinsic (_, _, []) | EVar _ -> (
      match Affine.of_expr env e with
      | Ok f when Affine.is_uniform f -> []
      | _ -> [e])
  | EBinop (_, a, b) -> varying_leaves env a @ varying_leaves env b
  | EUnop (_, a) | ECast (_, a) -> varying_leaves env a
  | EIntrinsic (_, _, args) -> List.concat_map (varying_leaves env) args
  | _ -> []

(* Two indices computed from the same thread (or loop) value are independent
   only as the [t / d], [t mod d] pair: [tid / 4] and [tid / 8] cover a band of
   C, not M x N. *)
let independent env ma na =
  let shares a b =
    List.exists
      (fun l -> List.mem l (varying_leaves env b))
      (varying_leaves env a)
  in
  List.for_all
    (fun a ->
      List.for_all
        (fun b ->
          (not (shares a b))
          ||
          match (a, b) with
          | EBinop (Div, t, d), EBinop (Mod, t', d')
          | EBinop (Mod, t', d'), EBinop (Div, t, d) ->
              t = t' && d = d'
          | _ -> false)
        na)
    ma

type roles = {gm : group; gn : group; batch_atoms : expr list}

let split_roles env kv_atom (px, fx) (py, fy) fc =
  let atoms_x = diff (Affine.atoms fx) [kv_atom] in
  let atoms_y = diff (Affine.atoms fy) [kv_atom] in
  let atoms_c = Affine.atoms fc in
  let m_atoms = diff (inter atoms_x atoms_c) atoms_y in
  let n_atoms = diff (inter atoms_y atoms_c) atoms_x in
  match (group_of fx fc m_atoms, group_of fy fc n_atoms) with
  | Some gm, Some gn ->
      let batch_atoms = diff atoms_c (gm.g_atoms @ gn.g_atoms) in
      if not (independent env gm.g_atoms gn.g_atoms) then
        Error Dependent_indices
      else if diff atoms_x (gm.g_atoms @ batch_atoms) <> [] then
        Error (Stray_index_atom px)
      else if diff atoms_y (gn.g_atoms @ batch_atoms) <> [] then
        Error (Stray_index_atom py)
      else Ok {gm; gn; batch_atoms}
  | _ -> Error No_row_column_split

let reduction_forms env kv (ox, oy) (c, ic) =
  let arr (a, _, _) = a in
  let idx (_, i, _) = i in
  let* fx = affine_of env (arr ox) (idx ox) in
  let* fy = affine_of env (arr oy) (idx oy) in
  let* fc = affine_of env c ic in
  let kv_atom = Affine.var_atom kv in
  let depends f = not (Affine.poly_is_zero (Affine.coefficient f kv_atom)) in
  if not (depends fx) then Error (Operand_independent_of_reduction (arr ox))
  else if not (depends fy) then
    Error (Operand_independent_of_reduction (arr oy))
  else if depends fc then Error Output_depends_on_reduction
  else Ok (fx, fy, fc)

let index_roles env kv ~k_extent (ox, oy) (c, ic, ec) =
  let* fx, fy, fc = reduction_forms env kv (ox, oy) (c, ic) in
  let arr (a, _, _) = a in
  let kv_atom = Affine.var_atom kv in
  let* r = split_roles env kv_atom (arr ox, fx) (arr oy, fy) fc in
  (* Orient so that C is row-major whenever either orientation allows it. *)
  let swap =
    (not (Affine.poly_is_one r.gn.s_c)) && Affine.poly_is_one r.gm.s_c
  in
  let (oa, fa), (ob, fb), gm, gn =
    if swap then ((oy, fy), (ox, fx), r.gn, r.gm)
    else ((ox, fx), (oy, fy), r.gm, r.gn)
  in
  let k_of f = Affine.coefficient f kv_atom in
  let* a =
    make_operand
      ?col_extent:k_extent
      oa
      fa
      ~row_stride:gm.s_x
      ~col_stride:(k_of fa)
      ~excluded:(kv_atom :: gm.g_atoms)
  in
  let* b =
    make_operand
      ?row_extent:k_extent
      ob
      fb
      ~row_stride:(k_of fb)
      ~col_stride:gn.s_x
      ~excluded:(kv_atom :: gn.g_atoms)
  in
  let* c =
    make_operand
      (c, ic, ec)
      fc
      ~row_stride:gm.s_c
      ~col_stride:gn.s_c
      ~excluded:(gm.g_atoms @ gn.g_atoms)
  in
  Ok
    ( a,
      b,
      c,
      (Affine.to_expr gm.g_index, Affine.to_expr gn.g_index, r.batch_atoms) )
