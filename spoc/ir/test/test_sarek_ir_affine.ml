(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_ir_affine (specs/matrix-pattern-advice.md).
 ******************************************************************************)

open Sarek_ir_types
module A = Sarek_ir_affine

let var ?(mutable_ = false) name id =
  {var_name = name; var_id = id; var_type = TInt32; var_mutable = mutable_}

let k = var "k" 1 (* scalar parameter: uniform *)

let n = var "n" 2

let i = var "i" 3 (* reduction loop variable: atom *)

let acc = var "acc" 4 (* mutable: rejected *)

let row = var "row" 5 (* let row = global_idx_y *)

let gid name = EIntrinsic (["Sarek_stdlib"; "Gpu"], name, [])

let c x = EConst (CInt32 (Int32.of_int x))

let ( + ) a b = EBinop (Add, a, b)

let ( * ) a b = EBinop (Mul, a, b)

let env : A.env =
  {
    classify_var =
      (fun v ->
        if v.var_id = i.var_id then A.Atom_leaf
        else if v.var_id = k.var_id || v.var_id = n.var_id then A.Uniform_leaf
        else A.Rejected_leaf);
    definition =
      (fun v ->
        if v.var_id = row.var_id then Some (gid "global_idx_y") else None);
  }

let form e =
  match A.of_expr env e with
  | Ok f -> f
  | Error _ -> Alcotest.fail "expected an affine form"

let is_one p = A.poly_is_one p

(* row * k + i: row atom (through the let) with stride k, i with stride 1. *)
let test_row_major_index () =
  let f = form ((EVar row * EVar k) + EVar i) in
  Alcotest.(check bool)
    "i coefficient 1"
    true
    (is_one (A.coefficient f (A.var_atom i))) ;
  let stride = A.coefficient f (gid "global_idx_y") in
  Alcotest.(check bool)
    "row coefficient is k"
    true
    (A.poly_equal stride (form (EVar k)).const) ;
  Alcotest.(check int) "two atoms" 2 (List.length (A.atoms f))

(* The loop variable is canonical whatever its binder's mutability. *)
let test_var_atom_canonical () =
  let i_binder = {i with var_mutable = true} in
  let f = form (EVar i) in
  Alcotest.(check bool)
    "binder and use agree"
    true
    (is_one (A.coefficient f (A.var_atom i_binder)))

(* (k + 1) * row distributes; a uniform product stays a coefficient. *)
let test_distribution () =
  (* A sum multiplying an atom stays one factor, so a padded stride divides. *)
  let f = form ((EVar k + c 1) * gid "global_idx_x") in
  let coef = A.coefficient f (gid "global_idx_x") in
  Alcotest.(check int) "(k + 1) * x: one monomial" 1 (List.length coef) ;
  let by = gid "block_idx_y" in
  let padded = form (gid "global_idx_x" * (EVar k + EVar n)) in
  let h = form (gid "block_dim_y" * by * (EVar k + EVar n)) in
  (match
     A.poly_div (A.coefficient h by) (A.coefficient padded (gid "global_idx_x"))
   with
  | Some q ->
      Alcotest.(check bool)
        "bdy * (k + n) / (k + n) = bdy"
        true
        (A.poly_equal q (form (gid "block_dim_y")).const)
  | None -> Alcotest.fail "padded stride does not divide") ;
  (* Two uniform sides still distribute. *)
  let u = form ((EVar k + c 1) * EVar n) in
  Alcotest.(check int) "(k + 1) * n: two monomials" 2 (List.length u.const) ;
  let g = form (EVar k * EVar n) in
  Alcotest.(check bool) "k * n is uniform" true (A.is_uniform g)

let test_thread_products_are_atoms () =
  (* tid / n is one opaque thread-varying atom; so is tx * ty. *)
  let f = form (EBinop (Div, gid "global_thread_id", EVar n)) in
  Alcotest.(check int) "tid / n: one atom" 1 (List.length (A.atoms f)) ;
  let g = form (gid "thread_idx_x" * gid "thread_idx_y") in
  Alcotest.(check int) "tx * ty: one atom" 1 (List.length (A.atoms g)) ;
  let u = form (gid "block_dim_x" * gid "block_idx_x") in
  Alcotest.(check bool)
    "block_dim_x is a uniform coefficient"
    true
    (A.coefficient u (gid "block_idx_x") <> [])

let expect_error name e pred =
  match A.of_expr env e with
  | Ok _ -> Alcotest.failf "%s: expected a rejection" name
  | Error err -> Alcotest.(check bool) name true (pred err)

let test_rejections () =
  expect_error "i * i" (EVar i * EVar i) (function
    | A.Not_affine _ -> true
    | _ -> false) ;
  expect_error
    "i mod 7"
    (EBinop (Mod, EVar i, c 7))
    (function A.Not_affine _ -> true | _ -> false) ;
  expect_error
    "perm[i]"
    (EArrayRead ("perm", EVar i))
    (function A.Data_dependent _ -> true | _ -> false) ;
  expect_error
    "perm[row]"
    (EArrayRead ("perm", EVar row) + EVar i)
    (function A.Data_dependent _ -> true | _ -> false) ;
  expect_error "mutable" (EVar acc + EVar i) (function
    | A.Unknown_var _ -> true
    | _ -> false)

let test_poly_div () =
  let kp = (form (EVar k)).const in
  let kn = A.poly_mul kp (form (EVar n)).const in
  (match A.poly_div kn kp with
  | Some q ->
      Alcotest.(check bool)
        "k*n / k = n"
        true
        (A.poly_equal q (form (EVar n)).const)
  | None -> Alcotest.fail "k*n / k") ;
  Alcotest.(check bool)
    "n / k undefined"
    true
    (A.poly_div (form (EVar n)).const kp = None) ;
  Alcotest.(check bool)
    "6 / 4 undefined"
    true
    (A.poly_div (A.poly_const 6) (A.poly_const 4) = None)

let () =
  Alcotest.run
    "sarek_ir_affine"
    [
      ( "forms",
        [
          Alcotest.test_case "row-major index" `Quick test_row_major_index;
          Alcotest.test_case
            "canonical loop variable"
            `Quick
            test_var_atom_canonical;
          Alcotest.test_case "distribution" `Quick test_distribution;
          Alcotest.test_case
            "opaque atoms"
            `Quick
            test_thread_products_are_atoms;
          Alcotest.test_case "rejections" `Quick test_rejections;
          Alcotest.test_case "polynomial division" `Quick test_poly_div;
        ] );
    ]
