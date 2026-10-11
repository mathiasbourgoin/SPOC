(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Unit tests for the variance analysis (Sarek_variance, Sarek_variance_ir)
   and the checks built on it (specs/convergence-semantics.md §3-§5), on
   hand-built typed ASTs. *)

let () = Sarek_stdlib.force_init ()

open Sarek_typed_ast

let loc = Sarek_ast.dummy_loc

let mk ?(ty = Sarek_types.t_unit) te = {te; ty; te_loc = loc}

let int_ty = Sarek_types.t_int32

let i32 n = mk ~ty:int_ty (TEInt32 n)

let var n = mk ~ty:int_ty (TEVar (n, 0))

let tid () = var "thread_idx_x"

let gt a b = mk ~ty:Sarek_types.t_bool (TEBinop (Sarek_ast.Gt, a, b))

let barrier () =
  mk
    (TEIntrinsicFun
       ( Sarek_env.CorePrimitiveRef "block_barrier",
         Some Sarek_core_primitives.ConvergencePoint,
         [] ))

let if_ c t = mk (TEIf (c, t, None))

let seq es = mk (TESeq es)

let let_mut n v b = mk (TELetMut (n, 0, v, b))

let let_ n v b = mk ~ty:b.ty (TELet (n, 0, v, b))

let assign n v = mk (TEAssign (n, 0, v))

let rejected e = Sarek_convergence.check_expr Sarek_convergence.init_ctx e <> []

let check_rejected name expected e () =
  Alcotest.(check bool) name expected (rejected e)

(* C-1: assigned a varying value *)
let mut_assign =
  let_mut
    "x"
    (i32 0l)
    (seq [assign "x" (tid ()); if_ (gt (var "x") (i32 16l)) (barrier ())])

(* C-1: control dependence *)
let ctrl_dep =
  let_mut
    "x"
    (i32 0l)
    (seq
       [
         if_ (gt (tid ()) (i32 16l)) (assign "x" (i32 1l));
         if_ (gt (var "x") (i32 0l)) (barrier ());
       ])

(* Control dependence is relative to the declaration: [y] lives inside the
   varying branch; the barrier is outside it. *)
let inner_decl =
  seq
    [
      if_
        (gt (tid ()) (i32 16l))
        (let_mut "y" (i32 0l) (seq [assign "y" (i32 1l)]));
      barrier ();
    ]

(* Shadowing: the inner [x] is uniform, the outer one varies. *)
let shadowing =
  let_
    "x"
    (tid ())
    (let_ "x" (i32 3l) (if_ (gt (var "x") (i32 0l)) (barrier ())))

(* C-2: if-expression value *)
let if_value =
  let_
    "x"
    (mk ~ty:int_ty (TEIf (gt (tid ()) (i32 16l), i32 1l, Some (i32 0l))))
    (if_ (gt (var "x") (i32 0l)) (barrier ()))

(* C-4: an intrinsic declared ThreadVarying (atomics) *)
let atomic_result =
  let call =
    mk
      ~ty:int_ty
      (TEIntrinsicFun
         (Sarek_env.CorePrimitiveRef "atomic_add_int32", None, [i32 0l; i32 1l]))
  in
  let_ "old" call (if_ (gt (var "old") (i32 0l)) (barrier ()))

(* C-5: private array read vs shared array read *)
let arr_ty mem = Sarek_types.TArr (int_ty, mem)

let arr_read name =
  mk
    ~ty:int_ty
    (TEArrGet (mk ~ty:(arr_ty Sarek_types.Local) (TEVar (name, 0)), i32 0l))

let private_read =
  let_
    "a"
    (mk
       ~ty:(arr_ty Sarek_types.Local)
       (TECreateArray (i32 4l, int_ty, Sarek_types.Local)))
    (if_ (gt (arr_read "a") (i32 0l)) (barrier ()))

let shared_read =
  mk
    (TELetShared
       ( "s",
         0,
         int_ty,
         Some (i32 4l),
         if_ (gt (arr_read "s") (i32 0l)) (barrier ()) ))

(* C-3: local helper (let rec) with a parameter-dependent barrier *)
let helper_call arg =
  let p =
    {
      tparam_name = "m";
      tparam_type = int_ty;
      tparam_index = 0;
      tparam_is_vec = false;
      tparam_id = 0;
    }
  in
  mk
    (TELetRec
       ( "f",
         0,
         [p],
         if_ (gt (var "m") (i32 0l)) (barrier ()),
         mk (TEApp (mk (TEVar ("f", 0)), [arg])) ))

(* C-24: barrier in the right operand of && *)
let short_circuit =
  let rhs = seq [barrier (); mk ~ty:Sarek_types.t_bool (TEBool true)] in
  if_
    (mk
       ~ty:Sarek_types.t_bool
       (TEBinop (Sarek_ast.And, gt (tid ()) (i32 1l), rhs)))
    (mk TEUnit)

let test_value () =
  Alcotest.(check bool)
    "tid varies"
    true
    (Sarek_convergence.is_thread_varying (tid ())) ;
  Alcotest.(check bool)
    "let value is its body"
    false
    (Sarek_convergence.is_thread_varying (let_ "x" (tid ()) (i32 0l)))

let () =
  Alcotest.run
    "Sarek_variance"
    [
      ( "variance",
        [
          Alcotest.test_case "values" `Quick test_value;
          Alcotest.test_case
            "C-1 assign"
            `Quick
            (check_rejected "rejected" true mut_assign);
          Alcotest.test_case
            "C-1 control dependence"
            `Quick
            (check_rejected "rejected" true ctrl_dep);
          Alcotest.test_case
            "C-1 relative to declaration"
            `Quick
            (check_rejected "accepted" false inner_decl);
          Alcotest.test_case
            "lexical shadowing"
            `Quick
            (check_rejected "accepted" false shadowing);
          Alcotest.test_case
            "C-2 if value"
            `Quick
            (check_rejected "rejected" true if_value);
          Alcotest.test_case
            "C-4 atomic"
            `Quick
            (check_rejected "rejected" true atomic_result);
          Alcotest.test_case
            "C-5 private read"
            `Quick
            (check_rejected "rejected" true private_read);
          Alcotest.test_case
            "shared read uniform"
            `Quick
            (check_rejected "accepted" false shared_read);
          Alcotest.test_case
            "C-3 helper varying arg"
            `Quick
            (check_rejected "rejected" true (helper_call (tid ())));
          Alcotest.test_case
            "C-3 helper uniform arg"
            `Quick
            (check_rejected "accepted" false (helper_call (i32 3l)));
          Alcotest.test_case
            "C-24 short circuit"
            `Quick
            (check_rejected "rejected" true short_circuit);
        ] );
    ]
