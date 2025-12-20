(******************************************************************************
 * Unit tests for Sarek_lower
 *
 * Tests lowering from typed AST to Kirc_Ast.
 ******************************************************************************)

open Sarek_ppx_lib.Sarek_ast
open Sarek_ppx_lib.Sarek_types
open Sarek_ppx_lib.Sarek_typed_ast
open Sarek_ppx_lib.Sarek_lower
open Sarek_ppx_lib.Kirc_Ast

let dummy_loc = {
  loc_file = "test";
  loc_line = 1;
  loc_col = 0;
  loc_end_line = 1;
  loc_end_col = 0;
}

(* Helper to create typed expressions *)
let mk_texpr te ty = { te; ty; te_loc = dummy_loc }

let int_texpr i = mk_texpr (TEInt i) t_int32
let float_texpr f = mk_texpr (TEFloat f) t_float32
let bool_texpr b = mk_texpr (TEBool b) t_bool
let var_texpr name id ty = mk_texpr (TEVar (name, id)) ty

(* Helper to check IR structure *)
let check_ir_is msg ir expected_constructor =
  if expected_constructor ir then
    Alcotest.(check pass) msg () ()
  else
    Alcotest.failf "%s: unexpected IR structure" msg

(* Test lowering literals *)
let test_lower_int () =
  let te = int_texpr 42 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "int literal lowers to Int"
    ir (function Int 42 -> true | _ -> false)

let test_lower_float () =
  let te = float_texpr 3.14 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "float literal lowers to Float"
    ir (function Float f -> abs_float (f -. 3.14) < 0.001 | _ -> false)

let test_lower_bool_true () =
  let te = bool_texpr true in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "true lowers to Int 1"
    ir (function Int 1 -> true | _ -> false)

let test_lower_bool_false () =
  let te = bool_texpr false in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "false lowers to Int 0"
    ir (function Int 0 -> true | _ -> false)

let test_lower_unit () =
  let te = mk_texpr TEUnit t_unit in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "unit lowers to Unit"
    ir (function Unit -> true | _ -> false)

(* Test lowering variables *)
let test_lower_int_var () =
  let te = var_texpr "x" 0 t_int32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "int var lowers to IntVar"
    ir (function IntVar (0, "x") -> true | _ -> false)

let test_lower_float_var () =
  let te = var_texpr "y" 1 t_float32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "float var lowers to FloatVar"
    ir (function FloatVar (1, "y") -> true | _ -> false)

let test_lower_double_var () =
  let te = var_texpr "z" 2 t_float64 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "double var lowers to DoubleVar"
    ir (function DoubleVar (2, "z") -> true | _ -> false)

(* Test lowering binary operations *)
let test_lower_add_int () =
  let te = mk_texpr
      (TEBinop (Add, int_texpr 1, int_texpr 2))
      t_int32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "int + int lowers to Plus"
    ir (function Plus (Int 1, Int 2) -> true | _ -> false)

let test_lower_add_float () =
  let te = mk_texpr
      (TEBinop (Add, float_texpr 1.0, float_texpr 2.0))
      t_float32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "float + float lowers to Plusf"
    ir (function Plusf (Float _, Float _) -> true | _ -> false)

let test_lower_sub_int () =
  let te = mk_texpr
      (TEBinop (Sub, int_texpr 5, int_texpr 3))
      t_int32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "int - int lowers to Min"
    ir (function Min (Int 5, Int 3) -> true | _ -> false)

let test_lower_mul_float () =
  let te = mk_texpr
      (TEBinop (Mul, float_texpr 2.0, float_texpr 3.0))
      t_float32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "float * float lowers to Mulf"
    ir (function Mulf (Float _, Float _) -> true | _ -> false)

let test_lower_div_int () =
  let te = mk_texpr
      (TEBinop (Div, int_texpr 10, int_texpr 2))
      t_int32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "int / int lowers to Div"
    ir (function Div (Int 10, Int 2) -> true | _ -> false)

let test_lower_mod () =
  let te = mk_texpr
      (TEBinop (Mod, int_texpr 10, int_texpr 3))
      t_int32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "int mod int lowers to Mod"
    ir (function Mod (Int 10, Int 3) -> true | _ -> false)

let test_lower_eq () =
  let te = mk_texpr
      (TEBinop (Eq, int_texpr 1, int_texpr 2))
      t_bool in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "int = int lowers to EqBool"
    ir (function EqBool (Int 1, Int 2) -> true | _ -> false)

let test_lower_lt () =
  let te = mk_texpr
      (TEBinop (Lt, int_texpr 1, int_texpr 2))
      t_bool in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "int < int lowers to LtBool"
    ir (function LtBool (Int 1, Int 2) -> true | _ -> false)

let test_lower_and () =
  let te = mk_texpr
      (TEBinop (And, bool_texpr true, bool_texpr false))
      t_bool in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "bool && bool lowers to And"
    ir (function And (Int 1, Int 0) -> true | _ -> false)

let test_lower_or () =
  let te = mk_texpr
      (TEBinop (Or, bool_texpr true, bool_texpr false))
      t_bool in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "bool || bool lowers to Or"
    ir (function Or (Int 1, Int 0) -> true | _ -> false)

(* Test lowering unary operations *)
let test_lower_not () =
  let te = mk_texpr
      (TEUnop (Not, bool_texpr true))
      t_bool in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "not bool lowers to Not"
    ir (function Not (Int 1) -> true | _ -> false)

let test_lower_neg_int () =
  let te = mk_texpr
      (TEUnop (Neg, int_texpr 42))
      t_int32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "-int lowers to Min(0, x)"
    ir (function Min (Int 0, Int 42) -> true | _ -> false)

let test_lower_neg_float () =
  let te = mk_texpr
      (TEUnop (Neg, float_texpr 3.14))
      t_float32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "-float lowers to Minf(0.0, x)"
    ir (function Minf (Float _, Float _) -> true | _ -> false)

(* Test lowering control flow *)
let test_lower_if () =
  let te = mk_texpr
      (TEIf (bool_texpr true, int_texpr 1, Some (int_texpr 2)))
      t_int32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "if-then-else lowers to Ife"
    ir (function Ife (Int 1, Int 1, Int 2) -> true | _ -> false)

let test_lower_if_no_else () =
  let te = mk_texpr
      (TEIf (bool_texpr true, mk_texpr TEUnit t_unit, None))
      t_unit in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "if-then (no else) lowers to If"
    ir (function If (Int 1, Unit) -> true | _ -> false)

let test_lower_while () =
  let te = mk_texpr
      (TEWhile (bool_texpr true, mk_texpr TEUnit t_unit))
      t_unit in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "while lowers to While"
    ir (function While (Int 1, Unit) -> true | _ -> false)

let test_lower_for () =
  let te = mk_texpr
      (TEFor ("i", 0, int_texpr 0, int_texpr 10, Upto, mk_texpr TEUnit t_unit))
      t_unit in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "for lowers to DoLoop"
    ir (function DoLoop (IntVar (0, "i"), Int 0, Int 10, Unit) -> true | _ -> false)

(* Test lowering sequence *)
let test_lower_seq () =
  let te = mk_texpr
      (TESeq [int_texpr 1; int_texpr 2; int_texpr 3])
      t_int32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "seq lowers to nested Seq"
    ir (function Seq (Int 1, Seq (Int 2, Int 3)) -> true | _ -> false)

(* Test lowering return *)
let test_lower_return () =
  let te = mk_texpr
      (TEReturn (int_texpr 42))
      t_int32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "return lowers to Return"
    ir (function Return (Int 42) -> true | _ -> false)

(* Test lowering intrinsics *)
let test_lower_intrinsic_const () =
  let te = mk_texpr
      (TEIntrinsicConst ("threadIdx.x", "get_local_id(0)"))
      t_int32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "intrinsic const lowers to Intrinsics"
    ir (function Intrinsics ("threadIdx.x", "get_local_id(0)") -> true | _ -> false)

(* Test lowering vector access *)
let test_lower_vec_get () =
  let vec = var_texpr "v" 0 (TVec t_float32) in
  let idx = int_texpr 5 in
  let te = mk_texpr (TEVecGet (vec, idx)) t_float32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "v.[i] lowers to IntVecAcc"
    ir (function IntVecAcc (VecVar (_, 0, "v"), Int 5) -> true | _ -> false)

let test_lower_vec_set () =
  let vec = var_texpr "v" 0 (TVec t_float32) in
  let idx = int_texpr 5 in
  let value = float_texpr 3.14 in
  let te = mk_texpr (TEVecSet (vec, idx, value)) t_unit in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "v.[i] <- x lowers to SetV(IntVecAcc, x)"
    ir (function SetV (IntVecAcc (VecVar _, Int 5), Float _) -> true | _ -> false)

(* Test lowering let *)
let test_lower_let () =
  let value = int_texpr 42 in
  let body = var_texpr "x" 0 t_int32 in
  let te = mk_texpr (TELet ("x", 0, value, body)) t_int32 in
  let state = create_state () in
  let ir = lower_expr state te in
  check_ir_is "let x = 42 in x lowers to Seq(Decl, Seq(Set, body))"
    ir (function
        | Seq (Decl (IntVar (0, "x")),
               Seq (Set (IntVar (0, "x"), Int 42),
                    IntVar (0, "x"))) -> true
        | _ -> false)

(* Test suite *)
let () =
  Alcotest.run "Sarek_lower" [
    "literals", [
      Alcotest.test_case "int" `Quick test_lower_int;
      Alcotest.test_case "float" `Quick test_lower_float;
      Alcotest.test_case "bool true" `Quick test_lower_bool_true;
      Alcotest.test_case "bool false" `Quick test_lower_bool_false;
      Alcotest.test_case "unit" `Quick test_lower_unit;
    ];
    "variables", [
      Alcotest.test_case "int var" `Quick test_lower_int_var;
      Alcotest.test_case "float var" `Quick test_lower_float_var;
      Alcotest.test_case "double var" `Quick test_lower_double_var;
    ];
    "binop", [
      Alcotest.test_case "add int" `Quick test_lower_add_int;
      Alcotest.test_case "add float" `Quick test_lower_add_float;
      Alcotest.test_case "sub int" `Quick test_lower_sub_int;
      Alcotest.test_case "mul float" `Quick test_lower_mul_float;
      Alcotest.test_case "div int" `Quick test_lower_div_int;
      Alcotest.test_case "mod" `Quick test_lower_mod;
      Alcotest.test_case "eq" `Quick test_lower_eq;
      Alcotest.test_case "lt" `Quick test_lower_lt;
      Alcotest.test_case "and" `Quick test_lower_and;
      Alcotest.test_case "or" `Quick test_lower_or;
    ];
    "unop", [
      Alcotest.test_case "not" `Quick test_lower_not;
      Alcotest.test_case "neg int" `Quick test_lower_neg_int;
      Alcotest.test_case "neg float" `Quick test_lower_neg_float;
    ];
    "control_flow", [
      Alcotest.test_case "if" `Quick test_lower_if;
      Alcotest.test_case "if no else" `Quick test_lower_if_no_else;
      Alcotest.test_case "while" `Quick test_lower_while;
      Alcotest.test_case "for" `Quick test_lower_for;
    ];
    "other", [
      Alcotest.test_case "seq" `Quick test_lower_seq;
      Alcotest.test_case "return" `Quick test_lower_return;
      Alcotest.test_case "intrinsic const" `Quick test_lower_intrinsic_const;
    ];
    "vector", [
      Alcotest.test_case "vec get" `Quick test_lower_vec_get;
      Alcotest.test_case "vec set" `Quick test_lower_vec_set;
    ];
    "let", [
      Alcotest.test_case "let" `Quick test_lower_let;
    ];
  ]
