(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Production link (specs/convergence-semantics.md §6): the SHIPPED checker
 * Sarek_convergence.check_expr against the extracted Rocq model
 * (ConvergenceSpec.check_env, ConvergenceSemantics.check_warp_env,
 * ConvergenceSpec.is_varying_in_env) on randomly generated model terms.
 *
 * Each model [expr] is translated to a typed AST:
 *   ELit       -> 0l                    EVary   -> thread_idx_x
 *   EBarrier   -> block_barrier ()      EWarpPoint -> a warp collective
 *   EVar x     -> variable "x<x>"      ELet x  -> let "x<x>" = ... in ...
 *   EBinop     -> +                     EUnop   -> unary minus
 *   EIf/EWhile/EFor/ESeq/ESuperstep/EReturn -> the same construct
 *   EApp args  -> a pure, uniform intrinsic applied to args
 * Names resolve lexically in production, exactly as the model's env does.
 * The warp collective and the pure intrinsic are names absent from the
 * primitive table, so their declared variance is uniform, as in the model
 * (EWarpPoint and EApp are not sources of variance there).
 *
 * Properties, in both modes: production reports a barrier error iff
 * check_env reports an error; a warp error iff check_warp_env does; and the
 * value variance agrees with is_varying_in_env. Sampled, not exhaustive.
 ******************************************************************************)

open Sarek_typed_ast
module M = Convergence_model.ConvergenceModel

let mk ?(ty = Sarek_types.t_unit) te = {te; ty; te_loc = Sarek_ast.dummy_loc}

let var_name x = "x" ^ string_of_int x

let barrier () =
  TEIntrinsicFun
    ( Sarek_env.CorePrimitiveRef "block_barrier",
      Some Sarek_core_primitives.ConvergencePoint,
      [] )

let warp_point () =
  TEIntrinsicFun
    ( Sarek_env.CorePrimitiveRef "link_warp_collective",
      Some Sarek_core_primitives.WarpConvergence,
      [] )

let rec to_texpr (e : M.expr) : texpr =
  match e with
  | M.ELit -> mk (TEInt32 0l)
  | M.EVary -> mk (TEVar ("thread_idx_x", 0))
  | M.EBarrier -> mk (barrier ())
  | M.EWarpPoint -> mk (warp_point ())
  | M.EVar x -> mk (TEVar (var_name x, 0))
  | M.EBinop (a, b) -> mk (TEBinop (Sarek_ast.Add, to_texpr a, to_texpr b))
  | M.EUnop a -> mk (TEUnop (Sarek_ast.Neg, to_texpr a))
  | M.EIf (c, t, el) -> mk (TEIf (to_texpr c, to_texpr t, Some (to_texpr el)))
  | M.EWhile (c, b) -> mk (TEWhile (to_texpr c, to_texpr b))
  | M.EFor (lo, hi, b) ->
      mk
        (TEFor
           ("loop_i", 0, to_texpr lo, to_texpr hi, Sarek_ast.Upto, to_texpr b))
  | M.ESeq es -> mk (TESeq (List.map to_texpr es))
  | M.ELet (x, v, b) -> mk (TELet (var_name x, 0, to_texpr v, to_texpr b))
  | M.ESuperstep (dv, b, c) ->
      mk (TESuperstep ("s", dv, to_texpr b, to_texpr c))
  | M.EApp args ->
      mk
        (TEIntrinsicFun
           ( Sarek_env.CorePrimitiveRef "link_pure_function",
             None,
             List.map to_texpr args ))
  | M.EReturn a -> mk (TEReturn (to_texpr a))

(* ── generator: the model's full constructor set ────────────────────────── *)

open QCheck2

let gen_var_id = Gen.int_range 0 3

let gen_expr : M.expr Gen.t =
  Gen.sized_size (Gen.int_range 0 6)
  @@ Gen.fix (fun self n ->
      let leaves = [M.ELit; M.EVary; M.EBarrier; M.EWarpPoint] in
      if n = 0 then
        Gen.oneof
          [Gen.oneof_list leaves; Gen.map (fun x -> M.EVar x) gen_var_id]
      else
        let sub = self (n / 2) in
        let sub2 = Gen.pair sub sub and sub3 = Gen.triple sub sub sub in
        let sublist = Gen.list_size (Gen.int_range 0 3) sub in
        Gen.oneof
          [
            Gen.oneof_list leaves;
            Gen.map (fun x -> M.EVar x) gen_var_id;
            Gen.map (fun (a, b) -> M.EBinop (a, b)) sub2;
            Gen.map (fun e -> M.EUnop e) sub;
            Gen.map (fun (c, t, el) -> M.EIf (c, t, el)) sub3;
            Gen.map (fun (c, b) -> M.EWhile (c, b)) sub2;
            Gen.map (fun (lo, hi, b) -> M.EFor (lo, hi, b)) sub3;
            Gen.map (fun es -> M.ESeq es) sublist;
            Gen.map
              (fun (x, (v, b)) -> M.ELet (x, v, b))
              (Gen.pair gen_var_id sub2);
            Gen.map
              (fun (dv, (b, c)) -> M.ESuperstep (dv, b, c))
              (Gen.pair Gen.bool sub2);
            Gen.map (fun args -> M.EApp args) sublist;
            Gen.map (fun e -> M.EReturn e) sub;
          ])

let gen_mode = Gen.oneof_list [M.Converged; M.Diverged]

(* ── production side ─────────────────────────────────────────────────────── *)

let production mode e =
  let ctx =
    {
      Sarek_convergence.init_ctx with
      mode =
        (match mode with
        | M.Converged -> Sarek_convergence.Converged
        | M.Diverged -> Sarek_convergence.Diverged);
    }
  in
  Sarek_convergence.check_expr ctx (to_texpr e)

let has_barrier =
  List.exists (function
    | Sarek_error.Barrier_in_diverged_flow _ -> true
    | _ -> false)

let has_warp =
  List.exists (function
    | Sarek_error.Warp_collective_in_diverged_flow _ -> true
    | _ -> false)

let print =
  Print.pair
    (fun _ -> "<expr>")
    (function M.Converged -> "C" | M.Diverged -> "D")

let count = 3000

let tests =
  [
    Test.make
      ~name:"link:barrier_errors_agree"
      ~count
      ~print
      (Gen.pair gen_expr gen_mode)
      (fun (e, m) -> has_barrier (production m e) = (M.check_env m [] e <> []));
    Test.make
      ~name:"link:warp_errors_agree"
      ~count
      ~print
      (Gen.pair gen_expr gen_mode)
      (fun (e, m) ->
        has_warp (production m e) = (M.check_warp_env m [] e <> []));
    Test.make ~name:"link:variance_agrees" ~count gen_expr (fun e ->
        Sarek_convergence.is_thread_varying (to_texpr e)
        = M.is_varying_in_env [] e);
  ]

let () = exit (QCheck_base_runner.run_tests ~verbose:true tests)
