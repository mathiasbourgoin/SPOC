(******************************************************************************)
(* test_convergence_extraction.ml
 *
 * Extraction conformance suite.
 * Verifies that the OCaml extracted from ConvergenceSpec.v (ConvergenceModel)
 * agrees with the hand-written inline abstract model on the extracted
 * semantic functions, including the env-threaded checkers check_env and
 * check_warp_env (superstep settlement, D-56).  This is CMBT links 2 (extraction) + 4 (test) combined:
 *
 *   Rocq spec → coqc extraction → ConvergenceModel.ml → this test
 *
 * The two models use structurally identical but distinct OCaml types.
 * to_extracted translates inline→extracted; normalize_errors does the reverse
 * for comparison.
 ******************************************************************************)

open QCheck2

(* ===== Inline abstract model (mirrors ConvergenceSpec.v) ===== *)

type expr =
  | ELit
  | EVary
  | EBarrier
  | EWarpPoint
  | EVar of int
  | EBinop of expr * expr
  | EUnop of expr
  | EIf of expr * expr * expr
  | EWhile of expr * expr
  | EFor of expr * expr * expr
  | ESeq of expr list
  | ELet of int * expr * expr
  | ESuperstep of bool * expr * expr
  | EApp of expr list
  | EReturn of expr

type exec_mode = Converged | Diverged

type error = BarrierError | WarpError

let rec is_varying_inline = function
  | EVary -> true
  | ELit | EBarrier | EWarpPoint | EVar _ -> false
  | EBinop (a, b) -> is_varying_inline a || is_varying_inline b
  | EUnop e -> is_varying_inline e
  | EIf (c, t, el) ->
      is_varying_inline c || is_varying_inline t || is_varying_inline el
  | EWhile (c, b) -> is_varying_inline c || is_varying_inline b
  | EFor (lo, hi, b) ->
      is_varying_inline lo || is_varying_inline hi || is_varying_inline b
  | ESeq es -> List.exists is_varying_inline es
  | ELet (_, v, b) -> is_varying_inline v || is_varying_inline b
  | ESuperstep (_, body, cont) ->
      is_varying_inline body || is_varying_inline cont
  | EApp args -> List.exists is_varying_inline args
  | EReturn e -> is_varying_inline e

let rec barrier_free_inline = function
  | EBarrier -> false
  | ELit | EVary | EWarpPoint | EVar _ -> true
  | EBinop (a, b) -> barrier_free_inline a && barrier_free_inline b
  | EUnop e -> barrier_free_inline e
  | EIf (c, t, el) ->
      barrier_free_inline c && barrier_free_inline t && barrier_free_inline el
  | EWhile (c, b) -> barrier_free_inline c && barrier_free_inline b
  | EFor (lo, hi, b) ->
      barrier_free_inline lo && barrier_free_inline hi && barrier_free_inline b
  | ESeq es -> List.for_all barrier_free_inline es
  | ELet (_, v, b) -> barrier_free_inline v && barrier_free_inline b
  | ESuperstep _ -> false (* SS-4: never barrier-free *)
  | EApp args -> List.for_all barrier_free_inline args
  | EReturn e -> barrier_free_inline e

let rec has_diverging_cf_inline = function
  | EIf (c, t, el) ->
      is_varying_inline c || has_diverging_cf_inline t
      || has_diverging_cf_inline el
  | EWhile (c, b) -> is_varying_inline c || has_diverging_cf_inline b
  | EFor (lo, hi, b) ->
      is_varying_inline lo || is_varying_inline hi || has_diverging_cf_inline b
  | EBinop (a, b) -> has_diverging_cf_inline a || has_diverging_cf_inline b
  | EUnop e -> has_diverging_cf_inline e
  | ESeq es -> List.exists has_diverging_cf_inline es
  | ELet (_, v, b) -> has_diverging_cf_inline v || has_diverging_cf_inline b
  | ESuperstep (_, body, cont) ->
      has_diverging_cf_inline body || has_diverging_cf_inline cont
  | EApp args -> List.exists has_diverging_cf_inline args
  | EReturn e -> has_diverging_cf_inline e
  | _ -> false

let rec check_inline m e =
  match e with
  | EBarrier -> ( match m with Diverged -> [BarrierError] | Converged -> [])
  | ELit | EVary | EWarpPoint | EVar _ -> []
  | EBinop (a, b) -> check_inline m a @ check_inline m b
  | EUnop e -> check_inline m e
  | EIf (c, t, el) ->
      let inner = if is_varying_inline c then Diverged else m in
      check_inline m c @ check_inline inner t @ check_inline inner el
  | EWhile (c, b) ->
      let inner = if is_varying_inline c then Diverged else m in
      check_inline m c @ check_inline inner b
  | EFor (lo, hi, b) ->
      let inner =
        if is_varying_inline lo || is_varying_inline hi then Diverged else m
      in
      check_inline m lo @ check_inline m hi @ check_inline inner b
  | ESeq es -> List.concat_map (check_inline m) es
  | ELet (_, v, b) -> check_inline m v @ check_inline m b
  | ESuperstep (_divergent, body, cont) ->
      (* SS-1/SS-4: entry error in Diverged mode whatever the flag *)
      let entry_errors =
        match m with Diverged -> [BarrierError] | Converged -> []
      in
      entry_errors @ check_inline m body @ check_inline m cont
  | EApp args -> List.concat_map (check_inline m) args
  | EReturn e -> check_inline m e

let rec check_warp_inline m e =
  match e with
  | EWarpPoint -> ( match m with Diverged -> [WarpError] | Converged -> [])
  | EBarrier -> ( match m with Diverged -> [BarrierError] | Converged -> [])
  | ELit | EVary | EVar _ -> []
  | EBinop (a, b) -> check_warp_inline m a @ check_warp_inline m b
  | EUnop e -> check_warp_inline m e
  | EIf (c, t, el) ->
      let inner = if is_varying_inline c then Diverged else m in
      check_warp_inline m c @ check_warp_inline inner t
      @ check_warp_inline inner el
  | EWhile (c, b) ->
      let inner = if is_varying_inline c then Diverged else m in
      check_warp_inline m c @ check_warp_inline inner b
  | EFor (lo, hi, b) ->
      let inner =
        if is_varying_inline lo || is_varying_inline hi then Diverged else m
      in
      check_warp_inline m lo @ check_warp_inline m hi
      @ check_warp_inline inner b
  | ESeq es -> List.concat_map (check_warp_inline m) es
  | ELet (_, v, b) -> check_warp_inline m v @ check_warp_inline m b
  | ESuperstep (_divergent, body, cont) ->
      (* SS-1/SS-4: entry error in Diverged mode whatever the flag *)
      let entry_errors =
        match m with Diverged -> [BarrierError] | Converged -> []
      in
      entry_errors @ check_warp_inline m body @ check_warp_inline m cont
  | EApp args -> List.concat_map (check_warp_inline m) args
  | EReturn e -> check_warp_inline m e

(* ===== Env-threaded checkers (mirror ConvergenceSpec.check_env and
   ConvergenceSemantics.check_warp_env after the superstep settlement,
   specs/convergence-semantics.md section 2, and D-56) ===== *)

let env_lookup env x =
  match List.find_opt (fun (y, _) -> y = x) env with
  | Some (_, v) -> v
  | None -> false

let rec is_varying_env env = function
  | EVary -> true
  | ELit | EBarrier | EWarpPoint -> false
  | EVar x -> env_lookup env x
  | EBinop (a, b) -> is_varying_env env a || is_varying_env env b
  | EUnop e -> is_varying_env env e
  | EIf (c, t, el) ->
      is_varying_env env c || is_varying_env env t || is_varying_env env el
  | EWhile (c, b) -> is_varying_env env c || is_varying_env env b
  | EFor (lo, hi, b) ->
      is_varying_env env lo || is_varying_env env hi || is_varying_env env b
  | ESeq es -> List.exists (is_varying_env env) es
  | ELet (x, v, b) -> is_varying_env ((x, is_varying_env env v) :: env) b
  | ESuperstep (_, body, cont) ->
      is_varying_env env body || is_varying_env env cont
  | EApp args -> List.exists (is_varying_env env) args
  | EReturn e -> is_varying_env env e

let rec has_reachable_return = function
  | EReturn _ -> true
  | EIf (_, t, el) -> has_reachable_return t || has_reachable_return el
  | ESeq es -> List.exists has_reachable_return es
  | ELet (_, v, b) -> has_reachable_return v || has_reachable_return b
  | EWhile (_, b) | EFor (_, _, b) -> has_reachable_return b
  | ESuperstep (_, body, cont) ->
      has_reachable_return body || has_reachable_return cont
  | _ -> false

let rec has_varying_return env = function
  | EIf (c, t, el) ->
      if is_varying_env env c then
        has_reachable_return t || has_reachable_return el
      else has_varying_return env t || has_varying_return env el
  | ESeq es -> List.exists (has_varying_return env) es
  | ELet (x, v, b) ->
      let env' = if is_varying_env env v then (x, true) :: env else env in
      has_varying_return env v || has_varying_return env' b
  | EWhile (c, b) -> has_varying_return env c || has_varying_return env b
  | EFor (lo, hi, b) ->
      has_varying_return env lo || has_varying_return env hi
      || has_varying_return env b
  | _ -> false

(* One walker for both checkers, parameterised by which point is flagged:
   check_env flags EBarrier, check_warp_env flags EWarpPoint; only check_env
   flags a superstep entry (it is a barrier error). Both thread the F-04
   varying-return flag through sequences. *)
let rec check_env_gen ~warp m env e =
  let go = check_env_gen ~warp in
  match e with
  | EBarrier when not warp -> (
      match m with Diverged -> [BarrierError] | Converged -> [])
  | EWarpPoint when warp -> (
      match m with Diverged -> [WarpError] | Converged -> [])
  | ELit | EVary | EBarrier | EWarpPoint | EVar _ -> []
  | EBinop (a, b) -> go m env a @ go m env b
  | EUnop e -> go m env e
  | EIf (c, t, el) ->
      let inner = if is_varying_env env c then Diverged else m in
      go m env c @ go inner env t @ go inner env el
  | EWhile (c, b) ->
      let inner = if is_varying_env env c then Diverged else m in
      go m env c @ go inner env b
  | EFor (lo, hi, b) ->
      let inner =
        if is_varying_env env lo || is_varying_env env hi then Diverged else m
      in
      go m env lo @ go m env hi @ go inner env b
  | ESeq es ->
      let rec seq sm = function
        | [] -> []
        | x :: rest ->
            let sm' = if has_varying_return env x then Diverged else sm in
            go sm env x @ seq sm' rest
      in
      seq m es
  | ELet (x, v, b) -> go m env v @ go m ((x, is_varying_env env v) :: env) b
  | ESuperstep (_, body, cont) ->
      let entry =
        match (warp, m) with false, Diverged -> [BarrierError] | _ -> []
      in
      entry @ go m env body @ go m env cont
  | EApp args -> List.concat_map (go m env) args
  | EReturn e -> go m env e

(* ===== Translation: inline types → extracted types ===== *)

module M = Convergence_model.ConvergenceModel

let rec to_extracted : expr -> M.expr = function
  | ELit -> ELit
  | EVary -> EVary
  | EBarrier -> EBarrier
  | EWarpPoint -> EWarpPoint
  | EVar x -> EVar x
  | EBinop (a, b) -> EBinop (to_extracted a, to_extracted b)
  | EUnop e -> EUnop (to_extracted e)
  | EIf (c, t, el) -> EIf (to_extracted c, to_extracted t, to_extracted el)
  | EWhile (c, b) -> EWhile (to_extracted c, to_extracted b)
  | EFor (lo, hi, b) -> EFor (to_extracted lo, to_extracted hi, to_extracted b)
  | ESeq es -> ESeq (List.map to_extracted es)
  | ELet (x, v, b) -> ELet (x, to_extracted v, to_extracted b)
  | ESuperstep (dv, body, cont) ->
      ESuperstep (dv, to_extracted body, to_extracted cont)
  | EApp args -> EApp (List.map to_extracted args)
  | EReturn e -> EReturn (to_extracted e)

let to_extracted_mode : exec_mode -> M.exec_mode = function
  | Converged -> Converged
  | Diverged -> Diverged

let normalize_errors (errs : M.error list) : error list =
  List.map
    (function M.BarrierError -> BarrierError | M.WarpError -> WarpError)
    errs

(* ===== Generator (same pattern as test_convergence_conformance) ===== *)

let gen_mode : exec_mode Gen.t = Gen.oneof_list [Converged; Diverged]

let gen_var_id : int Gen.t = Gen.int_range 0 3

let gen_expr : expr Gen.t =
  Gen.sized_size (Gen.int_range 0 6)
  @@ Gen.fix (fun self n ->
      if n = 0 then Gen.oneof_list [ELit; EVary; EBarrier; EWarpPoint]
      else
        let sub = self (n / 2) in
        let sub2 = Gen.pair sub sub in
        let sub3 = Gen.triple sub sub sub in
        let sublist = Gen.list_size (Gen.int_range 0 3) sub in
        Gen.oneof
          [
            Gen.return ELit;
            Gen.return EVary;
            Gen.return EBarrier;
            Gen.return EWarpPoint;
            Gen.map (fun x -> EVar x) gen_var_id;
            Gen.map (fun (a, b) -> EBinop (a, b)) sub2;
            Gen.map (fun e -> EUnop e) sub;
            Gen.map (fun (c, t, el) -> EIf (c, t, el)) sub3;
            Gen.map (fun (c, b) -> EWhile (c, b)) sub2;
            Gen.map (fun (lo, hi, b) -> EFor (lo, hi, b)) sub3;
            Gen.map (fun es -> ESeq es) sublist;
            Gen.map
              (fun (x, (v, b)) -> ELet (x, v, b))
              (Gen.pair gen_var_id sub2);
            Gen.map
              (fun (dv, (body, cont)) -> ESuperstep (dv, body, cont))
              (Gen.pair Gen.bool sub2);
            Gen.map (fun args -> EApp args) sublist;
            Gen.map (fun e -> EReturn e) sub;
          ])

(* ===== Properties: inline vs extracted must agree ===== *)

let () =
  let tests =
    [
      Test.make ~name:"extr:is_varying_agrees" ~count:2000 gen_expr (fun e ->
          let expected = is_varying_inline e in
          let got = M.is_varying (to_extracted e) in
          expected = got);
      Test.make ~name:"extr:barrier_free_agrees" ~count:2000 gen_expr (fun e ->
          let expected = barrier_free_inline e in
          let got = M.barrier_free (to_extracted e) in
          expected = got);
      Test.make
        ~name:"extr:has_diverging_cf_agrees"
        ~count:2000
        gen_expr
        (fun e ->
          let expected = has_diverging_cf_inline e in
          let got = M.has_diverging_cf (to_extracted e) in
          expected = got);
      Test.make
        ~name:"extr:check_converged_agrees"
        ~count:1500
        gen_expr
        (fun e ->
          let expected = check_inline Converged e in
          let got = normalize_errors (M.check Converged (to_extracted e)) in
          expected = got);
      Test.make
        ~name:"extr:check_diverged_agrees"
        ~count:1500
        gen_expr
        (fun e ->
          let expected = check_inline Diverged e in
          let got = normalize_errors (M.check Diverged (to_extracted e)) in
          expected = got);
      Test.make
        ~name:"extr:check_any_mode_agrees"
        ~count:1500
        (Gen.pair gen_expr gen_mode)
        (fun (e, m) ->
          let expected = check_inline m e in
          let got =
            normalize_errors (M.check (to_extracted_mode m) (to_extracted e))
          in
          expected = got);
      (* CMBT link for check_warp: inline model must agree with extracted M.check_warp
       * over random (mode, expr) pairs. *)
      Test.make
        ~name:"extr:check_warp_agrees"
        ~count:1500
        (Gen.pair gen_expr gen_mode)
        (fun (e, m) ->
          let expected = check_warp_inline m e in
          let got =
            normalize_errors
              (M.check_warp (to_extracted_mode m) (to_extracted e))
          in
          expected = got);
      Test.make
        ~name:"extr:is_varying_in_env_agrees"
        ~count:2000
        gen_expr
        (fun e -> is_varying_env [] e = M.is_varying_in_env [] (to_extracted e));
      Test.make
        ~name:"extr:check_env_agrees"
        ~count:1500
        (Gen.pair gen_expr gen_mode)
        (fun (e, m) ->
          let expected = check_env_gen ~warp:false m [] e in
          let got =
            normalize_errors
              (M.check_env (to_extracted_mode m) [] (to_extracted e))
          in
          expected = got);
      Test.make
        ~name:"extr:check_warp_env_agrees"
        ~count:1500
        (Gen.pair gen_expr gen_mode)
        (fun (e, m) ->
          let expected = check_env_gen ~warp:true m [] e in
          let got =
            normalize_errors
              (M.check_warp_env (to_extracted_mode m) [] (to_extracted e))
          in
          expected = got);
    ]
  in
  exit (QCheck_base_runner.run_tests ~verbose:true tests)
