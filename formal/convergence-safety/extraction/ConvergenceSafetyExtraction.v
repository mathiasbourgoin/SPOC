(******************************************************************************)
(* ConvergenceSafetyExtraction.v
 *
 * Rocq → OCaml extraction configuration for ConvergenceSpec.
 * Extracts the abstract model to ConvergenceModel.ml; committed to repo
 * so test/test_convergence_extraction.ml can use it as the oracle.
 *
 * Run to refresh after spec changes:
 *   cd formal/convergence-safety
 *   coqc -R theories ConvergenceSpec \
 *         -output-directory extraction \
 *         extraction/ConvergenceSafetyExtraction.v
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From ConvergenceSpec Require Import ConvergenceSpec.
From ConvergenceSpec Require Import ConvergenceSemantics.

(* Let exec_mode and error extract as regular OCaml variants —
 * do NOT use Extract Inductive here so the type definitions are emitted.
 *
 * T3-S8: eval_concrete (the operational evaluator with vary_val := fun t => t)
 * is extracted alongside the static checkers so the differential conformance
 * suite can exercise the operational semantics (outcome / trace / event) — not
 * only the static analysis — against an inline OCaml reference.
 *
 * The env-threaded checkers (is_varying_in_env, check_env, check_warp_env) are
 * extracted too: they are the model of the production checker that
 * specs/convergence-semantics.md settles, and the production-link test calls
 * them. *)

(* Emit directly into extraction/ (same pattern as type-safety's
 * SpecialExtraction.v) so no stray copy appears at the project root. *)
Set Extraction Output Directory "extraction".

Extraction "ConvergenceModel.ml"
  is_varying
  barrier_free
  has_diverging_cf
  check
  check_warp
  is_varying_in_env
  check_env
  check_warp_env
  merge_dim_usage
  empty_dim_usage
  eval_concrete.
