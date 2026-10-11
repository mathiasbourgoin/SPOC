(******************************************************************************)
(* UnifyExtraction.v
 *
 * Rocq -> OCaml extraction configuration for UnifySpec.
 * Extracts the pure unifier model to UnifyModel.ml; committed to repo so
 * test/test_type_safety_conformance.ml can use it as the oracle for T2-UNIFY.
 *
 * This driver is listed in _CoqProject, so `make -f CoqMakefile` (and so
 * scripts/check-formal-proofs.sh, CI) runs it. The gate re-extracts and
 * compares the result with the committed copy (canonicalised, see
 * scripts/canonicalize-extraction.py); a spec edit that is not propagated
 * fails CI. Regenerate after spec changes:
 *   cd formal/type-safety
 *   rocq makefile -f _CoqProject -o CoqMakefile && make -f CoqMakefile
 *   python3 ../../scripts/canonicalize-extraction.py \
 *     extraction/UnifyModel.ml extraction/UnifyModel.mli
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import UnifySpec.

(* Extract the core unification functions.
 * unify_fun: pure unifier (returns option pre_subst).
 * apply_subst: apply substitution to a pre_type.
 * follow_pvar: follow PVar chains through a substitution. *)

Set Extraction Output Directory "extraction".

Extraction "UnifyModel.ml"
  unify_fun
  apply_subst
  follow_pvar.
