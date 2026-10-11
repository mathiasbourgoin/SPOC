(******************************************************************************)
(* OperatorExtraction.v
 *
 * Rocq -> OCaml extraction for OperatorSpec.
 * Extracts the operator type inference model to OperatorModel.ml;
 * committed to repo so test/test_type_safety_conformance.ml can use it for
 * T3-S2 smoke tests.
 *
 * This driver is listed in _CoqProject, so `make -f CoqMakefile` (and so
 * scripts/check-formal-proofs.sh, CI) runs it. The gate re-extracts and
 * compares the result with the committed copy (canonicalised, see
 * scripts/canonicalize-extraction.py); a spec edit that is not propagated
 * fails CI. Regenerate after spec changes:
 *   cd formal/type-safety
 *   rocq makefile -f _CoqProject -o CoqMakefile && make -f CoqMakefile
 *   python3 ../../scripts/canonicalize-extraction.py \
 *     extraction/OperatorModel.ml extraction/OperatorModel.mli
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.
From TypeSafety Require Import ControlFlowSpec.
From TypeSafety Require Import OperatorSpec.

Set Extraction Output Directory "extraction".

Extraction "OperatorModel.ml"
  infer_op_type.
