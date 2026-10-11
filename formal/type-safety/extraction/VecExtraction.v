(******************************************************************************)
(* VecExtraction.v
 *
 * Rocq -> OCaml extraction configuration for VecSpec.
 * Extracts the memory-access type inference model to VecModel.ml; committed
 * to repo so test/test_type_safety_conformance.ml can use it for T2-VEC tests.
 *
 * This driver is listed in _CoqProject, so `make -f CoqMakefile` (and so
 * scripts/check-formal-proofs.sh, CI) runs it. The gate re-extracts and
 * compares the result with the committed copy (canonicalised, see
 * scripts/canonicalize-extraction.py); a spec edit that is not propagated
 * fails CI. Regenerate after spec changes:
 *   cd formal/type-safety
 *   rocq makefile -f _CoqProject -o CoqMakefile && make -f CoqMakefile
 *   python3 ../../scripts/canonicalize-extraction.py \
 *     extraction/VecModel.ml extraction/VecModel.mli
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.

(* Extract infer_mem_type (the algorithmic type checker for mem_expr) and
 * sarek_type_eq_dec (decidable equality used in EVecSet/EArrSet checks). *)
Set Extraction Output Directory "extraction".

Extraction "VecModel.ml"
  infer_mem_type
  sarek_type_eq_dec.
