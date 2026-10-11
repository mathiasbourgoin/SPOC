(******************************************************************************)
(* RegistryExtraction.v
 *
 * Rocq -> OCaml extraction configuration for RegistrySpec.
 * Extracts the record field-access type inference model to RegistryModel.ml;
 * committed to repo so test/test_type_safety_conformance.ml can use it for
 * T2-REGISTRY tests.
 *
 * This driver is listed in _CoqProject, so `make -f CoqMakefile` (and so
 * scripts/check-formal-proofs.sh, CI) runs it. The gate re-extracts and
 * compares the result with the committed copy (canonicalised, see
 * scripts/canonicalize-extraction.py); a spec edit that is not propagated
 * fails CI. Regenerate after spec changes:
 *   cd formal/type-safety
 *   rocq makefile -f _CoqProject -o CoqMakefile && make -f CoqMakefile
 *   python3 ../../scripts/canonicalize-extraction.py \
 *     extraction/RegistryModel.ml extraction/RegistryModel.mli
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.

(* Extract infer_rec_type (the algorithmic type checker for rec_expr),
 * field_lookup (field name lookup in a record type's field list), and
 * sarek_type_eq_dec (decidable equality, also extracted from VecSpec). *)
Set Extraction Output Directory "extraction".

Extraction "RegistryModel.ml"
  infer_rec_type
  field_lookup
  sarek_type_eq_dec.
