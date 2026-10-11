(******************************************************************************)
(* TypeSafetyExtraction.v
 *
 * Rocq → OCaml extraction configuration for TypeSafetySpec.
 * Extracts the abstract type-checker model to TypeSafetyModel.ml; committed
 * to repo so test/test_type_safety_conformance.ml can use it as the oracle.
 *
 * This driver is listed in _CoqProject, so `make -f CoqMakefile` (and so
 * scripts/check-formal-proofs.sh, CI) runs it. The gate re-extracts and
 * compares the result with the committed copy (canonicalised, see
 * scripts/canonicalize-extraction.py); a spec edit that is not propagated
 * fails CI. Regenerate after spec changes:
 *   cd formal/type-safety
 *   rocq makefile -f _CoqProject -o CoqMakefile && make -f CoqMakefile
 *   python3 ../../scripts/canonicalize-extraction.py \
 *     extraction/TypeSafetyModel.ml extraction/TypeSafetyModel.mli
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.

(* Let prim_type / reg_type / mem_space / sarek_type / lit / expr / type_error
 * extract as regular OCaml variants — do NOT use Extract Inductive here so
 * the type definitions are emitted in the generated .ml.
 *
 * infer_type and lookup_env are the two algorithmic functions under test;
 * they are the oracle against which Sarek_typer.infer is compared in the
 * differential conformance harness. *)

Set Extraction Output Directory "extraction".

Extraction "TypeSafetyModel.ml"
  infer_type
  lookup_env.
