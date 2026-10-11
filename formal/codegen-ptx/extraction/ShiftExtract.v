(** ShiftExtract.v — extract the right-shift model to OCaml (finding D-54).
 *
 * WHY THIS FILE EXISTS
 *
 * test/test_codegen_ptx_conformance.ml used to carry a hand-written OCaml
 * mirror of AGpuSemantics.v / PtxTypes.v. For [Shr] that mirror said
 * [a lsr b] — the same logical shift the Rocq model had — so the suite agreed
 * with the model and disagreed with the production emitter, which lowers
 * [Shr] to [shr.s32]/[shr.s64]. A mirror cannot find a model error: both are
 * the same mistake written twice.
 *
 * The shift model (theories/PtxShift.v) is now extracted here and LINKED by
 * the conformance test, which checks it against the production emitter
 * ([Sarek_ir_ptx_expr.emit_binop]): the instruction production chose, and the
 * numeric result on negative operands and out-of-range counts.
 *
 * Same technique and rationale as LayoutExtract.v: [Extraction "file.ml"]
 * emits ONE self-contained file with every dependency inlined, which dune can
 * build and a test can link. [Z] and [positive] are extracted as the
 * inductive types they are (no zarith dependency); the test converts to and
 * from [int64] with a few lines of its own.
 *
 * REGENERATING
 *
 *   make -f CoqMakefile
 *   python3 ../../scripts/canonicalize-extraction.py \
 *     extraction/sarek_ptx_shift_model.ml extraction/sarek_ptx_shift_model.mli
 *
 * scripts/check-formal-proofs.sh re-extracts from scratch and compares the
 * canonicalised result against the committed copy (see LayoutExtract.v for why
 * canonicalisation, not a raw byte-compare).
 *)

From CodegenPtx Require Import PtxShift.

From Stdlib Require Import Extraction.
From Stdlib Require Import extraction.ExtrOcamlNatInt.

Set Extraction Output Directory "extraction".

Extraction "sarek_ptx_shift_model.ml"
  to_unsigned to_signed shift_amount
  ashr_w lshr_w
  shr_operand_ok shr_flavour_of shr_width_of shr_insn shr_eval.
