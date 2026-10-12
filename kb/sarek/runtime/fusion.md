# Fusion

## Component Inventory

Reviewed fusion files: `sarek/sarek/Sarek_fusion.ml`, `sarek/sarek/FUSION.md`, `sarek/sarek/Fusion_error.ml`, and `sarek/sarek/test/test_fusion_error.ml`.

## Per-File Purpose

- `sarek/sarek/FUSION.md`: design intent, supported fusion patterns, cost model, constraints, and examples.
- `sarek/sarek/Sarek_fusion.ml`: fusion analysis, eligibility checks, map-map fusion, stencil/map fusion, substitution helpers, cost estimation, and auto-fusion.
- `sarek/sarek/Fusion_error.ml`: fusion-specific error variants and formatting.
- `sarek/sarek/test/test_fusion_error.ml`: error formatting tests.

## Launch-time fusion (launch window)

`specs/launch-window.md`. Files:
- `spoc/framework/Launch_hooks.ml` holds the synchronisation-point hook. Its
  callers are Transfer, Device, Profiling and Kernel.
- `sarek/execute/Kernel_access.ml` computes per-parameter read/write sets and
  the horizontal-fusability verdict.
- `sarek/execute/Fusion_horizontal.ml` is the work-group-offset rewrite.
- `sarek/execute/Launch_window.ml` holds the groups, conflicts, flush, cache
  and stats. The window is on by default on HIP, OpenCL, Vulkan, CUDA/PTX,
  CUDA/C and Metal;
  `arg_slot_limit` keeps a Metal fused kernel within its 31 buffer slots.
- `Execute.Window` is the public control.
- Epilogue fusion (`specs/epilogue-fusion.md`) lives in:
  - `sarek/execute/Index_slice.ml`: host evaluation of store indices;
  - `sarek/execute/Fusion_epilogue.ml`: consumer shape and the rewrite;
  - `sarek/execute/Launch_epilogue.ml`: merge conditions and the coverage
    cache;
  - `sarek/execute/Ir_rename.ml`: the renamer both rewrites share.

Invariants:
- Every synchronisation point flushes before it acts.
- Two launches that bind the same vector, with at least one write, are issued
  in submission order.
- A fused kernel runs each member's body unchanged. Only the work-group
  position intrinsics are rewritten.
- Direct and Custom backends are never deferred.
- Epilogue fusion is on by default only on a backend that compiles every kernel without contraction (HIP): elsewhere the fused kernel can differ from the pair by an ulp (`docs/fp-contraction-policy.md`, section 14).
- An epilogue is merged only after the coverage check: every element the
  consumer writes is stored exactly once by the producer, for these
  dimensions and scalars.

## Features/APIs

- Kernel analysis for reads, writes, side effects, barriers, shared memory, and estimated cost.
- `can_fuse` style eligibility decisions.
- Pairwise fusion and pipeline auto-fusion APIs.
- Map-map fusion and stencil fusion rewrite helpers.
- Fusion diagnostics and cost-model metadata.

## Invariants

- Fusion must preserve every original kernel's effects unless it replaces them with a proven equivalent fused kernel.
- Kernels with atomics, unsafe side effects, incompatible dimensions, or unsupported memory dependencies must not be fused.
- Substitution over IR expressions must recursively cover every expression form that can reference an intermediate value.
- Fusion diagnostics should match the documented constraints in `FUSION.md`.

## Potential Invariant Violations/Bugs

- Stencil substitution is incomplete. `subst_stencil_reads` recurses through only a subset of expression variants and returns unchanged for the rest at `sarek/sarek/Sarek_fusion.ml:873-908`. Expressions in records, app args, array-read expressions, if/match branches, or other missed forms can retain references to intermediates. Compare with fuller substitution helpers elsewhere in the file. Uncertain exact surface depends on which IR forms stencil fusion admits.

## Recently Resolved

- Pipeline APIs dropping unfused kernels was fixed by PR #138, merged as `06b7d70`. New list-returning APIs preserve unfused kernels, while legacy single-kernel APIs remain strict when a pipeline cannot collapse to one kernel.
- Mismatched one-to-one index fusion was rejected in PR #138, preventing a producer from being dropped when substitution cannot actually replace the consumer read.
- **Fusion ignores atomics — fixed 2026-07-02 (merged, commit b4c78853, hardened by a follow-up atomics-lvalue/SNative commit).** `kernel_uses_atomics` is now a real, single-source-of-truth analysis in `spoc/ir/Sarek_ir_analysis.ml:209-213`: it walks `kern_params`, `kern_locals`, `kern_body`, and `kern_funcs` (helper bodies, so atomics hidden inside a called helper are not missed), covers `SAssign` lvalues (not just the right-hand expression), and treats `SNative` conservatively as atomic-bearing. `Sarek_fusion.ml:352` now sets `has_atomics = Sarek_ir_analysis.kernel_uses_atomics k` (previously hardcoded `false` with a TODO). All four fusion gates now reject atomics: `can_fuse` (`Sarek_fusion.ml:434`), `can_fuse_reduction` (`:701`), `can_fuse_stencil` (`:863`), and `should_fuse` (`:1125`) each check `prod_info.has_atomics || cons_info.has_atomics`/`map_info.has_atomics || reduce_info.has_atomics` and refuse fusion. This closes the previously-flagged "fusion ignores atomics (has_atomics=false TODO)" bug; the invariant "kernels with atomics ... must not be fused" (see Invariants above) is now enforced in code, matching `FUSION.md:116-122`.

## Performance Or Maintainability Risks

- Legacy single-kernel pipeline APIs remain narrow: callers that need to preserve partially fused schedules should use the list-returning APIs added in PR #138.
- Analysis fields can drift from implementation because some fields are populated as constants.
- Rewrite helpers are hand-recursive and differ in expression coverage.
- Fusion diagnostics can become misleading when documented constraints are not enforced.

## Related Tests

- `sarek/sarek/test/test_fusion_error.ml`: error formatting only.
- `sarek/tests/unit/test_fusion.ml`: fusion analysis, pipeline preservation, auto-fusion decisions, mismatched-index rejection, and (as of the atomics fix) `test_can_fuse_with_direct_atomic` / `test_can_fuse_with_atomic_in_helper` covering direct and helper-reachable atomics (`sarek/tests/unit/test_fusion.ml:173-245`).

## Missing Tests

- Stencil fusion with intermediate reads embedded in each supported expression form.
- Fused versus unfused equivalence tests for map-map and stencil cases.
- Diagnostics tests that ensure documented constraints are emitted.

## Concrete Improvement/Fix Candidates

- Replace partial substitution walkers with a shared full-expression traversal.
- Add semantic regression tests that execute original and fused kernels on small arrays and compare outputs.
