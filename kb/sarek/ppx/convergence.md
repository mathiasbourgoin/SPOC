# Sarek Convergence Analysis

## Component Inventory

The check is specified in `specs/convergence-semantics.md`. Code:

- `sarek/ppx/Sarek_variance_ir.ml`: converts the typed AST into a small analysis IR with lexically resolved binders and resolved callees.
- `sarek/ppx/Sarek_variance.ml`: block-scope variance, a whole-kernel fixpoint (`Varying | Deps of params`), spec §3.
- `sarek/ppx/Sarek_convergence.ml`: the check (spec §2, §4, §5), `check_kernel`, `check_expr`, `is_thread_varying`; includes `Sarek_dim_usage`.
- `sarek/ppx/Sarek_dim_usage.ml`: barrier and dimension usage, native execution strategy.
- `sarek/ppx/Sarek_core_primitives.ml`: declared variance and convergence of built-in intrinsics.

## Features And APIs

- Variance: assignments with control dependence relative to the declaration (C-1), if/match values (C-2), helper result summaries per call and context-insensitive parameter variance (C-3), declared intrinsic variance (C-4), private-array reads (C-5), `&&`/`||` right operands (C-24).
- Helpers: bodies checked from Converged with their parameter variance (H-1); a call in diverged flow to a helper containing a synchronisation point is rejected at the call (H-2).
- Production link: `formal/convergence-safety/test/test_convergence_production_link.ml` compares `check_expr` with the extracted Rocq model on sampled terms.

## Invariants

- Thread-varying branch conditions put child control flow in a diverged context.
- Barrier and warp collective intrinsics are illegal in diverged context.
- Shared memory, barriers, block/thread intrinsics, and multidimensional indices force execution strategies that can model them.
- Supersteps carry an implicit barrier (`specs/convergence-semantics.md` §2): a superstep reached in diverged flow is an error with or without `[@divergent]`, which is inert (SS-1, SS-4); the body and the continuation are checked in the outer mode, so structured control flow inside the body reconverges before the implicit barrier (SS-2, SS-3).

## Potential Invariant Violations Or Bugs

- **FIXED (2026-07-02 audit):** indirect thread-varying divergence is now caught. `ctx.varying_vars` (a `StringSet.t`, `sarek/ppx/Sarek_convergence.ml:43-45`) is threaded through `check_expr` and `is_thread_varying_env` records `let`/`let mut` bindings whose value is thread-varying (`sarek/ppx/Sarek_convergence.ml:89, 108-111`); condition checks consult this env (e.g. `:170, 187, 193, 224, 237`). The previously-cited missed case — `let tid = thread_idx_x in if tid > 0 then block_barrier ()` — is now detected. Landed in commits `5e0cd683` ("fix(convergence): correct F-01 + F-02 barrier-safety checker bugs") and `c5c39c71` ("fix(convergence): F-04 — flag barriers after thread-varying early returns in sequences").
- **FIXED (2026-10-11):** the dead `true || ...` in `expr_uses_barriers`'s `TESuperstep` arm is gone. **FIXED (2026-10-11, C-7):** `[@divergent]` used to skip the diverged-entry check, accepting a superstep under a thread-varying `if` (it lowered to `SBarrier` inside `SIf`); the flag is now inert, and the continuation is no longer reset to `Converged`. Production, `ConvergenceSpec.v` and `ConvergenceSemantics.v` agree on the rule (D-57).
- **FIXED (2026-10-11, C-1..C-5, C-24):** variance through assignment, if/match values, helpers, atomics and other declared-varying intrinsics, private arrays, and short-circuit operands (`specs/convergence-semantics.md` §3-§5). Negative tests `sarek/tests/negative/test_conv_*.ml`; accepted kernels pinned in `sarek/tests/unit/test_convergence_accepted.ml`.
- Probable: dimensionality analysis treats `global_size_x/y/z` as simple-compatible in `sarek/ppx/Sarek_dim_usage.ml`, but native simple intrinsic generation rejects many of them in `sarek/ppx/Sarek_native_intrinsics.ml:191-218`.

## Performance Or Maintainability Risks

- The analysis is a fixpoint over a small IR; it is flow-insensitive per variable and assumes race freedom (spec §3).
- Convergence kind depends on primitive names and module-open rewriting; name aliases can be fragile.
- Execution strategy inference must remain synchronized with native generation capabilities.

## Related Tests

- `sarek/tests/unit/dune:22` includes `test_convergence`.
- Negative diverged barrier/warp tests are documented in `sarek/tests/negative/dune:9-11`.
- E2E converged barrier and superstep tests are declared in `sarek/tests/e2e/dune:74-75`.

## Missing Tests

- Aliased or module-qualified barrier/warp intrinsic calls.
- `global_size_*` native simple execution.
- Superstep divergence with nested expressions past the current syntactic cases.

## Concrete Improvement/Fix Candidates

- User intrinsics cannot declare variance or convergence (C-50).
- Contents tracking for private arrays (reads are always may-vary).
- Share an intrinsic capability table between convergence and native generation.
