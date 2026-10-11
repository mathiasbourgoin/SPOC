---
name: roster-spec
type: spec
status: live
feature: Launch-time specialisation of matrix products inside Execute, with a mode switch, size classes, reporting and caching
brief: owner request 2026-10-11 (matrix pattern detection — PR 3 of 3: dynamic mode); owner policy relayed by the coordinator 2026-10-11
date: 2026-10-11
version: 1.0.0
---

# Spec — matrix-pattern-dynamic

Builds on specs/matrix-pattern-advice.md (recogniser) and
specs/matrix-pattern-rewrite.md (rewrite). Owner policy (2026-10-11, relayed):
safe optimisations on by default, risky ones opt-in — the exact int32 rewrite
applies by default inside `Execute`; float rewrites only under the explicit
`accept_relaxed_accumulation` / `accept_operand_rounding`; a global switch
disables the rewrite entirely. Coordinator: plug in as a per-launch
specialisation hook on the launch window, not a parallel launch path.

## Where

`Execute.run_vectors` (and the new `Execute.run_vectors_outcome`, which also
returns the decision) asks `Matrix_specialise.choose` (library
`sarek.execute`) before launching. A `Specialised` answer launches the
rewritten kernel through the same path (`run_vectors_plain` → `run` → the
launch window, so the rewritten launch is deferred and fusable like any other)
with the region parameters appended; on the interpreter device it goes through
`run_interpreter_vectors` (the interpreter device's own `run` probes no
cooperative-matrix configuration). `Original` launches the kernel as given.
`Execute.run_vectors_plain` is the old behaviour, for tests and benchmarks
that need the original. `Sarek_matrix.run` keeps its PR 2 signature on top.

## Mode

`Matrix_specialise.mode = Off | Exact | Auto`, from `SAREK_MATRIX_REWRITE`
(`off`, `exact`, default `auto`), changeable with `set_mode`
(`Sarek_matrix.set_mode`):

| mode | int32 | float16 operands | float32 operands |
|---|---|---|---|
| `Off` | original | original | original |
| `Exact` | rewritten (bit-identical) | original | original |
| `Auto` (default) | rewritten | rewritten after `accept_relaxed_accumulation` | rewritten after `accept_operand_rounding` (+ relaxed); precision: the requested one, else the most accurate accepted (bf16x3, f16, bf16) |

*(agent decision)* The request named `Faithful` (default) and `Optimise`;
the later owner policy makes the float opt-ins the switch for result-changing
rewrites and asks for a global off switch, so the modes are named for what they
allow. `Auto` changes no result unless the program called an opt-in.

## Size classes

At each launch, from the region (`Sarek_matrix_mapping.region`):

- **Direct** (`Sarek_matrix_direct`): float16 operands (no rounding),
  row-major A, B and C, float32 output stored as is, accumulator starting from
  zero (from C it would update C in place through fragments; the interpreter
  evaluates a fragment operation once per invocation, so a later invocation
  would see an earlier store — measured: 4.9e6 γ_K on the interpreter, correct
  on gfx1100; such kernels are staged), region a whole number of 32×32 tiles, K a multiple of 16, no
  quotient predicate. One wave per 32×32 tile (2×2 fragments), 2 waves per
  block when the tile count is even, else 1; fragments loaded straight from
  global memory; batches on grid y.
- **Staged** (`Sarek_matrix_rewrite`): everything else the rewrite accepts.

The rewritten grid of either must fit `max_grid_dims`.

## Cache and report

Per IR (physical identity, ephemeron, as #424's launch memo): the static plan
and the rewritten kernels per (lowering, size class) — so a kernel's
rewritten IR is physically stable and hits the launch memo. Mode, opt-ins and
device are checked per launch. A kernel that is not a matrix product costs one
memo lookup. `Cache_hooks.on_clear_all` clears it.
`Matrix_specialise.set_reporter` receives every decision about a matrix
product; `SAREK_MATRIX_REPORT=1` prints each distinct (kernel, device, outcome)
once on stderr.

The PPX advice (PR 1) now says what Execute does with the nest.

## Tests

`sarek/tests/e2e/test_matrix_dynamic.ml`, every device: `Auto` and `Exact`
rewrite int32 on the interpreter and gfx11 (bit-identical to the plain
launch), `Off` never; `Exact` refuses float16 even after the opt-in; float16
64×64×64 and 256×256×1024 take `Direct`, 40×24×20 and 100×37×129 take
`Staged`, all within γ_K; one rewritten IR for two launches; the reporter sees
the rewrite. PR 2's tests use `run_vectors_plain` for the original.

## Requirements

FR-001 `Matrix_specialise` (mode, choose, report, cache); FR-002
`Execute.run_vectors` / `run_vectors_outcome` / `run_vectors_plain`; FR-003
`Sarek_matrix_direct`; FR-004 advice text; FR-005 tests; FR-006 host
overhead and speed measured.
