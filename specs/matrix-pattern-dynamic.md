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
`accept_automatic_float_rewrite` together with the accuracy opt-ins
(`accept_relaxed_accumulation` / `accept_operand_rounding`); a global switch
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
(see Opt-ins and parsing below; default `auto`), changeable with `set_mode`
(`Sarek_matrix.set_mode`):

| mode | int32 | float16 operands | float32 operands |
|---|---|---|---|
| `Off` | original | original | original |
| `Exact` | rewritten (bit-identical) | original | original |
| `Auto` (default) | rewritten | rewritten after `accept_automatic_float_rewrite` + `accept_relaxed_accumulation` | rewritten after `accept_automatic_float_rewrite` + `accept_operand_rounding` (+ relaxed); precision: the requested one, else float16 then bfloat16 if accepted (the bf16x3 split, not faster than the scalar kernel, only on request) |

Devices: by default only HIP (gfx11, 32-wide waves) is rewritten. The
interpreter is rewritten only on an explicit request (`Sarek_matrix.run`,
`run_vectors_outcome ~explicit:true`): by default it stays the independent
oracle that differential tests compare against, and the rewrite is slower
there (64³ int32: 82 → 680 ms sequential, measured by the PR 3 reviewer).
Every other device is declined before any analysis. A stream launch on the
interpreter is declined (its path has no streams). An exception in the
analysis yields `Original`: the analysis runs on the default launch path and
must never fail a launch that works without it. A launch-dimension-dependent
bound, initial value or epilogue is declined (it would change meaning under
the rewritten launch's dimensions). Only `Execute.run_vectors` specialises;
`Execute.run` and raw kernel launches run the kernel as given.

*(agent decision)* The request named `Faithful` (default) and `Optimise`;
the later owner policy makes the float opt-ins the switch for result-changing
rewrites and asks for a global off switch, so the modes are named for what they
allow. `Auto` changes no result unless the program called the opt-ins.

## Opt-ins and parsing (owner decision, 2026-10-11)

Accepting an accuracy deviation and letting `Execute` rewrite a program's OWN
float kernels are two decisions. `accept_relaxed_accumulation` and
`accept_operand_rounding` admit matrix-unit configurations for code that asks
for them (`Sarek_gemm.Coopmat`, the explicit `Sarek_matrix.run`); they do not
enable the automatic rewrite. The automatic float rewrite (`Execute.run_vectors`,
mode `Auto`) needs, in addition,
`Sarek_coopmat.accept_automatic_float_rewrite ~reason` (no env var, like the
accuracy opt-ins; one-way, first reason kept). Without it the outcome is
`Original` and its reason names the missing call. `Sarek_matrix.run` (explicit)
does not need it; int32 needs no opt-in. Granting one opt-in never enables more
than it names: the new opt-in alone does not admit f16 (the accuracy policy
still refuses).

`SAREK_MATRIX_REWRITE`, case-insensitive: `off`/`0`/`false`/`no` give `Off`;
`exact` gives `Exact`; `auto`/`on`/`1`/`true`/`yes` give `Auto`; unset or empty
gives `Auto`. Any other value prints one warning on stderr (variable, value,
accepted values) and gives `Auto` (`Matrix_specialise.parse_mode`, pure).
`SAREK_MATRIX_REPORT` accepts only `1` and is unchanged.

## Size classes

At each launch, from the region (`Sarek_matrix_mapping.region`):

- **Direct** (`Sarek_matrix_direct`): float16 operands (no rounding),
  A, B and C each row- or column-major (each fragment is loaded or stored in
  its operand's own layout), float32 output stored as is, accumulator starting from
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
rewritten IR is physically stable and hits the launch memo. A kernel whose IR
is rebuilt for every launch is analysed on every launch (as it misses the
launch memo too); rewritten kernels are built outside the lock. Mode, opt-ins and
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
the rewrite. `test_matrix_optin.exe` (one process per opt-in combination, HIP):
relaxed only leaves the automatic f16 launch original (reason names
`accept_automatic_float_rewrite`) while `Sarek_matrix.run` still rewrites; the
new opt-in only leaves both original; both rewrite.
`sarek/execute/test/test_matrix_specialise_policy.ml` (no device): the
`SAREK_MATRIX_REWRITE` parser, `lowering_for`, and `choose` on a fake HIP device
(the `~explicit` wiring). PR 2's tests use `run_vectors_plain` for the original.

## Requirements

FR-001 `Matrix_specialise` (mode, choose, report, cache); FR-002
`Execute.run_vectors` / `run_vectors_outcome` / `run_vectors_plain`; FR-003
`Sarek_matrix_direct`; FR-004 advice text; FR-005 tests; FR-006 host
overhead and speed measured.

## Results (tartarus, 2026-10-11)

RX 7900 XTX, ROCm 7.2.4, GPU shared. Logs: `spoc-pipeline-logs/matrix-pattern/pr3-*`.

**Correctness.** `test_matrix_dynamic.exe` (98 checks on 9 devices): the
default `run_vectors` rewrites int32 on gfx1100 only, bit-identical to
`run_vectors_plain`; the interpreters on explicit request only; `Exact` the
same; `Off` never; every other device declines. Review regressions: a
`block_dim_x` reduction bound is declined; the direct variant with one wave
per block (32×96×32) and with three batches on grid y (64×64×48) is within
0.07–0.10 γ_K and writes nothing past C; f32 without `~precision` after
accepting f16 and bf16x3 rounding takes f16; a kernel that is not a matrix
product is not reported. `Exact` leaves float16 alone after the opt-in.
float16 64×64×64 and 256×256×1024 take `Direct` (0.011–0.051 γ_K), 40×24×20 and
100×37×129 `Staged`; accumulating into C takes `Staged` at any size. One
rewritten IR for repeated launches; the reporter sees the rewrite. PR 2's
`test_matrix_rewrite` (260) and `test_matrix_rewrite_float` (60) still pass
with the decision moved into `Execute`; full `dune runtest` and
`make test_negative` pass.

**Host overhead** (`bench_launch_window.exe`, decode-like chain of non-matrix
kernels, 3 runs each, median host µs per enqueued launch, gfx1100): main
4.6–5.2 (window off) / 4.8–6.2 (on); this branch 4.8–5.6 / 5.3–5.9 —
within the run-to-run spread.

**Speed** (`bench_matrix_rewrite.exe 3 10 10`, three sessions, range of session
medians, µs (TFLOP/s)):

| kernel | 256×1024×3072 | 256×3072×3072 | 1000³ |
|---|---|---|---|
| f16 naive (scalar) | 2004–2047 (0.79–0.80) | 5142–5264 (0.92–0.94) | 1516–1546 (1.29–1.32) |
| **f16 rewritten** (direct when aligned) | **83–89 (18.1–19.4)** | **115–122 (39.6–41.9)** | 354–358 (5.6–5.7, staged) |
| `Sarek_gemm.Coopmat` f16, hand-written | 95–99 (16.3–17.0) | 137–139 (34.8–35.2) | — |
| int32 naive → rewritten | 2223–2275 → 1040–1043 | 5293–5543 → 2682–2687 | 1666–1704 → 1117–1118 |
| f32 naive → rewritten f16 / bf16 / bf16x3 | 2199–2224 → 642 / 991 / 2622 | 5235–5408 → 663 / 1020 / 5337–5357 | 1605–1622 → 432 / 672–680 / 2697–2704 |

The direct variant matches or beats the hand-written SCoopmat GEMM. The
bf16x3 split is slower than in PR 2's measurements (NaN-preserving rounding
added in review) and, staged, no faster than the scalar kernel at these sizes:
it buys near-float32 accuracy, not speed.
