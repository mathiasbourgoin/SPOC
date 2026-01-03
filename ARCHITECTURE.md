# Repository Layout (sdk / runtime / plugins)

This repo is now organized by responsibility. The goal is to split into opam
packages cleanly (sdk for plugin authors, runtime for end users, plugins for
backends).

## sdk/
- `framework/` — `spoc_framework` (interfaces only: `Framework_sig`).
- `ir/` — `sarek_ir` (kernel IR types).
- `ppx/lib/` — `sarek_ppx_lib` and `sarek_ppx` (PPX front‑end, no backend deps).
- `ppx_intrinsic/` — `sarek_ppx_intrinsic` (PPX extension for intrinsics).

## runtime/
- `framework/` — `spoc_framework_registry` (runtime registry + intrinsic registry).
- `core/` — `spoc_core` (Device, Vector, Memory, Transfer, Profiling).
- `sarek/` — `sarek` library (Execute, Kirc_v2, registry glue, IR interpreter).
- `stdlib/` — `sarek_stdlib` (GPU intrinsics, Float32/Int32/Int64/Math, Gpu).
- `float64/` — `sarek_float64` (Float64 helpers).
- `geometry/`, `visibility/` — small helper libs used by tests/samples.
- `tests/` — all unit/e2e/comparison/negative tests for the runtime + PPX.

## plugins/
- `cuda/` — `sarek_cuda` backend (codegen lives here).
- `opencl/` — `sarek_opencl` backend (codegen lives here).
- `native/` — `sarek_native` backend.
- `interpreter/` — `sarek_interpreter` backend.

## Dependencies (conceptual)
- sdk is foundational: no GPU libs, no registry state.
- runtime depends on sdk (`spoc_framework`, `sarek_ir`, ppx at build time) and
  the runtime registry (`spoc_framework_registry`).
- plugins depend on sdk + runtime registry + runtime core as needed.
- tests depend on runtime + plugins + ppx.

## Dune libraries (key names)
- `spoc_framework` (sdk/framework)
- `spoc_framework_registry` (runtime/framework)
- `sarek_ir` (sdk/ir)
- `sarek_ppx_lib`, `sarek_ppx`, `sarek_ppx_intrinsic` (sdk/ppx*)
- `spoc_core` (runtime/core)
- `sarek` (runtime/sarek)
- `sarek_stdlib`, `sarek_float64`, `sarek_geometry`, `sarek_visibility`
- `sarek_cuda`, `sarek_opencl`, `sarek_native`, `sarek_interpreter`
