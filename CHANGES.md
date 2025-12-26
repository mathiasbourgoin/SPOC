## 2025-12

### Added

- Full PPX rewrite replacing Camlp4
  - PPX-based kernel syntax ([%kernel ...], [%ktype ...])
  - Type registry for cross-module GPU types
  - Intrinsic registration system (Sarek_stdlib)
  - Float32 stdlib module with math intrinsics
  - Pragma support for loop unrolling hints
- Kernel fusion system (Sarek_fusion.ml)
  - Vertical fusion for producer-consumer patterns
  - Reduction fusion (map + reduce)
  - Stencil fusion with radius tracking
  - Auto-fusion heuristics (should_fuse, auto_fuse_pipeline)
- Clean intermediate representation (Sarek_ir.ml)
- BSP superstep syntax with barrier synchronization
- Warp convergence tracking (Sarek_convergence.ml)
- Core primitives with convergence info (Sarek_core_primitives.ml)
- Comprehensive test suite (unit, negative, e2e tests)

### Changed

- Removed Camlp4 dependency entirely
- Improved type inference for kernel parameters
- Better compile-time error messages for GPU-specific issues

## 20210823

- Add PPX extension to declare external GPGPU kernels
- Update Samples to use PPX instead of Camlp4 extension
- Update dune/opam files for opam release

## 20210816

### Added

- Build with dune
- Compatible with OCaml 4.12
- Switch to github actions for CI
