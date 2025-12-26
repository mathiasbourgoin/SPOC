# SpocLibs

Libraries built on top of SPOC.

## Libraries

- **Sarek** - Embedded DSL for writing GPGPU kernels in OCaml using PPX syntax
- **Sarek_stdlib** - Standard library of GPU intrinsics (math, barriers, atomics)
- **Sarek_ppx** - PPX preprocessor for Sarek kernel syntax
- **Benchmarks** - Examples and benchmarks using SPOC and Sarek
- **Samples** - Sample programs demonstrating library usage

## Optional Libraries

- **Cublas** - Bindings for NVIDIA cuBLAS library (requires CUDA SDK)
- **Compose** - Basic kernel composition utilities

## Build and Install

1. Ensure SPOC is built and installed
2. Build with dune:
   ```
   dune build
   dune install
   ```

## Usage

See the **Benchmarks** and **Samples** directories for examples.
