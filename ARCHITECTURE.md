# SPOC/Sarek Architecture

This repository contains the SPOC GPU computing framework and the Sarek DSL for writing GPU kernels in OCaml.

## Top-Level Structure

```
SPOC/
├── spoc/              # SPOC framework (low-level SDK)
├── sarek/             # Sarek runtime and compiler (high-level)
├── sarek-cuda/        # CUDA backend plugin
├── sarek-opencl/      # OpenCL backend plugin
├── sarek-vulkan/      # Vulkan backend plugin
├── sarek-metal/       # Metal backend plugin (macOS)
├── test_verification/ # End-to-end verification tests
└── scripts/           # Build and coverage scripts
```

## SPOC Framework (`spoc/`)

Low-level GPU computing SDK with backend-agnostic interfaces.

- **`framework/`** - Plugin system and shared error handling
  - `Framework_sig.ml` - Backend interface (`BACKEND` signature)
  - `Backend_error.ml` - Shared parameterized error module
  - `Device_type.ml` - Device abstraction
- **`ir/`** - Intermediate representation
  - `Sarek_ir_types.ml` - Typed IR AST
  - `Sarek_ir_pp.ml` - Pretty printing
  - `Sarek_ir_analysis.ml` - Static analysis
- **`registry/`** - Plugin and intrinsic registry
  - `Framework_registry.ml` - Backend registration
  - `Intrinsic_registry.ml` - Intrinsic function registry

## Sarek Runtime (`sarek/`)

High-level runtime, compiler, and CPU backends.

- **`core/`** - High-level API
  - `Vector.ml` - Type-safe GPU vectors (GADT-based)
  - `Device.ml` - Device discovery
  - `Kernel.ml` - Kernel management
  - `Transfer.ml` - Memory transfers
  - `Memory.ml` - Allocation abstractions
- **`ppx/`** - PPX compiler
  - `Sarek_ppx.ml` - Main entry point
  - `Sarek_parse.ml` - AST parsing
  - `Sarek_typer.ml` - Type checking
  - `Sarek_lower_ir.ml` - IR lowering
  - `Sarek_native_gen.ml` - Native code generation
- **`ppx_intrinsic/`** - PPX for intrinsic definitions
- **`sarek/`** - Execution layer
  - `Execute.ml` - Multi-backend dispatcher
  - `Sarek_ir_interp.ml` - IR interpreter
- **`framework/`** - Framework registry and cache
  - `Framework_registry.ml` - Backend discovery
  - `Framework_cache.ml` - Kernel cache
- **`plugins/`** - CPU backends
  - `native/` - Direct OCaml execution (no GPU)
  - `interpreter/` - IR interpreter (debugging)
- **`Sarek_stdlib/`** - GPU standard library
  - `Float32.ml`, `Int32.ml`, `Int64.ml` - Type operations
  - `Math.ml` - Math functions
  - `Gpu.ml` - GPU utilities
- **`Sarek_float64/`** - Double precision support
- **`Sarek_geometry/`** - Geometry helpers
- **`Visibility_lib/`** - Visibility computations
- **`tests/`** - Test suites
  - `unit/` - Unit tests
  - `e2e/` - End-to-end tests
  - `negative/` - Error handling tests

## GPU Backend Plugins

Each backend is a separate package implementing `Framework_sig.BACKEND`:

- **`sarek-cuda/`** - NVIDIA CUDA backend
- **`sarek-opencl/`** - OpenCL backend (multi-vendor)
- **`sarek-vulkan/`** - Vulkan compute backend
- **`sarek-metal/`** - Apple Metal backend

All GPU backends share a common structure:
- `{Backend}_plugin.ml` - Plugin implementation
- `{Backend}_codegen.ml` - Code generation
- `{Backend}_error.ml` - Structured errors
- `test/` - Backend-specific tests

## Dependencies

```
GPU Backends ───┐
                ├──> Sarek Runtime ───> SPOC Framework
CPU Backends ───┘
```

- **SPOC Framework**: No dependencies on GPU libraries or backends
- **Sarek Runtime**: Depends on SPOC framework, PPX at build time
- **GPU Backends**: Depend on SPOC + Sarek + GPU libraries (dynamically loaded)
- **CPU Backends**: Depend only on SPOC + Sarek

## Package Structure

OPAM packages (planned):
- `spoc` - Core framework
- `sarek` - Runtime and compiler
- `sarek-cuda` - CUDA backend (optional)
- `sarek-opencl` - OpenCL backend (optional)
- `sarek-vulkan` - Vulkan backend (optional)
- `sarek-metal` - Metal backend (optional)
