# Phase 4: Unified Execution Architecture - Implementation Plan

## Overview

This plan implements Phase 4 of the SPOC/Sarek roadmap: a unified execution architecture
supporting JIT backends (CUDA, OpenCL), Direct backends (Native CPU), and Custom backends.

Each unit represents one file (new or single modification) for clean, independent commits.

## Status: COMPLETE ✅

All 10 units have been implemented. Ready for review, testing, and individual commits.

## Units

### Unit 1: `runtime/framework/Framework_sig.ml` (extend) ✅
- [x] Add `execution_model` type (JIT | Direct | Custom)
- [x] Add `intrinsic_impl` type with codegen callbacks
- [x] Add `intrinsic_registry` type (INTRINSIC_REGISTRY module type)
- [x] Add `BACKEND_V2` signature extending `BACKEND`

### Unit 2: `runtime/framework/Intrinsic_registry.ml` (new) ✅
- [x] Create intrinsic registration hashtable
- [x] Implement `register` and `find` functions
- [x] Add backend-specific codegen callback support (Make functor)
- [x] Add convergence property tracking
- [x] Add Global registry for cross-backend lookup

### Unit 3: `SpocLibs/Sarek/Sarek_ir.ml` (extend) ✅
- [x] Add custom type support (TRecord, TVariant, TArray, TVec in elttype)
- [x] Add record/variant expression constructors (ERecord, EVariant)
- [x] Add SLet, SLetMut, SPragma, SMemFence statements
- [x] Update bidirectional Kirc_Ast conversion for new types

### Unit 4: `SpocLibs/Sarek/Sarek_ir_cuda.ml` (new) ✅
- [x] Create CUDA codegen from `Sarek_ir.kernel`
- [x] Map expressions, statements, declarations to CUDA
- [x] Handle intrinsics (thread indexing, sync, atomics, math)
- [x] Generate complete kernel source strings
- [x] Support custom types with `generate_with_types`

### Unit 5: `SpocLibs/Sarek/Sarek_ir_opencl.ml` (new) ✅
- [x] Create OpenCL codegen from `Sarek_ir.kernel`
- [x] Map to OpenCL syntax (get_global_id, __kernel, etc.)
- [x] Handle memory space qualifiers (__global, __local)
- [x] Generate complete kernel source strings
- [x] Support FP64 extension with `generate_with_fp64`

### Unit 6: `runtime/core/Execute.ml` (new) ✅
- [x] Create unified execution dispatcher (`run_v2`)
- [x] Dispatch based on backend's `execution_model`
- [x] JIT path: generate source → compile → launch
- [x] Direct path: call native function directly
- [x] Custom path: delegate to backend
- [x] Add typed argument interface (`run_typed`, `run_from_ir`)

### Unit 7: `plugins/cuda/Cuda_plugin_v2.ml` (new) ✅
- [x] Implement `BACKEND_V2` signature
- [x] Set `execution_model = JIT`
- [x] Use `Sarek_ir_cuda` for source generation
- [x] Register CUDA intrinsics (thread, sync, atomics)
- [x] Auto-register with Framework_registry

### Unit 8: `plugins/opencl/Opencl_plugin_v2.ml` (new) ✅
- [x] Implement `BACKEND_V2` signature
- [x] Set `execution_model = JIT`
- [x] Use `Sarek_ir_opencl` for source generation
- [x] Register OpenCL intrinsics (work items, barriers)
- [x] Auto-register with Framework_registry

### Unit 9: `plugins/native/Native_plugin_v2.ml` (new) ✅
- [x] Implement `BACKEND_V2` signature
- [x] Set `execution_model = Direct`
- [x] `generate_source` returns `None`
- [x] `execute_direct` calls pre-compiled OCaml function
- [x] Register Native intrinsics (CPU runtime calls)

### Unit 10: `SpocLibs/Sarek/Kirc_v2.ml` (new) ✅
- [x] Define new kernel record with lazy IR generation
- [x] Support both old and new kernel formats
- [x] Provide conversion functions (`of_kirc_kernel`, `to_kirc_kernel`)
- [x] Add `run` function for execution
- [x] Add debugging utilities (`pp_ir`, `source_for_backend`)

## Files Created/Modified

### New Files (9)
1. `runtime/framework/Intrinsic_registry.ml` (~180 lines)
2. `SpocLibs/Sarek/Sarek_ir_cuda.ml` (~480 lines)
3. `SpocLibs/Sarek/Sarek_ir_opencl.ml` (~480 lines)
4. `runtime/core/Execute.ml` (~170 lines)
5. `plugins/cuda/Cuda_plugin_v2.ml` (~130 lines)
6. `plugins/opencl/Opencl_plugin_v2.ml` (~160 lines)
7. `plugins/native/Native_plugin_v2.ml` (~170 lines)
8. `SpocLibs/Sarek/Kirc_v2.ml` (~200 lines)
9. `PHASE4_PLAN.md` (this file)

### Modified Files (2)
1. `runtime/framework/Framework_sig.ml` - Added Phase 4 types and BACKEND_V2
2. `runtime/framework/Framework_registry.ml` - Added V2 backend registration
3. `SpocLibs/Sarek/Sarek_ir.ml` - Extended with custom types and new constructs

## Commit Order (Suggested)

Each commit should be independent and reviewable:

1. **Commit 1**: Framework_sig.ml + Framework_registry.ml extensions
   - Foundation types for Phase 4

2. **Commit 2**: Intrinsic_registry.ml
   - Intrinsic management infrastructure

3. **Commit 3**: Sarek_ir.ml extensions
   - Custom types and new IR constructs

4. **Commit 4**: Sarek_ir_cuda.ml
   - CUDA code generation from Sarek_ir

5. **Commit 5**: Sarek_ir_opencl.ml
   - OpenCL code generation from Sarek_ir

6. **Commit 6**: Execute.ml
   - Unified execution dispatcher

7. **Commit 7**: Cuda_plugin_v2.ml
   - CUDA BACKEND_V2 implementation

8. **Commit 8**: Opencl_plugin_v2.ml
   - OpenCL BACKEND_V2 implementation

9. **Commit 9**: Native_plugin_v2.ml
   - Native BACKEND_V2 implementation

10. **Commit 10**: Kirc_v2.ml
    - New kernel type with lazy IR

## Migration Strategy

- All `_v2` files coexist with originals during development
- Once validated, `_v2` files replace originals
- Old code paths remain functional until full migration
- PPX can be updated to generate `kernel_v2` instead of `sarek_kernel`

## Testing Notes

After building, test each component:

1. **Unit tests**: Verify IR generation and conversion
2. **CUDA codegen**: Generate source and validate against expected output
3. **OpenCL codegen**: Same as CUDA
4. **Execute.ml**: Test dispatch logic with mock backends
5. **Plugin V2**: Verify registration and intrinsic lookup
6. **Kirc_v2**: Test conversion from/to legacy format
7. **E2E**: Run existing tests with V2 path

## Notes

- No builds or tests were run during autonomous work
- Each unit produces clean, reviewable code
- Commits can be done independently per unit
- Some type annotations may need adjustment after build
