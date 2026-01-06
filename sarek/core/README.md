(******************************************************************************
 * sarek/core - Runtime Core Modules
 *
 * This README documents the core runtime pieces that sit between the Sarek PPX
 * (IR generation) and the backend plugins (CUDA/OpenCL/Vulkan/Native/Interpreter).
 * The goal is type safety (no Obj.t), clean abstraction boundaries, and clear
 * logging/debug hooks.
 ******************************************************************************)

## Module Map

- `Log` / `Error`: Unified logging with `SAREK_DEBUG` env flag; `Error` delegates
  to `Log` for levels/components.
- `Device`: Unified device view from registered backends; predicates and
  capability queries are based on `capabilities.is_cpu`, etc.
- `Memory`: Typed device buffers via `Memory.BUFFER` (no Obj.t), alloc/transfer
  helpers, typed `bind_to_kargs`.
- `Kernel` / `Kernel_arg`: Typed kernel compilation/launch; extensible `kargs`
  wrapping; `Kernel_arg` GADT for vectors/scalars without Obj.t.
- `Vector` (with `Vector_types`, `Vector_storage`, `Vector_transfer`): High-level
  vectors with location tracking, storage helpers, host pointers, sync callbacks,
  copy/subvector/partition host logic.
- `Transfer`: Device/host transfers and buffer allocation/ensure-buffer logic,
  using the unified buffer interface.
- `Runtime`: High-level run helpers (re-exporting `Framework_sig.dims`), kernel
  cache, and arg builders.
- `Profiling`, `Advanced`: supplemental runtime utilities.

## Design Principles

- Type safety: GADTs for vectors/args, `Memory.BUFFER` for typed buffers, no Obj.t.
- Backend agnostic core: Uses `Framework_registry` to find backends; no backend
  specifics in core modules.
- Separation of concerns: Storage vs transfer vs façade modules; logging flows
  through `Log`.

## Key Types & Interfaces

- **Device**: `Device.t` with `capabilities.is_cpu`, `supports_fp64`, etc.
  Predicates `is_cpu`/`is_gpu` use capabilities instead of name heuristics.
- **Memory.BUFFER**: `device`, `size`, `elem_size`, `device_ptr`, transfer
  functions, `bind_to_kargs`, `free`. `Vector.DEVICE_BUFFER` reuses this module
  type.
- **Kernel args**: `Kernel_arg.t` GADT, `Kernel.args` typed builders; extensible
  `Framework_sig.kargs` wrapping/unwrapping.
- **Vector kinds**: `scalar_kind` + `custom_type`; locations (`CPU`, `GPU dev`,
  `Both`, `Stale_*`), auto-sync flag, subvector metadata (parent/start/depth).
- **Transfer**: `ensure_buffer`, `to_device`, `to_cpu`; zero-copy path chosen by
  backend; shared `bigarray_to_ptr` in `Vector_transfer`.

## Logging & Debugging

- Enable components via `SAREK_DEBUG=kernel,transfer,device,memory,execute,all`.
- Levels: Debug/Info/Warn/Error; `Error` uses `Log` internally.
- Format/printf helpers in `Log` (`debugf`, etc.).

## Usage Snippets

```ocaml
(* Vector creation *)
let v = Vector.create_float32 1024
let sub = Vector.sub_vector v ~start:100 ~len:50 ()

(* Memory allocation via Runtime *)
let buf = Runtime.alloc_float32 dev 256
Runtime.to_device buf ba; Runtime.from_device buf ba'

(* Kernel args and launch *)
let args = Kernel.create_args dev in
Kernel.set_arg_buffer args 0 buf;
let grid = Runtime.dims1d 4 and block = Runtime.dims1d 256 in
Kernel.launch kernel ~args ~grid:(Runtime.to_framework_dims grid)
  ~block:(Runtime.to_framework_dims block) ()
```

## Testing

- Core unit tests: `dune runtest sarek/core/test`
  - Covers Memory, Kernel, Device, Vector, Kernel_arg, Vector_storage,
    Vector_transfer.
- `make test-all` includes these tests (runs `dune runtest` under the hood).

## Backend Interaction

- Backends are discovered via `Framework_registry`; `Device.init` enumerates
  available frameworks, `Memory`/`Kernel` call through wrapped `kargs`.
- Zero-copy allocations are attempted in `Transfer.ensure_buffer` for CPU/host
  backends when supported by `B.Memory.alloc_zero_copy`.

## Notes / Future Work

- Partition/gather is host-side only; device-side redistribution is deferred to
  `Transfer`/backends.
- `Vector` façade delegates storage/sync helpers to split modules to keep it
  slim; further refactors should preserve the typed interfaces.
