# sarek/core - Runtime Core Modules

This layer sits between the Sarek PPX (IR generation) and backend plugins
(CUDA/OpenCL/Vulkan/Native/Interpreter). It provides type-safe devices, memory,
vectors, transfers, and kernel execution, with unified logging and testing.

## Module Map

- `Log` / `Error`: Unified logging (`SAREK_DEBUG` env); `Error` delegates to `Log`.
- `Device`: Device enumeration/predicates using `capabilities.is_cpu`, etc.
- `Memory`: Typed device buffers (`Memory.BUFFER`), alloc/transfer helpers, `bind_to_kargs`.
- `Kernel` / `Kernel_arg`: Typed kernel compilation/launch; extensible `kargs`; GADT args.
- `Vector` (+ `Vector_types`, `Vector_transfer`, `Vector_storage`):
  kinds/locations, sync callbacks, host pointers, storage/copy/subvector/partition helpers.
- `Transfer`: Device/host transfers, buffer allocation/ensure-buffer, zero-copy path.
- `Runtime`: High-level run helpers, dims re-export, kernel cache.
- `Profiling`, `Advanced`: supplemental utilities.
- Related components: [spoc/](../spoc/), [spoc/framework/](../spoc/framework/),
  [spoc/ir/](../spoc/ir/), [spoc/registry/](../spoc/registry/) for SDK types,
  plugin interfaces, IR, and registries.
- Additional dirs: [sarek/ppx/](../sarek/ppx/) (Sarek PPX compiler),
  [sarek/plugins/](../sarek/plugins/) (backend implementations),
  [sarek/tests/](../sarek/tests/) (broader test suites).

## Design Principles

- Type safety (no `Obj.t`): GADTs for vectors/args, typed buffers.
- Backend-agnostic core: uses `Framework_registry`; no backend-specific code.
- Separation: storage vs transfer vs façade; single logging pipeline.

## Key Types & Interfaces

- Device predicates (`is_cpu`/`is_gpu`) rely on capabilities; `Device.capabilities` mirrors `Framework_sig`.
- `Memory.BUFFER`: `device`, `size`, `elem_size`, `device_ptr`, transfer fns, `bind_to_kargs`, `free`. `Vector.DEVICE_BUFFER` reuses it.
- `Kernel_arg.t` GADT, `Kernel.args` builders; extensible `Framework_sig.kargs`.
- Vector kinds (`scalar_kind`/`custom_type`), locations (`CPU`, `GPU dev`, `Both`, `Stale_*`), subvector metadata.
- Transfer uses shared `bigarray_to_ptr` (`Vector_transfer`) and attempts zero-copy allocations.

## Logging & Debugging

- `SAREK_DEBUG=transfer,kernel,device,memory,execute,all` (comma-separated).
- Levels: Debug/Info/Warn/Error via `Log`; `Error` formats to `Log`.

## Usage Snippets

```ocaml
(* Vector creation *)
let v = Vector.create_float32 1024
let sub = Vector.sub_vector v ~start:100 ~len:50 ()

(* Memory allocation and transfer *)
let buf = Runtime.alloc_float32 dev 256
Runtime.to_device buf ba; Runtime.from_device buf ba'

(* Kernel launch *)
let args = Kernel.create_args dev in
Kernel.set_arg_buffer args 0 buf;
let grid = Runtime.dims1d 4 and block = Runtime.dims1d 256 in
Kernel.launch k ~args ~grid:(Runtime.to_framework_dims grid)
  ~block:(Runtime.to_framework_dims block) ()
```

## Testing

- Run core tests: `dune runtest sarek/core/test`
  - Covers Memory, Kernel, Device, Vector, Kernel_arg, Vector_storage, Vector_transfer.
- `make test-all` includes these tests.

## Backend Interaction

- Backends discovered via `Framework_registry`; `Memory`/`Kernel`/`Transfer` unwrap/wrap `kargs`.
- Zero-copy attempted when backend `alloc_zero_copy` supports CPU/host sharing.

## Notes

- Partition/gather is host-side only; device redistribution is backend/Transfer responsibility.
- `Vector` façade delegates storage/sync helpers to split modules to stay small.
