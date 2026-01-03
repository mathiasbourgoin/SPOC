# Sarek

**Sarek** is an embedded DSL for writing GPGPU kernels in OCaml using PPX syntax extensions.

## Quick Start

```ocaml
(* Define a kernel *)
let vector_add =
  [%kernel fun (a : int32 vector) (b : int32 vector) (c : int32 vector) ->
    c.(thread_idx_x) <- a.(thread_idx_x) + b.(thread_idx_x)]

(* Use with SPOC *)
let () =
  let dev = Spoc.Devices.init () |> Array.get 0 in
  let a = Spoc.Vector.create int32 1024 in
  let b = Spoc.Vector.create int32 1024 in
  let c = Spoc.Vector.create int32 1024 in
  Kirc.run vector_add (a, b, c) (1, 1, 1) (1024, 1, 1) 0 dev
```

## Features

- **PPX syntax**: `[%kernel ...]` for kernel definitions, `[%ktype ...]` for GPU types
- **Type inference**: Automatic type checking with GPU-specific types
- **Kernel fusion**: Automatic fusion of producer-consumer kernel pipelines
- **BSP model**: `superstep { }` blocks with barrier synchronization
- **Convergence tracking**: Compile-time warp divergence detection

## Dependencies

Sarek depends on **SPOC** for device management and memory transfers.

## Documentation

- [FUSION.md](FUSION.md) - Kernel fusion system
- [BSP.md](BSP.md) - Bulk Synchronous Parallel model
- [Sarek_stdlib](../Sarek_stdlib/README.md) - Standard library intrinsics

## Examples

See the **SpocLibs/Benchmarks** and **SpocLibs/Samples** directories.
