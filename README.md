# SPOC - Stream Processing with OCaml

[![Build Status](https://github.com/mathiasbourgoin/SPOC/actions/workflows/build.yml/badge.svg)](https://github.com/mathiasbourgoin/SPOC/actions?query=branch%3Amaster)

SPOC is a GPU computing framework for OCaml that provides tools for writing and executing kernels on GPUs and other accelerators.

## What is SPOC?

**SPOC** is the foundational framework providing device abstraction, plugin architecture, and runtime infrastructure for GPU computing in OCaml.

**Sarek** is a PPX-based DSL (Domain Specific Language) that allows writing GPU kernels directly in OCaml syntax. Kernels written with Sarek compile to multiple backend targets (CUDA, OpenCL, Vulkan, Metal, CPU).

## Recent Development

This codebase has undergone significant modernization:

- **OCaml 5.4 support** with effect handlers and domains
- **Code quality improvements** across all GPU backends
- **Structured error handling** replacing untyped exceptions
- **Plugin-based architecture** for extensible backend support
- **Test coverage** with unit and end-to-end tests
- **Documentation** for all major components

The framework is actively maintained and uses modern OCaml features while preserving compatibility with existing SPOC code.

## Features

### GPU Kernel Development

Write GPU kernels in OCaml syntax using the `[%kernel ...]` PPX extension:

```ocaml
let%kernel vector_add (a : float32 vector) (b : float32 vector) (c : float32 vector) =
  let idx = get_global_id 0 in
  c.(idx) <- a.(idx) + b.(idx)
```

Kernels compile to multiple backends automatically without code changes.

### Backend Support

| Backend | Target | Status | Documentation |
|---------|--------|--------|---------------|
| **CUDA** | NVIDIA GPUs | ✓ | [sarek-cuda/](sarek-cuda/) |
| **OpenCL** | Multi-vendor GPUs/CPUs | ✓ | [sarek-opencl/](sarek-opencl/) |
| **Vulkan** | Cross-platform GPUs | ✓ | [sarek-vulkan/](sarek-vulkan/) |
| **Metal** | Apple Silicon/Intel Macs | ✓ | [sarek-metal/](sarek-metal/) |
| **Native** | CPU (parallel) | ✓ | [sarek/plugins/native/](sarek/plugins/native/) |
| **Interpreter** | CPU (debugging) | ✓ | [sarek/plugins/interpreter/](sarek/plugins/interpreter/) |

### Core Features

- **Type Safety**: GADTs and phantom types for compile-time guarantees
- **Zero-Copy**: Efficient memory sharing between host and device
- **Automatic Selection**: Runtime backend selection based on available hardware
- **Intrinsics**: Extensive library of GPU intrinsics (math, atomics, barriers)
- **Custom Types**: Support for records and variants in kernels
- **Debug Logging**: Controlled via `SAREK_DEBUG` environment variable

### Framework Architecture

```
spoc/              Low-level SDK and plugin interface
├── framework/     Plugin registration and backend interface
├── ir/            Intermediate representation (IR)
└── registry/      Intrinsic function registry

sarek/             Runtime and PPX compiler
├── core/          Device abstraction and memory management
├── framework/     Framework integration
├── ppx/           Sarek PPX compiler
├── sarek/         Unified execution dispatcher
└── plugins/       Native and Interpreter backends

GPU Backends:
├── sarek-cuda/    NVIDIA CUDA backend
├── sarek-opencl/  OpenCL backend (multi-vendor)
├── sarek-vulkan/  Vulkan/GLSL backend
└── sarek-metal/   Apple Metal backend
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Installation

### Prerequisites

- OCaml 5.4.0+ (local opam switch included in repository)
- dune 3.20+
- GPU backends (optional):
  - **CUDA**: NVIDIA driver + CUDA toolkit
  - **OpenCL**: OpenCL implementation for your device
  - **Vulkan**: Vulkan SDK + glslangValidator or Shaderc
  - **Metal**: macOS 10.13+ (included with Xcode)

The Native (CPU parallel) and Interpreter (CPU sequential) backends work without any GPU drivers.

### Building

```bash
# Build all packages
dune build

# Build specific backend
dune build sarek-cuda
dune build sarek-opencl
```

The framework uses dynamic linking, so you can build without GPU drivers installed. GPU support is detected at runtime.

### Verifying Installation

```bash
# Run benchmarks on all available devices
make benchmarks

# This will test kernels on all detected backends
# and report performance and correctness
```

## Usage

### Basic Example

```ocaml
open Sarek

(* Define a kernel *)
let%kernel saxpy (a : float32 vector) (x : float32 vector) 
                  (y : float32 vector) (alpha : float) =
  let i = get_global_id 0 in
  y.(i) <- alpha *. x.(i) +. a.(i)

let () =
  (* Initialize framework *)
  let device = Device.get_device 0 in
  
  (* Create vectors *)
  let n = 1024 in
  let a = Vector.create Float32 n in
  let x = Vector.create Float32 n in
  let y = Vector.create Float32 n in
  
  (* Execute kernel *)
  saxpy ~grid:(n/256, 1, 1) ~block:(256, 1, 1) a x y 2.5
```

### Backend Selection

```ocaml
(* List available devices *)
let devices = Device.list_devices () in
List.iter (fun dev ->
  Printf.printf "%s (%s)\n" 
    (Device.name dev) 
    (Device.backend_name dev)
) devices

(* Select specific backend *)
let cuda_device = Device.get_device_by_backend "CUDA" in
let opencl_device = Device.get_device_by_backend "OpenCL" in
```

See [sarek/sarek/README.md](sarek/sarek/README.md) for comprehensive usage documentation.

## Testing

```bash
# Run all tests
dune runtest

# Run specific backend tests
dune test sarek-cuda
dune test sarek-opencl

# Run with specific backend
SAREK_BACKEND=cuda dune runtest
```

See [COVERAGE.md](COVERAGE.md) for coverage measurement instructions.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and design
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Current project status
- [Backend Documentation](sarek-cuda/) - Individual backend READMEs

For API documentation, see inline comments and README files in each package directory.

## Requirements

- **OCaml**: 5.4.0+ (uses domains, effects)
- **System**: 64-bit Linux, macOS, Windows (limited testing)
- **GPU**: Optional - Native and Interpreter backends work on any system

## Project History

This work originates from Mathias Bourgoin's PhD thesis at UPMC-LIP6 laboratory (Paris) and was partially funded by the [OpenGPU](http://opengpu.net/) project. Development continued at Verimag laboratory (Grenoble, 2014-2015) and LIFO laboratory (Orléans, 2015-2018).

Current maintainer: Mathias Bourgoin ([Nomadic Labs](https://nomadic-labs.com))

## License

See [LICENSE.md](LICENSE.md) for license information.

## Resources

- **GitHub Pages**: [http://mathiasbourgoin.github.io/SPOC/](http://mathiasbourgoin.github.io/SPOC/)
- **GitHub Actions**: [Build status and CI](https://github.com/mathiasbourgoin/SPOC/actions)
- **Issues**: [Bug reports and feature requests](https://github.com/mathiasbourgoin/SPOC/issues)
