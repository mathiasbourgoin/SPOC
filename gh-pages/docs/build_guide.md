---
layout: page
title: Building SPOC from Source
---

# Building and Testing SPOC

Sarek is not yet published to the opam repository: build it from a clone of the repository, for use in your own project or for development.

## Prerequisites

- **OCaml** >= 5.4.0
- **Dune** >= 3.15
- **ocamlfind**
- **ctypes** & **ctypes-foreign**
- **ppxlib**
- **alcotest** (for running tests)

## Compilation & Installation

### 1. Install Dependencies
```bash
git clone https://github.com/mathiasbourgoin/Sarek.git
cd Sarek
opam install . --deps-only --working-dir
```

### 2. Build and Install
```bash
dune build
dune install
```

## Running Tests

### Unit Tests
Execute the full test suite using dune:
```bash
dune test
```

### Benchmarks
Run the automated benchmark suite to verify performance across available backends:
```bash
# Run the full suite on all available devices
make benchmarks
```

## API Documentation
You can generate the latest API documentation locally:
```bash
dune build @doc
```
The documentation will be available in `_build/default/_doc/_html/`.
