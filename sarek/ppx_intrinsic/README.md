# Sarek PPX Intrinsic

**Package**: `sarek.ppx.intrinsic`  
**Purpose**: PPX for defining GPU intrinsic functions

Minimal PPX extension for declaring GPU intrinsic functions and types that work across multiple backends.

## Overview

The `%sarek_intrinsic` PPX allows defining functions and types that generate backend-specific code at runtime while providing OCaml implementations for CPU execution.

## Usage

```ocaml
(* Define an intrinsic type *)
let%sarek_intrinsic float32 =
  {device = (fun _ -> "float"); ctype = Ctypes.float}

(* Define an intrinsic function. [device] maps a framework name ("CUDA",
   "OpenCL", "Metal", ...) to a template in which each %s is an argument. *)
let%sarek_intrinsic (sin_float32 : float32 -> float32) =
  {
    device = (function "OpenCL" | "Metal" -> "sin(%s)" | _ -> "sinf(%s)");
    ocaml = sin;
  }
```

## Using intrinsics from another library

A project outside this repository can declare its own intrinsics and call them
from its kernels. Two processes need to see each declaration: the **kernel
PPX**, which types the call and lowers it to an intrinsic call, and the
**program**, whose runtime registry supplies the device spelling at JIT time
and the host implementation to the Interpreter. The recipe below covers both.
It is exercised end to end by `sarek/tests/downstream/` (positive) and
`sarek/tests/negative/test_unlinked_intrinsic.ml` (the error you get when the
second stanza is missing).

**1. The intrinsic library.** Any directory, any name; keep dune's default
`(wrapped true)`.

```dune
; src/intrinsics/dune
(library
 (name my_intrinsics)
 (kind ppx_rewriter)              ; lets kernel stanzas list it in (pps ...)
 (libraries spoc.registry sarek.ppx.lib)
 (preprocess
  (pps sarek.ppx.intrinsic)))
```

```ocaml
(* src/intrinsics/bits.ml *)
let device = function
  | "OpenCL" -> "as_float(%s)"
  | "Metal" -> "as_type<float>(%s)"
  | _ -> "__int_as_float(%s)"              (* CUDA, HIP *)

let%sarek_intrinsic (float_of_bits : int32 -> float32) =
  {device; ocaml = Int32.float_of_bits}
```

**2. Every stanza that compiles kernels** lists the library twice: in
`libraries` (the program links it) and in `pps` (the kernel PPX links it).

```dune
(executable
 (name main)
 (libraries sarek sarek.stdlib my_intrinsics)
 (preprocess
  (pps sarek.ppx my_intrinsics)))
```

**3. In a kernel**, call it as `Module.name`, the last component of its module
path. Alias the module outside the kernel (a kernel resolves `M.x`, not
`A.M.x`):

```ocaml
module Bits = My_intrinsics.Bits

let k =
  [%kernel
    fun (a : int32 vector) (b : float32 vector) ->
      let i = thread_idx_x in
      b.(i) <- Bits.float_of_bits a.(i)]
```

What each piece is for:

- **Module path.** The intrinsic is registered under the module's real OCaml
  path, `["My_intrinsics"; "Bits"]`, built from the library name dune passes
  to the PPX (`--cookie library-name=...`). The kernel PPX emits that path
  as-is: as the Native backend's call target and as a compile-time existence
  witness. A `(wrapped false)` library breaks this: its modules are top-level
  but the path still carries the library prefix. You get a compile error
  (`Unbound module My_intrinsics`), not a silent miscompile. A module named
  after its library (`my_intrinsics.ml`) is registered as `["My_intrinsics"]`.
  If a wrapped library has a hand-written main module, that module must
  re-export each intrinsic module (`module Bits = Bits`). Otherwise the
  witness also fails with `Unbound module`.
- **`(kind ppx_rewriter)` plus `pps`.** Registration is a side effect of
  module initialisation. The kernel PPX knows only intrinsics whose modules
  were linked into its own driver executable. dune refuses a plain library in
  `pps`, and it links every module of a `ppx_rewriter` library listed there.
  If you prefer a separate driver library
  (`(library (name my_ppx) (kind ppx_rewriter) (libraries sarek.ppx my_intrinsics) (modules))`),
  the intrinsic library then needs `(library_flags (-linkall))`. Without it,
  dune drops the unreferenced modules from the driver.
- **The program.** No `-linkall` is needed. Each kernel references every
  intrinsic it calls by its full path, so the defining module is linked and
  its runtime registration runs.

If the kernel PPX does not know a qualified name, compilation fails at the
call site:

```
Error: Unknown kernel function or constant Bits.float_of_bits: the kernel PPX
has no intrinsic, [@sarek.module] item or binding with this name, so it cannot
be compiled to device code. If Bits.float_of_bits is declared with
let%sarek_intrinsic in another library, that library must be linked into the
PPX driver: give it (kind ppx_rewriter) and list it in (preprocess (pps
sarek.ppx <library>)) ...
```

When the PPX knows other names under the same module, the message lists them.
That covers typos and a partly linked library. Before this check, the PPX
accepted any unknown `M.f`. It printed the call verbatim into device source,
and the failure surfaced only at JIT time, from the device compiler.

**Backend coverage.** Native calls the `ocaml` implementation directly. The
Interpreter calls it through the host implementation that `let%sarek_intrinsic`
registers. It does this when every argument and the result are scalars
(`int32`, `int64`, `float32`, `float`/`float64`, `bool`, `unit`). For other
signatures it raises `Unknown intrinsic: <path>`. The C-family source
backends expand the `device` template: CUDA C (NVRTC), HIP, OpenCL and Metal.
The direct PTX emitter (framework `CUDA/PTX`), GLSL (Vulkan) and WGSL (WebGPU)
do not consult the registry. They refuse such a kernel with an error that
names the intrinsic. Short module names share one namespace in kernels: two
linked libraries that both define `Bits.f` collide, and the last one linked
wins.

## Features

- Dual code generation (GPU device code + OCaml host code)
- Backend-specific code via device functions
- Automatic registration with Sarek_registry
- Type-safe intrinsic declarations

## Architecture

For a function, the PPX generates:
1. Runtime registration for JIT compilation (`Sarek_registry.register_fun`)
2. PPX-time registration for compile-time type checking
   (`Sarek_ppx_registry.register_intrinsic`)
3. The OCaml implementation, bound under the intrinsic's name
4. For an all-scalar signature, the host implementation the Interpreter calls
   (`Sarek_registry.register_host_impl`)

This separation breaks circular dependencies between sarek_stdlib and sarek_ppx.
