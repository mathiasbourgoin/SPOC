---
layout: page
title: Kernel Language Notes
---

# Kernel Language Notes

This page covers three details of the kernel language that are easy to get wrong: loops (what `for` can and cannot express, and how to ask for unrolling), private per-thread arrays, and the size rule for `create_array` and `let%shared`. The rest of the language is introduced in [Concepts](concepts.html) and the [PPX README](https://github.com/mathiasbourgoin/Sarek/blob/main/sarek/ppx/README.md).

## Loops

`for i = a to b do ... done` and `for i = a downto b do ... done` run `i` from `a` to `b` inclusive, one step at a time. There is no step: a strided loop is a `while` with a mutable counter.

```ocaml
open Sarek

(* Each thread sums every (grid size)-th element, starting at its own index. *)
let grid_stride_sum =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let acc = mut 0.0 in
      let i = mut global_thread_id in
      while i < n do
        acc := acc +. input.(i) ;
        i := i + (block_dim_x * grid_dim_x)
      done ;
      output.(global_thread_id) <- acc]
```

### Unrolling

`pragma ["unroll N"] (loop)` asks the device compiler to unroll the `for` or `while` loop it wraps by a factor of `N`. It works for loops whose trip count is only known at run time. `pragma ["unroll"]` asks for the compiler's choice.

```ocaml
let unrolled_sum =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (k : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      let acc = mut 0.0 in
      pragma
        ["unroll 8"]
        (for i = 0 to k - 1l do
           acc := acc +. input.((tid * k) + i)
         done) ;
      output.(tid) <- acc]
```

What each backend does with the pragma on a loop:

| Backend | Output | Factor `N` |
|---|---|---|
| CUDA (CUDA/C), HIP | `#pragma unroll N` | passed to the device compiler |
| OpenCL | `#pragma unroll N` | passed to the device compiler |
| Metal | `#pragma unroll N` | passed to the device compiler |
| Vulkan (GLSL) | `[[unroll]]` for exactly `["unroll"]` on a loop; any other hint, including `["unroll 8"]`, is dropped | dropped |
| WebGPU (WGSL) | nothing | dropped |
| CUDA (PTX) | nothing | dropped |
| Native, Interpreter | no source is generated | not applicable |

That the device compiler unrolls by the requested factor was verified only for hipcc on gfx1100 (by inspecting the generated ISA); for the other compilers the factor is passed on, and whether they honour it is up to them. Sources: `Sarek_ir_cuda.ml:557`, `Sarek_ir_opencl.ml:598`, `Sarek_ir_metal.ml:766`, `Sarek_ir_glsl.ml:1240` (the test) and `:1368-1376`, `Sarek_ir_wgsl.ml:1187`, `Sarek_ir_ptx_stmt.ml:233`. A pragma that is dropped never changes the result, only the speed. The backends that honour the factor print the strings of the list joined by a space, so the factor reaches the device compiler as written.

Unrolling by hand (writing the loop body several times) is therefore only needed on the backends that drop the factor.

### Automatic unrolling

Sarek hints `#pragma unroll` (or `[[unroll]]` on GLSL) on a `for` loop by itself when

- both bounds are integer constants (literals, or `+ - *` of literals),
- the trip count is between 1 and 16, and
- the trip count times the largest expansion of the hinted loops nested in its body is at most 64 (three nested loops of 8 hint the inner two only).

A loop that already carries a pragma is left alone, so `pragma ["unroll 1"] (for ...)` opts one loop out. `SAREK_AUTO_UNROLL=0` turns the pass off for every loop. The variable is read at each code generation, but the generated source of a kernel is memoised, so set it before the kernel's first launch (or call `Execute.clear_launch_cache`). Only the CUDA, HIP, OpenCL, Metal and GLSL generators run the pass, so WGSL and PTX never get a hint. Unrolling does not change which operations run or their order. Sources: `Sarek_ir_unroll.ml:11-18` (limits and variable), `:121-130` (`annotate`), `:45-52` (inclusive bounds), `Sarek_ir_cuda.ml:958`, `Sarek_ir_opencl.ml:1166`, `Sarek_ir_metal.ml:1234`, `Sarek_ir_glsl.ml:2149`. All variables are in [Environment variables](environment.html).

## Private and shared arrays

| Declaration | Lives in | One copy per |
|---|---|---|
| `create_array n Local` | registers or private memory | thread |
| `create_array n Shared`, `let%shared (t : ty) = n` | shared memory | block |

Global memory is not allocated from a kernel: pass a vector parameter instead. `create_array n Global` is refused at compile time with a located error.

The size `n` of a `Local` or `Shared` array, and of `let%shared (t : ty) = n`, must be a positive compile-time `int32` constant: a literal such as `16l`, or literals combined with `+`, `-` and `*`. The PPX folds the expression to one literal before any backend sees it, and every intermediate result must fit in `int32`. A runtime size is refused, because OpenCL C and Metal reject variable-length arrays and PTX and WGSL need a literal extent. To handle a runtime length `n`, declare a constant upper bound and index only the first `n` slots. The size-less form `let%shared (t : ty) = ()` is not subject to this rule.

```ocaml
(* Each thread keeps its four inputs in a private array. *)
let private_array =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      let window = create_array 4l Local in
      for j = 0 to 3 do
        window.(j) <- input.((tid * 4l) + j)
      done ;
      output.(tid) <-
        window.(0) +. window.(1) +. window.(2) +. window.(3)]

(* A 16 x 16 tile shared by the block: the size is a folded constant. *)
let tile_in_shared =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) ->
      let open Sarek_stdlib.Std in
      let%shared (tile : float32) = 16l * 16l in
      let tid = thread_idx_x in
      tile.(tid) <- input.(global_thread_id) ;
      block_barrier () ;
      output.(global_thread_id) <- tile.(tid)]
```

Source: `Sarek_typer.ml` (`const_int_value`, `fold_const_size`, `check_create_array`).

On WGSL a constant-size `Local` array lowers to a function-scope `var a : array<T, N>`.
