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

`output` needs one slot per launched thread: a thread whose `global_thread_id` is at least `n` skips the loop but still stores its zero.

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

That the device compiler unrolls by the requested factor was verified only for hipcc on gfx1100 (by inspecting the generated ISA); for the other compilers the factor is passed on, and whether they honour it is up to them. Sources: `Sarek_ir_cuda.ml:562`, `Sarek_ir_opencl.ml:603`, `Sarek_ir_metal.ml:771`, `Sarek_ir_glsl.ml:1247` (the test) and `:1380-1388`, `Sarek_ir_wgsl.ml:1200`, `Sarek_ir_ptx_stmt.ml:233`. A pragma that is dropped never changes the result, only the speed. The backends that honour the factor print the strings of the list joined by a space, so the factor reaches the device compiler as written.

Unrolling by hand (writing the loop body several times) is therefore only needed on the backends that drop the factor.

### Automatic unrolling

Sarek hints `#pragma unroll` (or `[[unroll]]` on GLSL) on a `for` loop by itself when

- both bounds are integer constants (literals, or `+ - *` of literals),
- the trip count is between 1 and 16, and
- the trip count times the largest expansion of the hinted loops nested in its body is at most 64 (three nested loops of 8 hint the inner two only).

A loop that already carries a pragma is left alone, so `pragma ["unroll 1"] (for ...)` opts one loop out. `SAREK_OPT_DISABLE=unroll` turns the pass off for every loop. The variable is read when the program starts; the generated source of a kernel is memoised with the policy it was made under, so nothing is stale. `SAREK_EXPLAIN=1` prints `sarek-explain <kernel> gen:<generator> unroll applied exact` once per kernel and generator. Only the CUDA, HIP, OpenCL, Metal and GLSL generators run the pass, so WGSL and PTX never get a hint. Unrolling does not change which operations run or their order. Sources: `Sarek_ir_unroll.ml` (`max_trip`, `annotate`, `trip_count`), called from `Sarek_ir_cuda.ml`, `Sarek_ir_opencl.ml`, `Sarek_ir_metal.ml` and `Sarek_ir_glsl.ml` (`generate_with_types`). All variables are in [Environment variables](environment.html).

### Launch bounds

`pragma ["sarek.max_threads N"] (body)` around the whole kernel body declares that the kernel is never launched with more than `N` threads per block (`1 <= N <= 1024`). A device compiler that assumes the worst case (1024 threads) must then cap registers per thread; a smaller bound lifts that cap. Measured offline with hipcc on gfx1100 (2026-10-16, a kernel holding 240 floats in registers): without the hint `.vgpr_count 192` with 216 bytes of scratch spill; with `__launch_bounds__(256)` `.vgpr_count 249` and no spill. The default is unchanged: a kernel without the pragma generates exactly the source it did before.

```ocaml
let k = [%kernel fun (out : int32 vector) (n : int32) ->
  pragma ["sarek.max_threads 256"]
    (let tid = thread_idx_x + (block_dim_x * block_idx_x) in
     if tid < n then out.(tid) <- tid)]
```

| Backend | What is emitted |
|---|---|
| CUDA (CUDA/C), HIP | `__global__ __launch_bounds__(N) void` |
| CUDA (PTX) | `.maxntid bx, by, bz` from the launch block (per-dimension in PTX), when the block respects `N` |
| OpenCL | `__attribute__((reqd_work_group_size(bx, by, bz)))` from the launch block, when it respects `N`; the kernel is then valid for that block only (the source is memoised per block) |
| Metal | `[[max_total_threads_per_threadgroup(N)]] kernel void` (generation tested; not compiled on a Mac in this change) |
| Vulkan (GLSL), WebGPU (WGSL) | nothing: the work-group size is the launch block already |
| Native, Interpreter | nothing in source |

On every backend, Native and Interpreter included, a launch whose block has more than `N` threads is refused before it starts with `Execute_error.Invalid_dimensions` naming the kernel, the bound and the block. A fused kernel (horizontal or epilogue fusion) takes the smallest bound of its members. The pragma is refused at compile time if `N` is not an integer in `1..1024`, if it is not around the whole kernel body, or if it appears twice. A matrix-pattern rewrite chooses its own launch geometry and drops the bound. Sources: `Sarek_launch_bounds.ml`, `Execute.check_launch_bounds`, the `kern_launch_bounds` field of `Sarek_ir_types.kernel`.

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
