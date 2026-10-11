---
layout: page
title: Matrix Units
---

# Matrix Units

This page shows how a matrix product (GEMM) in a Sarek program ends up on a GPU's matrix units, which results that can change, and how to inspect, enable or disable it. Matrix units compute small matrix products in hardware (AMD WMMA on RDNA3, NVIDIA tensor cores); in Sarek's IR they are cooperative-matrix statements (`SCoopmat`) of shape 16x16x16. The kernel language has no syntax for them, so a program reaches them in one of three ways:

| Path | You write | Float results change only after |
|---|---|---|
| [Automatic rewrite](#automatic-rewrite) | an ordinary GEMM kernel, launched with `Execute.run_vectors` | the accuracy opt-ins and `accept_automatic_float_rewrite` |
| [`Sarek_matrix.run`](#explicit-entry-points) | the same kernel, launched with this drop-in | the accuracy opt-ins (so does `Execute.run_vectors_outcome ~explicit:true`) |
| [`Sarek_gemm.Coopmat`](#explicit-entry-points) | nothing: a ready-made float16 GEMM | `accept_relaxed_accumulation` |

The `int32` rewrite is bit-identical to the original kernel and needs no opt-in, so it is the only one on by default.

## Where matrix units run

| Backend | Status |
|---|---|
| HIP, gfx11 (RDNA3) with 32-wide waves | executes: `float16`, `bfloat16` (float32 accumulator), 8-bit integers (`int32` accumulator) |
| HIP, any other device (gfx10, gfx12, CDNA, the gfx1036 iGPU) | refused with a message naming the rule |
| CUDA/C (the hidden NVRTC backend), sm_72+ with 32-wide warps | the `int32` rewrite is applied on an explicit request only (`Sarek_matrix.run`), never automatically (8-bit integers, `int32` accumulator, `nvcuda::wmma`). Evidence is compiler output: the generated code compiles with NVRTC for sm_80 and reaches the `u8` tensor-core instruction, but it has never been executed. ZLUDA lacks `wmma.*` and unsigned 8-bit `mma` and is declined. The float rewrites are not offered |
| Vulkan | executes the `int32` rewrite, when the device advertises the 8-bit integer configuration (RADV on the RX 7900 XTX, 64-wide subgroups). The float rewrites are not offered: GLSL lowers integer fragments only |
| Metal, OpenCL, WGSL, CUDA/PTX | refuse the statements |
| Native | not a rewrite backend: the automatic rewrite declines it |
| Interpreter | executes, as the reference implementation |

Sources: `Hip_coopmat.ml:43-54` (configurations), `Hip_coopmat.ml:57-58` (gfx11 and wave32 rule), `Cuda_coopmat.ml` (CUDA/C probe; CUDA/PTX keeps `coopmat = None` in `Cuda_plugin_base.ml`), `Sarek_ir_metal.ml:784`, `Sarek_ir_opencl.ml:616`, `Sarek_ir_wgsl.ml:1207`, `Sarek_ir_ptx_stmt.ml:254` (refusals), `Sarek_ir_glsl.ml:1395` (Vulkan lowering), `Sarek_ir_cuda.ml:574` (CUDA/C), `Matrix_specialise.ml:356-363` (Native). The CUDA lowering has been compiled but never run on hardware; the HIP lowering is validated on an RX 7900 XTX (gfx1100) in [`specs/coopmat-hip-wmma.md`](https://github.com/mathiasbourgoin/Sarek/blob/main/specs/coopmat-hip-wmma.md). Per-backend defaults of the HIP plugin are in the [sarek-hip README](https://github.com/mathiasbourgoin/Sarek/blob/main/sarek-hip/README.md).

## Compile-time advice

The PPX recognises a GEMM loop nest in a `[%kernel ...]` and reports it as an OCaml alert named `sarek_matrix`. It prints at the `for` loop, does not fail the build, and changes no generated code. For the naive kernel of the [matrix multiplication example](../examples/matrix_mul.html) it prints:

```
Alert sarek_matrix: Sarek_advice.matrix_product
Sarek: kernel 'matmul_naive' contains a matrix product (GEMM) that matrix units could compute.
  C[m,n] = sum over i = 0..(k - 1) of A[m,i] * B[i,n]
  A = a[((global_idx_y * k) + i)] (row-major, ld k)
  B = b[((i * n) + global_idx_x)] (row-major, ld n)
  C = c[((global_idx_y * n) + global_idx_x)] (row-major, ld n)
  m = global_idx_y, n = global_idx_x
  elements: float32 x float32 -> float32 accumulator
  configuration: 16x16x16 f16*f16+f32->f32 subgroup (operands rounded to float16)
  HIP gfx11 (wave32): executes
  CUDA/C sm_70+: none (the rewrite lowers the integer case only)
  Vulkan: none (the GLSL emitter lowers integer configurations only)
  Metal, OpenCL, WGSL, PTX, Native: no matrix-unit lowering
  accuracy: relaxed beyond the accumulation order: no matrix unit takes float32 operands, so they are rounded (float16: relative error up to 2^-11 each; bfloat16; or a three-term bfloat16 split, near float32); a rewrite needs Sarek_coopmat.accept_operand_rounding ~precision and accept_relaxed_accumulation
  Execute.run_vectors rewrites it onto matrix units only after the opt-in above AND Sarek_coopmat.accept_automatic_float_rewrite (and SAREK_MATRIX_REWRITE not off or exact); Sarek_matrix.run needs the opt-in above only.
  Silence with [@alert "-sarek_matrix"] on the kernel expression or its binding.
```

The `accuracy` line names the opt-ins the kernel would need, and the next line says what `Execute.run_vectors` does with the nest. The alert is emitted only when the element types have a matrix-unit configuration (see [Which kernels are rewritten](#which-kernels-are-rewritten)); a loop that looks like a GEMM but is not (a dot product, a triangular bound, a tiled kernel) gets no advice. A kernel that contains a `[%native]` block, in its body or a helper, is not analysed. Sources: `Sarek_matrix_alert.ml:1-30`, `:159-160` and `:185-189` (`[%native]`).

Silence it with the standard OCaml alert controls:

```ocaml
open Sarek

let gemm_f32 =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (c : float32 vector)
        (m : int32) (n : int32) (k : int32) ->
      let open Sarek_stdlib.Std in
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]
[@@alert "-sarek_matrix"]
```

The attribute also works on the kernel expression, as `([%kernel ...] [@alert "-sarek_matrix"])`. For a whole file use `[@@@alert "-sarek_matrix"]`; for a library, add `-alert -sarek_matrix` to its `(flags ...)`; to make the advice a build error, use `-alert @sarek_matrix`.

`SAREK_MATRIX_SCAN=<file>` makes the PPX append one line per analysed kernel and finding (recognised or near miss) to `<file>`. It is meant for scanning a code base for GEMM loops. The variable is not a dune dependency, so use a fresh build directory and `DUNE_CACHE=disabled`, or cached PPX output is not re-run.

## Automatic rewrite

`Execute.run_vectors` asks `Matrix_specialise` before every launch whether to run a rewritten kernel instead of the one it was given. The rewritten kernel writes exactly the outputs the original launch writes, with one subgroup per output tile instead of one thread per output. `Execute.run` and raw launches never specialise. `Execute.run_vectors_plain` launches the kernel as given.

### Modes

`SAREK_MATRIX_REWRITE` sets the mode when the program starts, and `Sarek_matrix.set_mode` changes it later:

| Mode | `int32` product | float product |
|---|---|---|
| `off` | original | original |
| `exact` | rewritten, bit-identical | original |
| `auto` (default) | rewritten, bit-identical | rewritten after the opt-ins below |

Accepted values of the variable, case-insensitively and ignoring surrounding spaces: `off`, `0`, `false`, `no`; `exact`; `auto`, `on`, `1`, `true`, `yes`. Unset or empty means `auto`; any other value prints one warning on stderr and means `auto` (`parse_mode` in `Matrix_specialise.ml`). A value such as `0` or `OFF` therefore switches the rewrite off.

```ocaml
(* Disable the rewrite for the rest of the process. *)
let () = Sarek_matrix.set_mode Sarek_matrix.Off
```

The mode also applies to `Sarek_matrix.run`: `Off` never rewrites and `Exact` rewrites only `int32` (`Matrix_specialise.ml:237-252`, and `:333-335` where `off` is checked first).

### Which kernels are rewritten

A kernel qualifies when all of these hold; otherwise the original runs and the reason is reported (see [Reporting](#reporting-and-cache)).

- The kernel body is one GEMM nest: `acc := acc + A[...] * B[...]` over a `for` loop with bounds that are the same for every thread, operands read from global vectors, and a single store to `C` (optionally with an epilogue such as `alpha * acc + beta * C[...]`). The kernel writes nothing else (no other array write, no atomics) and has exactly one such nest (`only_write_is`, `Matrix_specialise.ml:110-129`).
- Row, column and at most one batch index are affine in the loop variables and thread indices; each operand has a unit stride in one dimension (row-major or column-major). Guards are conjunctions of comparisons with a bound that is the same for every thread.
- No bound, initial value or epilogue reads `block_dim_*` or `grid_dim_*`, and no index involves `int64`.
- The element types have a configuration, below.
- The device is one of HIP with 32-wide waves (gfx11), Vulkan with 32- or 64-wide subgroups (the block is four subgroups: 128 or 256 threads), or CUDA/C with 32-wide warps (sm_72+, explicit request only); the configuration is offered by the device (Vulkan and CUDA/C: the `int32` case only); and the rewritten grid fits `max_grid_dims`.

| A and B | Accumulator, C | Result of the rewrite |
|---|---|---|
| `int32` | `int32` | bit-identical (each operand is split into four byte planes, 10 products per tile, 32-bit wrapping sum) |
| `float16` | `float32` (C `float32` or `float16`) | within a derived bound of the exact value, not bit-identical: the hardware chooses the addition order |
| `float32` | `float32` | operands are first rounded to `float16`, `bfloat16`, or a three-term `bfloat16` split; a larger deviation than the previous row |
| `int64`, `float64`, mixed | | not rewritten |

Sources: element cases `Sarek_matrix_advice.ml:43-48`; preconditions and the full list of rejected shapes in [`specs/matrix-pattern-advice.md`](https://github.com/mathiasbourgoin/Sarek/blob/main/specs/matrix-pattern-advice.md) and [`specs/matrix-pattern-rewrite.md`](https://github.com/mathiasbourgoin/Sarek/blob/main/specs/matrix-pattern-rewrite.md); bounds, initial value or epilogue reading `block_dim_*`/`grid_dim_*` (`uses_launch_dims`, `Sarek_matrix_scheme.ml:294`); grid size against `max_grid_dims` (`fits_device`, `Matrix_specialise.ml:288-299`); backend rule `Matrix_specialise.ml:268-285`.

An exception while analysing the kernel gives the original kernel, never a failed launch (`Matrix_specialise.ml:158-159` for the analysis, `:401-405` for building the rewritten kernel).

### Size classes

At each launch the region the original launch covers is computed from the scalar arguments and the grid; the kernel is then one of two shapes (`Matrix_specialise.ml:305-322`, `Sarek_matrix_direct.ml:40-55`):

| Class | When | Behaviour |
|---|---|---|
| direct | `float16` operands, row-major A, B and C, `float32` output stored as is with no epilogue, accumulator starting at zero, a rectangular region (`d = 0`) of whole 32x32 tiles, a reduction length divisible by 16 (`Sarek_matrix_direct.ml:40-52`) | each wave loads fragments straight from global memory |
| staged | everything else the rewrite accepts | tiles are staged through shared memory, with zero fill, so any size, layout, offset or batch stride works |

### Accuracy opt-ins

A float rewrite changes results, so the program must say it understands how. All opt-ins are process-wide and one-way (there is no way to withdraw one), take a non-empty `~reason` (`Invalid_argument` otherwise), keep the first reason given, and have no environment variable: they are statements in the program's source (`Sarek_coopmat.ml:78-129`).

| Call | Accepts | Needed for |
|---|---|---|
| `Sarek_coopmat.accept_relaxed_accumulation ~reason` | the hardware's addition order: result within a derived bound `gamma_K * sum(abs(a*b))` of the exact value | every float product |
| `Sarek_coopmat.accept_operand_rounding ~precision ~reason` | rounding float32 operands to `Operands_f16`, `Operands_bf16` or `Operands_bf16x3`; each precision is accepted separately | a `float32` product, for the chosen precision |
| `Sarek_coopmat.accept_automatic_float_rewrite ~reason` | that `Execute.run_vectors` applies the accepted deviations to the program's own float kernels, unprompted | the automatic path only |

The three calls together enable the automatic float rewrite of a `float32` kernel (`Sarek_coopmat.mli`, section "Automatic float rewrite"):

```ocaml
let () =
  Sarek_coopmat.accept_relaxed_accumulation
    ~reason:"GEMM tolerance checked against the gamma_K bound" ;
  Sarek_coopmat.accept_operand_rounding
    ~precision:Sarek_coopmat.Operands_bf16
    ~reason:"bfloat16 operands are accurate enough for this model" ;
  Sarek_coopmat.accept_automatic_float_rewrite
    ~reason:"let Execute rewrite the model's float GEMMs"
```

The two accuracy opt-ins alone admit matrix-unit configurations for code that asks for them (`Sarek_gemm.Coopmat`, `Sarek_matrix.run`); they do not enable the automatic rewrite. Without `accept_automatic_float_rewrite`, a float launch through `Execute.run_vectors` runs the original kernel and the report names the missing call (`automatic_float_optin` in `Matrix_specialise.ml`). The `int32` rewrite needs none of them.

Precision of a `float32` product: if the program asks for one (`~precision` on `Sarek_matrix.run` or `Execute.run_vectors_outcome`), that precision must have been accepted. Otherwise `float16` is used when accepted, else `bfloat16`. The `bfloat16x3` split reaches near-float32 accuracy but was measured no faster than the scalar kernel, so it is used only on request (`Matrix_specialise.ml:205-224`). Bounds measured on gfx1100 for each precision are in the Results of [`specs/matrix-pattern-rewrite.md`](https://github.com/mathiasbourgoin/Sarek/blob/main/specs/matrix-pattern-rewrite.md); they assume finite operands within the narrower type's range (`float16` overflows above 65504).

The bound each configuration is tested against, in units of `sum(abs(a*b))` over the reduction, with `gamma_K = K*u / (1 - K*u)`, `u = 2^-24` and `K` the number of products added:

| Operands and precision | Error bound | Measured max error on gfx1100, as a multiple of `gamma_K` |
|---|---|---|
| `float16` operands | `gamma_K` | 0.010 to 0.028 |
| `float32` rounded to `Operands_f16` | `2*u16 + u16^2 + gamma_K`, `u16 = 2^-11` | 0.77 to 15.8 |
| `float32` rounded to `Operands_bf16` | `2*ub + ub^2 + gamma_K`, `ub = 2^-8` | 6.0 to 114 |
| `float32` as `Operands_bf16x3` | `2^-24 + gamma_6K` | 0.04 to 0.15 |

The measurements are from [`specs/matrix-pattern-rewrite.md`](https://github.com/mathiasbourgoin/Sarek/blob/main/specs/matrix-pattern-rewrite.md), Results (RX 7900 XTX, 2026-10-11; shapes 64x64x128, 100x37x129, 256x256x1024; operands uniform in [-1, 1)). A ratio above 1 for the rounded precisions is expected: the operand rounding is a larger error than `gamma_K` and is covered by the bound of its own row. Only the first row is an accumulation-order error alone; `gamma_K` is defined in `docs/design/f16-relaxed-accuracy.md` section 5.2.

### The interpreter

By default the interpreter is never rewritten: it stays the independent reference that differential tests compare against, and the rewrite is slower there. It is rewritten only on an explicit request (`Sarek_matrix.run`, or `Execute.run_vectors_outcome ~explicit:true`), and not for a launch on a stream. Every other backend is declined before any analysis (`Matrix_specialise.ml:324-331`).

### Reporting and cache

`Execute.run_vectors_outcome` is `run_vectors` that also returns the decision:

```ocaml
let launch_and_say ~device ~ir ~args ~block ~grid =
  let outcome =
    Sarek.Execute.run_vectors_outcome ~device ~ir ~args ~block ~grid ()
  in
  print_endline (Sarek_execute.Matrix_specialise.outcome_text outcome)
```

`Matrix_specialise.set_reporter` registers a callback that receives every decision about a kernel that is a matrix product. `SAREK_MATRIX_REPORT=1` (exactly `1`, read when the program starts) prints each distinct kernel, device and outcome once on stderr:

```
[sarek matrix] <kernel> on <device>: rewritten: <lowering>, <staged|direct>, <configuration>
[sarek matrix] <kernel> on <device>: original kernel: <reason>
```

Kernels that are not matrix products, devices that can never rewrite, and mode `off` are not reported (`Matrix_specialise.ml:375-388`, `:390-393`).

The analysis result and the rewritten kernels are cached per kernel IR (by physical identity), per lowering and size class, so repeated launches of one kernel analyse it once. Mode, opt-ins and device are checked at every launch. A kernel whose IR is rebuilt for each launch is analysed each time, as it also misses the launch memo.

## Explicit entry points

`Sarek_matrix.run` takes the `device`, `ir`, `args`, `block` and `grid` of `Execute.run_vectors` plus `?precision`, and returns what ran. It also rewrites on the interpreter, and a float kernel needs only the accuracy opt-ins, not `accept_automatic_float_rewrite`. `Execute.run_vectors_outcome ~explicit:true` is the same explicit request and waives `accept_automatic_float_rewrite` in the same way:

```ocaml
let explicit_launch ~device ~ir ~args ~block ~grid =
  match
    Sarek_matrix.run
      ~precision:Sarek_coopmat.Operands_bf16
      ~device ~ir ~args ~block ~grid ()
  with
  | Sarek_matrix.Rewritten _ as outcome ->
      print_endline (Sarek_matrix.outcome_text outcome)
  | Sarek_matrix.Original why -> prerr_endline ("original kernel ran: " ^ why)
```

`Sarek_gemm.Coopmat.run` launches a ready-made float16 GEMM `D = A x B + C` (A `m x k` and B `k x n` in `float16`, C and D `m x n` in `float32`, all row-major). The shape must tile exactly (with the default tiling: `m` and `n` multiples of 32, `k` a multiple of 16, and an even number of 32x32 tiles; otherwise `Invalid_argument`), and it needs `accept_relaxed_accumulation` and a device with the configuration (`Sarek_gemm_coopmat.ml:53-70`, `:283-309`):

```ocaml
let gemm_f16 ~device ~a ~b ~c ~d =
  Sarek_gemm.Coopmat.run ~device ~a ~b ~c ~d ~m:256 ~n:1024 ~k:3072 ()
```

## Related

- [Launch window and fusion](performance.html): the rewritten launch is deferred and fusable like any other.
- [Environment variables](environment.html): every variable on this page.
- [Memory and transfers](memory.html).
