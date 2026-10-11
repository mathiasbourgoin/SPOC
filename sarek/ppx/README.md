# sarek/ppx - Sarek PPX Compiler

The Sarek PPX (`sarek.ppx`) turns `[%kernel fun ... -> ...]` expressions into kernels. At build time it parses the kernel, type-checks it, checks barrier safety, rewrites it into a form every backend can express, lowers it to the Sarek IR, and emits an OCaml value that carries the IR and a native OCaml version of the kernel. Backends turn the IR into CUDA, HIP, OpenCL, GLSL, MSL, PTX or WGSL at run time, or run it directly (Native, Interpreter).

Enable it in a dune stanza with `(preprocess (pps sarek.ppx))`; the full stanza is in [Using Sarek in a dune project](../../gh-pages/docs/getting_started.md#using-sarek-in-a-dune-project).

## Writing a kernel

A kernel is the expression `[%kernel fun (params) -> body]`. Annotate every parameter: `float32 vector`, `int32 vector`, `float32`, `int32`, a record type declared with `[@@sarek.type]`, and so on. GPU intrinsics (`global_thread_id`, `thread_idx_x`, `block_barrier`, math functions) come from `Sarek_stdlib.Std`; open it inside the kernel.

```ocaml
let vector_add =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (c : float32 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) + b.(tid)]
```

The expression has the type of a pair whose second component carries the IR (`Sarek.Kirc_types`). Launch it with `Execute.run_vectors` ([sarek/sarek/README.md](../sarek/README.md)). Write integer literals with the `l` suffix (`0l`, `255l`) to get `int32`.

## Compilation pipeline

The stages run in this order for each `[%kernel ...]`:

| # | Stage | Module | What it does |
|---|---|---|---|
| 1 | Parse | `Sarek_parse` | OCaml syntax to the Sarek AST |
| 2 | Type inference | `Sarek_typer` | Typed AST; GPU type constraints |
| 3 | Convergence check | `Sarek_convergence` | Rejects barriers reached by only some threads of a block |
| 4 | Monomorphization | `Sarek_mono` | One concrete copy of each polymorphic helper per use |
| 5 | Defunctionalization | `Sarek_defunc` | Removes first-class function values bound by `let` |
| 6 | Tail-recursion elimination | `Sarek_tailrec` | Tail calls become loops; bounded non-tail recursion is inlined |
| 7 | Tag erasure | `Sarek_tag_erasure` | Erases statically known variant tags (device path only) |
| 8 | Lowering | `Sarek_lower_ir` | Sarek IR (`Sarek_ir_types`) |
| 9 | Quoting | `Sarek_quote`, `Sarek_native_gen` | OCaml expression holding the IR and the native kernel |
| 10 | Matrix advice | `Sarek_matrix_alert` | An OCaml alert for loop nests that matrix units could compute |

The native OCaml kernel is built from the typed AST as it stands after stage 5, so it keeps recursion and variant tags that the device path removes (`Sarek_ppx.ml`, `expand_kernel`).

## Language features

### Records and variants

Declare a record or variant type with `[@@sarek.type]`; vectors of it have a layout shared by the host and every backend.

```ocaml
type float32 = float

type point = {x : float32; y : float32} [@@sarek.type]

type color = Red | Value of float32 [@@sarek.type]

let point_offset =
  [%kernel
    fun (points : point vector) (result : point vector) (offset : float32) ->
      let open Sarek_stdlib.Std in
      let tid = global_idx_x in
      let p = points.(tid) in
      result.(tid) <- {x = p.x +. offset; y = p.y +. offset}]

let variant_demo =
  [%kernel
    fun (xs : float32 vector) (dst : float32 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        let c = if xs.(tid) > 0.0 then Value xs.(tid) else Red in
        match c with
        | Red -> dst.(tid) <- 0.0
        | Value v -> dst.(tid) <- v +. 1.0]
```

### Shared memory and supersteps

`let%shared (name : type) = size in` allocates block-local memory. A `let%superstep name = body in` block ends with a barrier, so phases that read what other threads wrote are separated without writing the barrier by hand. This kernel transposes one 16 by 16 tile per block; launch it with 16 by 16 blocks.

```ocaml
let tile_transpose =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (width : int32) (height : int32) ->
      let open Sarek_stdlib.Std in
      let%shared (tile : float32) = 256l in
      let tx = thread_idx_x in
      let ty = thread_idx_y in
      let x = (block_idx_x * 16l) + tx in
      let y = (block_idx_y * 16l) + ty in

      (* Superstep 1: load a tile from global memory *)
      let%superstep load =
        if x < width && y < height then
          tile.((ty * 16l) + tx) <- input.((y * width) + x)
      in

      (* Superstep 2: store it transposed *)
      let%superstep store =
        let out_x = (block_idx_y * 16l) + tx in
        let out_y = (block_idx_x * 16l) + ty in
        if out_x < height && out_y < width then
          output.((out_y * height) + out_x) <- tile.((tx * 16l) + ty)
      in
      ()]
```

A raw `block_barrier ()` is also available.

### Atomics

Atomic operations on shared (`int32 array`) and global (`int32 vector`) memory are in `Sarek_stdlib.Gpu`: `atomic_add_int32` for shared memory, `atomic_add_global_int32` for vectors. This histogram counts into shared memory and merges into the global result.

```ocaml
let histogram =
  [%kernel
    fun (input : int32 vector) (bins : int32 vector) (n : int32) (num_bins : int32) ->
      let open Sarek_stdlib.Std in
      let open Sarek_stdlib.Gpu in
      let%shared (local_hist : int32) = 256l in
      let tid = thread_idx_x in
      let gid = global_thread_id in
      let%superstep init = if tid < num_bins then local_hist.(tid) <- 0l in
      let%superstep count =
        if gid < n then begin
          let bin = input.(gid) mod num_bins in
          let _old = atomic_add_int32 local_hist bin 1l in
          ()
        end
      in
      let%superstep merge =
        if tid < num_bins then begin
          let _old = atomic_add_global_int32 bins tid local_hist.(tid) in
          ()
        end
      in
      ()]
```

### Local helper functions and control flow

Helpers defined inside a kernel are inlined. Mutable local variables are declared with `mut`, read by name and written with `:=`. `if`, `for` and `while` work as in OCaml; `match` is for variants (a `match` on an integer literal is refused, use `if`).

```ocaml
let control_flow =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) ->
      let open Sarek_stdlib.Std in
      let clamp (x : float32) (lo : float32) (hi : float32) : float32 =
        if x < lo then lo else if x > hi then hi else x
      in
      let square (x : float32) : float32 = x *. x in
      let tid = global_idx_x in
      let value = input.(tid) in
      let result = if value > 0.0 then value else 0.0 -. value in

      let sum = mut 0.0 in
      for i = 0 to 10 do
        sum := sum +. (float i *. value)
      done ;

      let x = mut value in
      while x > 1.0 do
        x := x /. 2.0
      done ;

      let r = tid mod 3l in
      let final = if r = 0l then result else if r = 1l then sum else x in
      output.(tid) <- clamp (square final) 0.0 100.0]
```

### Polymorphic helpers

A module-level function marked `[@sarek.module]` can be polymorphic. Each kernel that calls it gets a copy specialised to the argument types, so one definition serves `int32`, `float32` and record vectors. OCaml must type-check the helper outside a kernel, so the two aliases `float32` and `vector` give those names an OCaml meaning.

```ocaml
type 'a vector = 'a array

type point3d = {px : float32; py : float32; pz : float32} [@@sarek.type]

let[@sarek.module] do_transpose (input : 'a vector) (output : 'a vector)
    (width : int) (height : int) (tid : int) : unit =
  let n = width * height in
  if tid < n then begin
    let col = tid mod width in
    let row = tid / width in
    output.((col * height) + row) <- input.((row * width) + col)
  end

let transpose_int32 =
  [%kernel
    fun (input : int32 vector) (output : int32 vector) (width : int32) (height : int32) ->
      let open Sarek_stdlib.Std in
      do_transpose input output width height global_thread_id]

let transpose_point3d =
  [%kernel
    fun (input : point3d vector) (output : point3d vector) (width : int32) (height : int32) ->
      let open Sarek_stdlib.Std in
      do_transpose input output width height global_thread_id]
```

See [Transpose](../../gh-pages/examples/transpose.md) for the host side.

### Recursion

A tail-recursive function becomes a loop. Define it before the `fun`, with annotated parameters and result:

```ocaml
let factorial =
  [%kernel
    let open Sarek_stdlib.Std in
    let rec fact_aux (acc : int32) (n : int32) : int32 =
      if n <= 1l then acc else fact_aux (acc * n) (n - 1l)
    in
    fun (output : int32 vector) (n : int32) ->
      let idx = global_idx_x in
      if idx = 0l then output.(idx) <- fact_aux 1l n]
```

A non-tail-recursive function must carry `pragma ["sarek.inline N"]`, which unrolls it N levels deep; the kernel then works for arguments up to that depth.

```ocaml
let fibonacci =
  [%kernel
    let open Sarek_stdlib.Std in
    let rec fib (n : int32) : int32 =
      pragma
        ["sarek.inline 6"]
        (if n <= 0l then 0l
         else if n = 1l then 1l
         else fib (n - 1l) + fib (n - 2l))
    in
    fun (output : int32 vector) (n : int32) ->
      let idx = global_idx_x in
      if idx = 0l then output.(idx) <- fib n]
```

Without the pragma the PPX rejects the kernel: `function 'fib' is recursive but not tail-recursive`.

## Compile-time checks

The PPX rejects, with a located error, kernels that:

- do not type-check, for example adding an `int32` to a `float32`;
- call `block_barrier ()` under a condition that depends on the thread, which would deadlock the block (`Barrier called in diverged control flow`);
- use non-tail recursion without a `sarek.inline` bound;
- use a C, CUDA or OpenCL reserved word as an identifier (`Sarek_reserved`).

The negative tests in [sarek/tests/negative/](../tests/negative/) pin each of these. They are built only in the `negative` dune profile, where a build failure is the expected result.

Observed gap: the divergent-barrier check did not fire for `if thread_idx_x > 16l then block_barrier ()` written after `let open Sarek_stdlib.Std in` (the kernel compiled), while the same kernel without the `let open` was rejected (checked on `main` at 466f6e36, 2026-10-11). Do not rely on the check in kernels that open `Std`.

## Modules

| Group | Modules |
|---|---|
| Frontend | `Sarek_parse`, `Sarek_typer`, `Sarek_types`, `Sarek_env`, `Sarek_scheme`, `Sarek_ast`, `Sarek_typed_ast` |
| Analysis and rewriting | `Sarek_convergence`, `Sarek_mono`, `Sarek_defunc`, `Sarek_tailrec` (with `_analysis`, `_elim`, `_pragma`, `_bounded`), `Sarek_tag_erasure` |
| IR and code generation | `Sarek_lower_ir`, `Sarek_quote`, `Sarek_quote_ir`, `Sarek_ir_ppx`, `Sarek_native_gen` (with `_base`, `_expr`, `_kernel`, `_helpers`, `_intrinsics`) |
| Infrastructure | `Sarek_ppx` (entry point), `Sarek_ppx_registry`, `Sarek_core_primitives`, `Sarek_error`, `Sarek_reserved`, `Sarek_debug`, `Sarek_matrix_alert`, `Sarek_unsupported` |

The PPX depends on `ppxlib` and the IR types in `spoc/ir`; it has no GPU runtime dependency. Set `SAREK_PPX_DEBUG` to trace its stages.

## Intrinsics from another library

Intrinsics are declared with `let%sarek_intrinsic` (library `sarek.ppx.intrinsic`). The recipe for a library outside this repository, with the dune stanzas both the library and the kernels need, is in [sarek/ppx_intrinsic/README.md](../ppx_intrinsic/README.md). The standard library is in [sarek/Sarek_stdlib/](../Sarek_stdlib/).

## Tests

```bash
# Unit tests of the PPX modules
dune runtest sarek/ppx/test

# End-to-end tests (Native and Interpreter always; GPU backends when present)
dune build @sarek/tests/runtest

# One end-to-end test on every backend
dune exec sarek/tests/e2e/test_histogram.exe -- --benchmark
```

## See also

- [sarek/sarek/README.md](../sarek/README.md): launching kernels
- [sarek/core/README.md](../core/README.md): devices, vectors and transfers
- [Concepts](../../gh-pages/docs/concepts.md): the SIMT model, memory and supersteps
