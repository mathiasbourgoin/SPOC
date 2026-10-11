## 2026-10

### Fixed

- Superstep convergence rule settled (`specs/convergence-semantics.md` §2,
  global review D-38, D-57, C-7). A `let%superstep` reached under
  thread-varying control is rejected with or without `[@divergent]`; the
  attribute is still accepted but no longer has any effect. Before, it skipped
  the check, so `if thread_idx_x > 16l then let%superstep[@divergent] s = ... in ()`
  compiled to a barrier inside a divergent branch. A thread-varying `if`
  inside a superstep body, with no barrier in its branches, stays legal: it
  reconverges before the implicit barrier. The Rocq model
  (`formal/convergence-safety`) implements the same rule; `[@divergent]`
  supersteps are now inside the verified fragment, and the warp checker
  handles varying early returns like the barrier checker (D-56). The
  non-blocking `KNOWN-ISSUE` line of `make test_negative` is gone.

### Added

- Matrix products are rewritten at launch, inside `Execute.run_vectors` (spec
  `specs/matrix-pattern-dynamic.md`). A kernel that is a recognised matrix
  product runs on matrix units when the device has them (HIP gfx11; the
  interpreter only on explicit request, so that it stays the oracle): int32
  by default (bit-identical), float only after
  `accept_relaxed_accumulation` / `accept_operand_rounding`. `SAREK_MATRIX_REWRITE`
  (`off`, `exact`, `auto`, default `auto`) or `Matrix_specialise.set_mode`
  switches it; `Execute.run_vectors_outcome` returns the decision,
  `Execute.run_vectors_plain` launches the kernel as given,
  `Matrix_specialise.set_reporter` / `SAREK_MATRIX_REPORT=1` report it. The
  aligned float16 size class loads fragments straight from global memory
  (`Sarek_matrix_direct`): 256×3072×3072 runs at 39.6–41.9 TFLOP/s on an
  RX 7900 XTX against 34.8–35.2 for the hand-written `Sarek_gemm.Coopmat`
  (three sessions, `bench_matrix_rewrite.exe 3 10 10`, 2026-10-11, spec
  Results). Plans and rewritten kernels
  are cached per IR; a kernel that is not a matrix product costs one memo
  lookup per launch (host launch cost unchanged within run-to-run spread).

- Automatic rewrite of recognised matrix products onto matrix units (spec
  `specs/matrix-pattern-rewrite.md`). `Sarek_matrix.run` (library
  `sarek.matrix`) is a drop-in for `Execute.run_vectors`: when the kernel is a
  GEMM nest whose indices and guards it can bound, the device offers the
  configuration (HIP gfx11, interpreter) and the policy admits it, it launches a
  generated SCoopmat kernel (64×64 tiles staged through shared memory, any size,
  layout, batch, epilogue) over exactly the outputs the original launch writes,
  and otherwise the original; the outcome says which and why. int32 products
  are rewritten by default through 8-bit byte planes and are bit-identical;
  float16-operand products need `accept_relaxed_accumulation`; float32 products
  need `~precision` and the new `Sarek_coopmat.accept_operand_rounding`
  (`Operands_f16`, `Operands_bf16`, `Operands_bf16x3`: a three-term bfloat16
  split, 0.04–0.15 γ_K measured). New: the `BFloat16` cooperative-matrix
  component (bf16 operand loaded from a float32 buffer with rounding; HIP gfx11
  and interpreter), `ECast (TUint8, _)` as a narrowing store, host evaluation of
  uniform expressions (`Sarek_ir_eval_int`). On an RX 7900 XTX the rewrite is
  1.4–2.5× (int32) and 3.4–9× (f16) faster than the naive kernel, and 4–6×
  slower than the hand-written `Sarek_gemm.Coopmat`.

- Epilogue fusion in the launch window (spec `specs/epilogue-fusion.md`). A
  pending launch P and an element-wise consumer C that reads P's output at its
  own index (`let i = global_thread_id in ...`) are merged: C's body runs right
  after each of P's stores, in the thread that made the store.
  - The merge happens only when the host evaluation of both launches' index
    computations (`Index_slice`) shows that P stores every element C writes,
    exactly once. C must also read nothing else P writes, and write nothing P
    accesses except the intermediate.
  - Chains and horizontal groups compose.
  - On by default on HIP only, because its compiler is forced to
    `-ffp-contract=off`. Elsewhere a compiler could contract across the
    merged code, so it is opt-in with `Execute.Window.set_epilogue_backend`.
    `SAREK_LAUNCH_EPILOGUE=0` turns it off.
  - spoc_llm decode, RX 7900 XTX: 335 → 281 launches/token on the 0.8B and
    749 → 665 on the 35B, logits bit-identical.

- Launch window with horizontal fusion (spec `specs/launch-window.md`).
  `Execute.run`/`run_vectors` JIT launches without `~stream` are deferred on
  HIP, OpenCL and Vulkan up to the next synchronisation point:
  - every Transfer read-back, upload into a buffer a pending launch binds,
    copy, staging, free, flush, stream and event operation;
  - `Device.synchronize`, `Profiling` events and timers;
  - raw `Kernel` launches, `run_source`, stream launches and `Soa_launch`;
  - device destroy and cache clear.

  Pending launches that bind no common buffer with a write, and have the same
  block shape and grid y/z, are issued as one kernel. Each work-group runs one
  member's unchanged body, with `block_idx_x`, `grid_dim_x` and the global
  index intrinsics rewritten to the member's own values. The result is
  bit-identical to the unfused launches by construction, on every backend.
  Fused kernels are cached by member sequence and reused across launches.

  Errors move only at device compilation and driver launch time: checks and
  code generation still raise in the launching call, but a failure at those
  two stages is raised by the call that flushed the window. If a fused kernel
  fails, its members are issued one by one.

  Controls: `Execute.Window.set_enabled`, `set_backend_enabled`, `flush` and
  `stats`; `SAREK_LAUNCH_WINDOW=0` turns the window off. These launches stay
  immediate:
  - CUDA, Metal and WebGPU (each can opt in);
  - CPU devices, whose zero-copy buffers alias;
  - Native and Interpreter.

  A kernel is never fused if it reads `grid_dim_x`, uses intrinsics from
  outside `Sarek_stdlib`, or declares shared memory of non-constant size.

  `bench_launch_window` on an RX 7900 XTX runs a decode-like chain of 24
  layers × 18 launches. Window off vs on, 5 interleaved repetitions, median
  µs/token: 4173 → 3158, with 432 → 288 launches per token, and results
  bit-identical. Tested on HIP (RX 7900 XTX, 7950X iGPU), OpenCL (rusticl,
  both GPUs), Vulkan (RADV), Native and Interpreter.
- Matrix-product (GEMM) recognition and a compile-time advice (spec
  `specs/matrix-pattern-advice.md`). `Sarek_matrix_pattern.analyze` finds, in
  a kernel's IR, every loop that accumulates `A[ia] * B[ib]` (or `fma`) over a
  reduction index and stores the sum once to `C[ic]`, with affine indices
  (`Sarek_ir_affine`) that separate into row, column and batch indices:
  row- and column-major operands, 1-D (`tid / n`, `tid mod n`) and 2-D thread
  mappings, loop nests, batches, padded leading dimensions, offsets, an
  accumulator starting from C, and `alpha·acc + beta·C` epilogues; every other
  reduction loop (overlapping or non-unit-stride operands, dependent indices,
  non-affine or gathered indices, ...) is reported with the reason it is not
  one.
  `Sarek_matrix_advice` names the matrix-unit configuration that would apply
  (f16 operands: `f16×f16+f32→f32`, bounded, needs
  `accept_relaxed_accumulation`; float32: needs operand rounding, no opt-in yet;
  int32: exact through 8-bit byte planes) and the backends that run it. The
  PPX reports each such nest as the OCaml alert `sarek_matrix` located at its
  `for` loop: non-fatal, silenced with `[@alert "-sarek_matrix"]` on the kernel
  or `[@@alert "-sarek_matrix"]` on its binding (or per file/library, or made an
  error with `-alert @sarek_matrix`). `SAREK_MATRIX_SCAN=<file>` makes the PPX
  log every finding (fresh build directory, `DUNE_CACHE=disabled`). Nothing is rewritten. `Sarek_fusion.should_fuse` now gives
  a matrix-product operand its own reason. `Sarek_ir_conv` moved from
  `sarek.transpile` to `sarek.frontend` (the PPX converts its IR for the
  analysis). Scanned over the 254 kernels the repository builds (outside the new corpus): 4 advised,
  all matrix products; no false positive.

- `Transfer.copy_bytes_to_host ?stream ~src ~src_byte_offset dst`: the
  byte-level read-back, mirror of `copy_bytes_to_device`. Copies
  `Bigarray.Array1.size_in_bytes dst` bytes of a vector of any scalar element
  kind, from byte `src_byte_offset`, into a C-layout Bigarray of any element
  kind (e.g. a range of an `int32` device-only vector into a `char` slot of a
  host pool). Same stream, keep-alive and location contract as
  `copy_to_host`, with which it shares one implementation (`d2h_range`); the
  source's location is unchanged. With `~stream`, the destination's contents
  are undefined until the stream or a later event has completed, and
  `free_pinned`/`unregister_host` refuse it meanwhile. Refused with
  `Feature_not_supported` on OpenCL, Vulkan and WebGPU. Tested on Native,
  Interpreter, HIP (RX 7900 XTX and the 7950X iGPU) and CUDA through ZLUDA;
  Metal shares the `copy_to_host` path and was not run. Spec
  `specs/copy-bytes-to-host.md`.

- Cooperative matrix (`SCoopmat`) on HIP through the RDNA3 matrix units, and
  an explicit opt-in for float accumulation (spec `specs/coopmat-hip-wmma.md`).
  On gfx11 devices with 32-wide wavefronts the CUDA-C generator (shared by HIP)
  lowers 16×16×16 `f16 × f16 + f32 → f32` to
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` and the eight `u8/s8 × u8/s8 +
  s32/u32` configurations to `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`; the
  HIP backend now reports these through `capabilities.coopmat`, and any other
  HIP device (the gfx1036 iGPU, gfx10, gfx12, CDNA, wave64) is reported with no
  configuration, so the launch gate refuses with the HIP rule in the message.
  Float accumulation is bounded, not bit-identical (Regime B of
  `docs/design/f16-relaxed-accuracy.md`): the launch gate and the interpreter
  refuse it until the program calls
  `Sarek_coopmat.accept_relaxed_accumulation ~reason`; with it, the interpreter
  evaluates the f16 configuration in a fixed sequential binary32 order.
  `Sarek_gemm.Coopmat` is a tiled f16 GEMM built from these statements
  (`d = a·b + c`, row-major). Measured on an RX 7900 XTX (ROCm 7.2.4): the
  integer configurations are bit-identical to the interpreter over all 65536
  8-bit operand pairs; the f16 GEMM is within γ_K·(Σ|a·b| + |c|) of the exact
  result at 64×64×128 and 256×256×3072 (worst 0.044 of the bound), repeatable,
  and identical across tilings; at M = 256, K = 3072, N = 1024/3072/12288 it
  runs at 20–24, 36–43 and 51–53 TFLOP/s (best tiling, median of two sessions)
  against 3.2–3.3 for the scalar f32 `Sarek_gemm`, 10–16 for rocBLAS with f32
  output and 26–84 for rocBLAS with f16 output
  (`sarek-hip/test/bench_hip_coopmat_gemm.exe`,
  `tools/probes/rocblas_gemm_f16.cpp`). The CUDA arm (`nvcuda::wmma`) is
  compile-checked only — NVRTC, and HMMA/IMMA in sm_80 SASS — and stays
  refused at launch because the CUDA backend does not probe `coopmat`. A
  cooperative-matrix launch whose block is not a whole number of 32-wide waves
  is refused. Metal, OpenCL, WGSL and PTX still refuse `SCoopmat`.
  `Sarek_coopmat.device_support` gains `ds_evidence`; `TUint8` is
  `unsigned char` in CUDA-C.

- `Transfer.stage_to_device ?stream src ~dst ~dst_offset`: the cheap way to
  update a small vector that kernels read (at most
  `Transfer.max_staged_bytes` = 64 KiB per call). The source is copied into a
  1 MiB per-device pinned staging ring before the call returns, so the caller
  may reuse it at once, and the copy is enqueued on the stream (default: the
  device's default stream) without waiting for the GPU. `Vector.set` on such a
  vector downloads it after every launch and re-uploads it synchronously at
  the next one, waiting for the GPU queue both times; its documentation now
  says so. Measured with `sarek-hip/test/bench_hip_small_writes.exe` (RX 7900
  XTX, 8 int32 + 8 float32 rewritten, 300 us host work, 4 launches per
  iteration, three runs): 709–712 us per iteration with `Vector.set`,
  333–343 with `stage_to_device`, 320–324 without any rewrite. Native,
  Interpreter and Metal perform a synchronous ranged copy; OpenCL, Vulkan and
  WebGPU refuse. Spec `specs/transfer-pitfalls.md`.

- `Transfer.register_host` documents measured ROCm pitfalls: copies right
  after registering are intermittently slow (7 of 50 first passes at
  0.95–6.17 GB/s, about 26 otherwise, whether the memory was touched before
  registering or not; `bench_hip_register_host.exe`), a copy may not span two
  registrations (`hipErrorInvalidValue`), and ROCm does not count registered
  pages as pinned (`VmPin: 0 kB`).

- Metal implements device-only vectors, ranged copies, pinned memory, streams
  and events (spec `specs/metal-ranged-copies.md`). Buffers stay
  shared-storage MTLBuffers (Apple silicon unified memory), so a ranged copy is
  one bounds-checked `memcpy` at `contents + offset`; `alloc_pinned` is
  page-aligned host memory and `register_host` a no-op (nothing to page-lock
  for the GPU). Buffers above the device's `maxBufferLength` are refused with
  an error naming the limit, and `capabilities.total_global_mem` reports
  `recommendedMaxWorkingSetSize` instead of a 4 GiB constant. Measured on an
  Apple M4 (macOS 15.6.1), 256 MiB ranged copies, best of 5: 42.6 GB/s
  host->device and 42.5–42.7 GB/s device->host, pageable and page-aligned alike
  (`sarek-metal/test/bench_metal_copy.exe`, information only).

- `Int32.float_of_bits : int32 -> float32` and
  `Int32.bits_of_float : float32 -> int32` in the Sarek stdlib: bit
  reinterpretation with OCaml's `Stdlib.Int32` names and semantics, lowered on
  every backend — CUDA C/HIP `__int_as_float`/`__float_as_int`, OpenCL
  `as_float`/`as_int`, Metal `as_type<float>`/`as_type<int>`, GLSL
  `intBitsToFloat`/`floatBitsToInt`, WGSL `bitcast<f32>`/`bitcast<i32>`, PTX
  `mov.b32`; Native and Interpreter run OCaml's own functions (a signalling
  NaN is quieted there, as in OCaml, because float32 is carried as a double).
  The GLSL, WGSL and PTX lowerings only apply to the stdlib `Int32` path, so
  another module's `float_of_bits` is not captured. The C-family backends now print
  the literal `Int32.min_int` as `(-2147483647 - 1)` and WGSL as
  `(-2147483647i - 1i)`: `-2147483648` is a `long` in C/OpenCL/MSL, which
  `as_float` rejects, and `-2147483648i` does not fit `i32`. Covered by
  `sarek/tests/e2e/test_int32_bitcasts.ml` over ±0, ±inf, denormals, max
  finite, a NaN payload and a signalling NaN: decode and encode (NaN rows
  checked for NaN-ness, since the host widens float32 to a double), a round
  trip inside one kernel and one through a float32 device buffer across two
  launches (bit-exact, signalling NaN included, on every GPU), a packed-record
  decode, and a `0x80000000l` literal operand. It passed 2026-10-10 on HIP, OpenCL,
  Vulkan and CUDA/PTX via ZLUDA (RX 7900 XTX and Ryzen 7950X iGPU), on Metal
  and OpenCL (Apple M4), and on Native and Interpreter; and by
  `sarek/tests/unit/test_int32_bitcasts_codegen.ml` (spellings, glslang/naga/
  clang validation with negative controls, refusals). Spec
  `specs/stdlib-bitcasts.md`.

### Changed

- `lsr` compiles to one unsigned shift on every backend. It is now its own IR
  operator (`Lshr32`/`Lshr64`) instead of a 4–5 operation expansion over the
  arithmetic shift: CUDA/HIP emits `((int)((unsigned int)(a) >> (b)))`,
  OpenCL and Metal `((int)((uint)(a) >> (b)))` (64-bit: `unsigned long long`
  / `ulong`), GLSL `int(uint(a) >> uint(b))`, WGSL
  `bitcast<i32>(bitcast<u32>(a) >> u32(b))`, PTX `shr.b32`/`shr.b64`. Results
  are unchanged for shift amounts 0..width-1 (others stay unspecified, as in
  OCaml), and an `lsr` operand no longer has to be a variable or a constant.
  spoc_llm's q5_K matvec written with `lsr`, on an RX 7900 XTX: 31.6 -> 23.9 us
  (hot, median of 3) and 36.3 -> 31.1 us (cold). On gfx1100 the old expansion
  kept clang from unrolling the inner loop (193 VALU, 2 `global_load_b128`);
  with the native shift clang unrolls it (274 VALU, 12 `b128`), the same code
  as the hand-written `asr`-and-mask form. Spec `specs/codegen-lsr-unroll.md`.

- `for` loops with constant bounds and a trip count of at most 16 get an
  unroll hint by default (`#pragma unroll` in CUDA/HIP, OpenCL and Metal,
  `[[unroll]]` with `GL_EXT_control_flow_attributes` in GLSL, nothing in WGSL
  and PTX), as long as the nest they head expands to at most 64 copies. Results
  are unchanged. An explicit `pragma` on the loop takes precedence
  (`pragma ["unroll 1"]` opts out) and `SAREK_AUTO_UNROLL=0`, set before a
  kernel's first launch, disables the hint.
  On spoc_llm's current matvec kernels the hint changes nothing on gfx1100:
  clang -O3 already unrolls those loops once `lsr` is a native shift.

- Metal kernel launches are asynchronous. Dispatches are appended to one open
  command buffer per queue. At each launch it is committed if the queue's
  previous command buffer has completed or once it holds `SPOC_METAL_BATCH`
  dispatches (default 32); otherwise the next launch, wait, event record or
  synchronisation commits it. Every
  host access to a buffer waits for the pending command buffers that bind it,
  and a launch waits for pending work on other queues that binds its buffers.
  Before, every launch committed its own command buffer and waited for it. A
  failed command buffer now raises `kernel_launch_failed` (it was ignored), at
  the first wait that observes it, naming every kernel the command buffer ran.
  `Event.elapsed` measures GPU end times (it subtracted host record times, and
  a fresh start event gave about the epoch).
  Measured on an Apple M4: 195 us per launch before, 4.3 us (3 arguments) /
  8.0 us (10 arguments) after (`bench_metal_launch_overhead.exe`, 10000
  launches); spoc_llm Qwen3.5-0.8B Q8_0 greedy decode 8.6 tok/s before, 67–70
  tok/s after, same tokens.

- Intrinsic libraries outside sarek are a supported extension point
  (`sarek/ppx_intrinsic/README.md`, "Using intrinsics from another library"):
  declare the library `(kind ppx_rewriter)` and list it in
  `(pps sarek.ppx <lib>)`. Covered end to end by `sarek/tests/downstream/`,
  which runs on Native and Interpreter, and on HIP, CUDA C, OpenCL and Metal
  when those devices are present. The PTX, GLSL and WGSL backends do not
  consult the intrinsic registry. They refuse such kernels with an error that
  names the intrinsic.
- `let%sarek_intrinsic` also registers a host implementation
  (`Sarek_registry.register_host_impl`) for every all-scalar signature. The
  Interpreter falls back to it, so it now runs intrinsics declared outside its
  built-in tables. Before, it raised `Unknown intrinsic`.
- Device-only vectors, ranged and stream-ordered copies, pinned host memory,
  kernel launches on a stream, and events (`Spoc_core.Transfer`,
  `Execute.run`/`run_vectors ?stream`; spec
  `specs/device-buffers-async-copy.md`). `create_device_only` allocates a
  vector with no host mirror; every host access to it refuses instead of
  allocating. `copy_to_device`, `copy_bytes_to_device`, `copy_ptr_to_device`
  and `copy_to_host` copy a range at an offset, synchronously or on a
  stream, and SPOC keeps the operands alive until the stream or a later
  event completes. `alloc_pinned`/`free_pinned` and
  `register_host`/`unregister_host` give page-locked Bigarrays. Implemented
  on HIP and CUDA; synchronous memmove semantics on Native and Interpreter;
  refused with `Feature_not_supported` on OpenCL, Vulkan, Metal and WebGPU.
  Measured on an RX 7900 XTX (ROCm 7.2.4), 256 MiB host->device into a
  device-only vector, best of 5: pageable 28.1 GB/s, pinned 28.6 GB/s
  synchronous; 27.5 / 28.3 GB/s on a stream
  (`sarek-hip/test/bench_hip_pinned_copy.exe`, information only).

### Changed

- Repeated launches of the same kernel are cheaper (spec
  `specs/cheap-repeated-launches.md`). `Execute.run`/`run_vectors` memoise,
  per physical kernel IR, the device capability check and the generated
  source (keyed by backend, block and SoA parameters); failures are never
  memoised and the argument checks still run on every launch.
  `Compile_cache.make_key` memoises keys per physical source string. The HIP
  launch marshals all arguments into one buffer instead of one allocation
  per argument. New `Execute.clear_launch_cache` drops the memo; backend
  cache clears and device teardown call it. Host cost per `run_vectors`
  launch on an RX 7900 XTX (`sarek-hip/test/bench_launch_overhead.exe`,
  10 000 launches, no sync, tartarus, 2026-10-10, base beb81321 vs this
  change): 3 arguments 11.4 -> 4.4 us, 10 arguments 17.4 -> 6.5 us. Native
  is unchanged (its launch cost is execution).

- `let%sarek_intrinsic` registers an intrinsic under its library's module path,
  using the `library-name` cookie dune passes to the PPX, instead of guessing
  from the source directory name. sarek's own libraries keep their paths. A
  downstream library in a directory not named after it (for example
  `src/intrinsics/` holding `foo_intrinsics`) used to register as `["Bits"]`
  rather than `["Foo_intrinsics"; "Bits"]`.
- **Breaking:** a qualified name in a kernel (`M.f`) that the kernel PPX
  cannot resolve is now a compile error: "Unknown kernel function or
  constant M.f ...". The error names the fix and lists the names the PPX knows
  under `M`. Before, the PPX accepted it untyped. Native ran it as an OCaml
  call, but every source backend printed it verbatim into device code, so it
  failed only at JIT time. Three in-tree sites depended on this:
  `test_visibility_private` now links `sarek.visibility` into its PPX, and
  `test_debug_native` and the `reduction_max` benchmark kernel now spell their
  values in Sarek.
- **Breaking:** `Framework_sig.BACKEND` gains `Async` (an `ASYNC` module), `wrap_stream`,
  `unwrap_stream`, `wrap_event`, `unwrap_event` and `run_source_on_stream`;
  `PLUGIN_BASE` gains `Async`. An out-of-tree backend that does not provide
  them adds `module Async = Framework_sig.Async_refused (...)`,
  `include Framework_sig.Make_refs (...)` and a `run_source_on_stream` that
  calls `Framework_sig.refuse_stream_launch` (about 20 lines; see
  `sarek-opencl/Opencl_plugin.ml`).
- **Breaking:** `Vector.host_storage` has a third constructor, `Device_only_storage`. Code
  that matches on `vec.host` must handle it (every in-tree site refuses).
- **Breaking:** `Transfer.STREAM` gains `stream_ref` and `state`; `destroy_stream` now
  synchronises the stream before destroying it.
- **Breaking:** `Execute` refuses a vector longer than `Int32.max_int` elements at launch;
  the injected length used to wrap silently.

### Fixed

- Vector ids are allocated with `Atomic.fetch_and_add`. The plain `ref` +
  `incr` handed out duplicate ids when vectors were created from several
  domains (4 domains x 200 000 creations: 450 907 duplicates of 800 000), and
  the id is buffer identity for launch-window conflicts, epilogue fusion and
  the transfer keep-alive ledger. Test: `test_vector_ids`.
- `Memory.host_to_device` / `device_to_host` size the copy from the bigarray's
  own bytes and raise `Invalid_argument` when its element size differs from the
  buffer's or it holds more bytes than the buffer (a float32 bigarray used to
  type-check against a float64 buffer and over-read / overflow the host
  array). `Memory.device_to_device` refuses a source larger than the
  destination. D2D still bounces through host memory: the `BUFFER` signature
  has no backend device-to-device copy. Test: `test_memory_sizing`.
- hiprtc and nvrtc option arrays: `CArray.of_list string` let the C copies of
  the option strings be freed before the compile call. Under a different
  allocation pattern hiprtc read a garbage option ("error reading '8{...'"),
  and `test_hip_f16` failed deterministically with the launch window on. The
  strings are now explicit char arrays, kept alive until the call returns.

- Ranged-copy range checks no longer overflow: `copy_to_device`,
  `copy_bytes_to_device`, `copy_ptr_to_device`, `copy_to_host`,
  `copy_bytes_to_host` and `stage_to_device` compared `offset + size` with the
  total, which wraps for an offset near `max_int` and let the copy through
  (Native segfaulted reading `max_int - 2` bytes past the buffer). They now
  compare `offset > total - size`. The synchronous paths also keep the vector
  alive across the backend call, not only the Bigarray, so its buffer cannot
  be finalised mid-copy.
- CUDA: NVRTC is found on a CUDA 13 install that has only the soname
  `libnvrtc.so.13` (no development symlink). The loader tries
  `Cuda_nvrtc.nvrtc_candidates` in order: `libnvrtc.so`, then `.so.13`, `.so.12`,
  `.so.11`, then the macOS and Windows names (`nvrtc64_130_0.dll` added). Test
  `sarek-cuda/test/test_cuda_nvrtc_candidates.ml`. Trade-off: a host with a
  CUDA 12 driver, both `.so.13` and `.so.12`, and no `libnvrtc.so` symlink now
  gets NVRTC 13, whose PTX the 12.x driver rejects at module load and which has
  no compute_50..72 targets; install the development symlink pointing at the
  toolkit that matches the driver.
- `docs/optimization/l15b-mma-probe.md` no longer says ZLUDA does not execute
  `mma.sync`: on ZLUDA v7-preview.3 (RX 7900 XTX) m16n8k16 f32.f16 runs as a
  dot-product emulation (inexact, within the `γ_K` bound) and m16n8k32 s8 runs
  bit-exact; u8, satfinite, m16n8k8, f16 accumulate and every `wmma` form are not
  translated. `test_ptx_mma_probe` now checks each form separately against a CPU
  reference, with a control kernel, and declares the PTX ISA version the
  device's SM needs. Its statuses are ZLUDA's, not NVIDIA's.
- HIP: freeing a device buffer no longer leaves its device selected. The free
  runs from GC finalisers, so with two HIP devices a finaliser could fire
  inside a launch on the other device and make it fail with
  `hipErrorInvalidDevice`. Regression test
  `sarek-hip/test/test_hip_free_keeps_current.ml`.

- HIP `Stream.synchronize` now selects the stream's device first, as the CUDA
  backend has since backlog-225: synchronising `default_stream d2` while d1
  was current drained d1 and released d2's kernel-argument keep-alives.

- The Native and Interpreter CPU backends are now installed, as
  `sarek.native` and `sarek.interpreter`. They had no `public_name`, so code
  outside this repository could not link them, although the `sarek` package
  description and both plugin READMEs already advertised them. A cram test
  (`sarek/plugins/test/install_consumer.t`) builds an out-of-tree project
  whose OCAMLPATH is the install tree and links both.
- `benchmarks/descriptions/generated/reduction_max_generated.md`: all four
  device sources contained `Float.neg_infinity`, which is not valid device
  code. They now contain `-FLT_MAX`.

## 2026-07

### Added

- Direct PTX emitter completion: records/variants with aligned C-ABI layout,
  match, helper-function inlining, static + dynamic `.shared`, `.local`
  arrays, full atomic set (CAS, wrapping inc/dec, 64-bit add/exch), f64
  softmath (trig, exp/log family, hypot, …), scalar cast matrix
- `Sarek_worklist` — portable dynamic-parallelism (work-queue) library
- `Sarek_gemm` — portable tiled SGEMM library for shared-memory backends
- `Sarek_df64` — float-float extended-precision arithmetic in pure Sarek
- Portable float64: `G`-suffix float64 literals, `sarek.real64` host
  plumbing, complete interpreter float64 math oracle
- Tier-0 defunctionalization pass (first-class kernel-local functions)
- Tuple-typed vectors on device and Native paths (shared shape registry)
- Static tag erasure for kernel-local variant slots
- Aligned C-ABI aggregate host layout (L8), with the `PtxLayout.v` formal
  model restated for the aligned ABI (0 admits)
- float32 `fma` intrinsic on all backends; GLSL `precise` qualifier on
  float locals
- CI installs `cuda-nvdisasm-12-6` and `ci/assert-toolchain.sh` asserts
  `nvdisasm` (version + a ptxas→nvdisasm probe). It was absent from the built
  image, so `test_cuda_f16_sass` — the gate the NVIDIA f16 guarantee rests on,
  and the only one that could surface a 12.6-vs-13.3 ptxas divergence —
  self-skipped in CI while reporting green.
- `docs/fp-contraction-policy.md` — cross-backend floating-point contraction
  policy: what each backend may contract, what actually prevents it, and
  whether that mechanism is verified or merely believed. The interpreter is
  named as the cross-backend oracle. Linked from every site that previously
  carried an ad-hoc contraction comment. §7 ("what cannot be verified without
  NVIDIA hardware") is now partly closed: an f16 kernel HAS been executed on
  an NVIDIA GPU — GTX 1070 Max-Q, sm_61, CUDA 12.9, driver 580.119.02 —
  agreeing with the interpreter on all 63488 finite binary16 inputs, with a
  liveness control (63085 mismatches on deliberately mismatched kernels)
  proving the sweep can go red. That also settles the tie-rounding question
  the section named, exercises the *driver's* ptxas rather than the offline
  one, adds a second toolkit version (12.9, between CI's 12.6 and the
  document's 13.3), and independently confirms on hardware that the CUDA
  barrier removed in #110 was inert. sm_61 is below the sm_75…sm_121 range
  the host-side sweeps cover, so it is a new sample, not a repeat.
- FP-conformance guard on the CUDA/nvrtc path: `-use_fast_math`, `-ftz=true`,
  `--prec-div=false` and `--prec-sqrt=false` are now REJECTED at the point an
  option array reaches `nvrtcCompileProgram` (`--fmad=true` warns). The guard
  screens the whole ARRAY, because nvrtc accepts an option and its value as two
  elements: `["--ftz"; "true"]` compiled a subnormal-flushing kernel past a
  per-element check (confirmed against libnvrtc 13.3, `.ftz` in the emitted
  PTX). A value-taking name consumes the next element and is fail-closed. They flush
  binary32 subnormals or downgrade div/sqrt, and no later flag undoes that.
  `sarek-cuda/test/test_cuda_fp_conformance.ml` reproduces the hazard
  host-side (`-ftz=true` turns `FMUL`/`FADD` into `FMUL.FTZ`/`FADD.FTZ` at
  sm_90, CUDA 13.3) and each of its four cases was proved red by mutation.
- `sarek/tests/e2e/test_vulkan_no_contraction.ml` (on the `e2e-gpu` alias, so
  it executes rather than merely building) — settles whether a Vulkan driver
  honours SPIR-V `NoContraction` (#126). Same shader compiled twice,
  differing only by `precise`, run on the same device/driver/process; the
  contracted target is taken from the device's own `fma()` rather than an IEEE
  model, because RADV's `fma` is not correctly rounded and a modelled target
  can make a genuinely contracted result read as clean. **Measured on RX 7900
  XTX (RADV NAVI31) and the Raphael iGPU, Mesa 26.1.4-arch3.1: 0 of 7
  contraction shapes contracted with or without `precise`, and the emitted RDNA
  ISA is opcode-identical between the two builds.** Both in-tree claims are
  refuted: RADV neither ignores the decoration nor needs it. Mesa ANV remains
  unmeasured — no Intel GPU on this machine.
- `sarek-hip/test/test_hip_f16_shapes.ml` + `scripts/f16_shape_isa_audit.sh` —
  exhaustive f16 expression-shape audit for AMDGPU fusion demotion (#106). All
  20 shapes the DSL can emit, each swept over all 63488 finite binary16 inputs,
  on gfx1100 and gfx1036: **0 disagreements as shipped**. Removing the opacity
  barrier breaks 9 of 20 and reproduces the original 620 exactly — the harness
  was calibrated against that known positive before its nulls were trusted, and
  it fails closed if the barrier-removed control stops going red. Disassembly
  additionally shows *four* demotion opcodes, not two (`v_mul_f16` and
  `v_sub_f16` are new), and three shapes that are demoted in machine code yet
  numerically clean — which a numeric-only audit would have mis-reported as
  unaffected. See `docs/optimization/amdgpu-f16-fusion-shape-audit.md`.
- ZLUDA/AMD support for the CUDA/PTX backend (PTX launch ABI fix)
- `SAREK_REQUIRE_PTX` for `test_soa_emitter_equiv`: set it to exactly `1` and
  the test fails when no enumerated device reports the `CUDA/PTX` framework,
  instead of exiting 0 with the SoA legs either skipped or reported as PASSED
  with a `(SoA skipped: non-PTX)` note. Nothing in the repository's dune
  rules, Makefile targets or CI jobs sets it, and unset (or any value other
  than `1`) nothing changes, so an absent device still reads as skipped
  rather than as a pass or a failure. It asserts enumeration only — whether
  the SoA cases pass on that device is what the per-case rows say.
  Opt-in-by-hand; see `gh-pages/docs/device_selection.md`.
- **KNOWN RESIDUAL (backlog-225, not fixed).** `test_soa_emitter_equiv`'s
  `check_roundtrip` and the `Vector.fill` row of `check_relaunch` have each
  been seen, once, to return `y` equal to the un-doubled host value on
  ZLUDA's CUDA/PTX view of this host's devices — a leaf write that appears
  not to have landed before read-back. 2 occurrences noticed over "some
  2,500" executions of the test binary; a separate non-reproduction campaign
  afterward ran 2,443 further executions (2,000 + 300 + 40 concurrent + 40 on
  a cold ZLUDA cache + 3 under `dune runtest --force` + 60 isolated), all
  zero, which does not reconcile arithmetically against the ~2,500 figure —
  whether the 2 occurrences fall inside or outside that 2,443 cannot be
  determined from the record. No root cause established; nothing here
  claims a fix. `Transfer.flush` was added on the `check_roundtrip` leg as a
  defensive measure, not offered as the fix — the `check_relaunch` occurrence
  was on a leg that already synchronizes. Also found and filed but NOT
  implicated as the cause (both failing legs already call
  `Device.set_current` correctly via `Device.synchronize`): `Cuda_api.ml`'s
  `Stream.synchronize` was the only function in its module omitting
  `Device.set_current` before touching a stream/device (fixed here), and
  `Cuda_plugin_base.ml`'s `current_device` is a plain global ref while
  `Cuda_api.ml` underneath it deliberately uses a `Domain.DLS` key for the
  same state (not fixed; filed). See the full comment at
  `sarek/tests/e2e/test_soa_emitter_equiv.ml:703` and
  `docs/optimization/tier1b-emitter-soa-handoff.md`'s backlog-225 status
  block.
- T3-SEMANTIC milestone lock for both formal projects; conformance +
  mutation tests wired into `dune runtest`

### Changed

- **BREAKING (`Sarek_ir_codegen`, shipped in `spoc.opam`'s `spoc.ir` library).**
  The record-declaration ordering surface added for backlog-203 is replaced by
  one that also orders variants (see the backlog-211 entry under Fixed).
  Removed: `sort_record_types_by_dependency`, `referenced_record_names`,
  `gen_record_typedefs`. Renamed: `Record_type_cycle` → `Type_decl_cycle`, which
  now reports a cycle with either kind on it. Added: `type_decl`, `tie_break`,
  `sort_type_decls_by_dependency`, `referenced_type_names`, `gen_type_decls`,
  `gen_c_type_decls`. No compatibility aliases, deliberately: an alias for
  `sort_record_types_by_dependency` would still compile inside a per-kind
  emission loop and silently reintroduce the cross-kind bug. The three C-family
  backends also lose their local `gen_variant_def` wrappers —
  `gen_c_type_decls` takes `~constructor_prefix` directly. Nothing outside
  `spoc/ir` and its tests referenced any of the removed names.
- Breaking: `Sarek_type_helpers.HELPERS` gained a required `val field_names :
  string list` — the immediate field names in `to_values` order. An in-place
  record-field store needs a field's POSITION in the `VRecord`'s value array,
  and nothing in the previous signature could supply one: `get_field` only
  reads, and `to_values`/`from_value` round-trip through the OCaml record and
  therefore copy. For a PPX-generated RECORD helper the correspondence holds by
  construction: `field_names` comes from the same label list that generates
  `to_values` and `get_field`, so the three cannot drift. Two carve-outs, because
  the justification for a breaking change should not be stated more widely than
  it holds: the PPX emits `field_names = []` for a VARIANT (no named fields, so
  every lookup returns `None` and a field store on one is refused by name rather
  than writing an arbitrary slot), and one in-tree implementor is *not* PPX-
  generated — the hand-written test mock in
  `sarek/sarek/test/test_sarek_type_helpers.ml`, which maintains the order
  manually and whose own test checks `field_index` against `to_values`' output
  rather than against hardcoded slots so that a drift there fails. A hand-written
  implementation outside this repository has to add the list, and carries the
  ordering obligation itself. There is no non-breaking spelling — a module type
  cannot carry a default, and the index is not derivable from the rest of the
  signature.

- **BREAKING (backlog-185/200): all five source-emitting backends now REFUSE a
  `[%native]` block instead of emitting for it.** Before this change CUDA,
  OpenCL and Metal raised on `SNative` unless a prior transpile had left a tag
  in `current_framework` — `Sarek_transpile.set_framework` wrote that
  module-level ref for CUDA/OpenCL/Metal/WGSL, so after any earlier transpile a
  later direct `generate` on a `[%native]` kernel injected with the leaked tag
  instead of raising. GLSL and WGSL silently emitted
  `/* native code not supported in <lang> */` and continued, producing a
  shader missing the operation the kernel asked for with no diagnostic. GLSL
  and WGSL now raise the same shared `Sarek_ir_codegen.native_block_refusal`
  the other three do, naming what the caller should do instead (express the
  operation in Sarek, or route it through PTX, which is the one backend that
  still serves `[%native]` directly). Along the way, the per-generation facts
  each backend needs while emitting — the kernel's variant table on every
  backend, plus GLSL's helper names, f64-softmath state and vec-param table,
  and WGSL's scalar-param names — moved from module-level `ref`s (a
  module-level `Hashtbl` in the case of GLSL's vec-param table), shared and
  clobbered across generations including across domains, to a value threaded
  through the emit calls. The
  `current_framework` tag was not threaded; it was removed outright and
  replaced by a per-backend compile-time constant, so there is no longer a
  module-level slot for it to leak through. The sequential half of this — a
  later generation on the same domain reading an earlier one's leftover state —
  is pinned by `framework_tag_is_not_module_state`; the cross-domain half, by
  `generations_do_not_interleave`.

- The CUDA branch of `sarek_f32_barrier` no longer emits
  `asm volatile("" : "+f"(x))`. At an f16 narrowing it contributes zero PTX
  instructions, so `ptxas` receives an identical instruction stream and the
  cubins are byte-identical with and without it — re-measured on CUDA 13.3 for
  sm_75 through sm_121. What keeps the f32 multiply out of the narrowing on
  NVIDIA is `ptxas`, machine-checked by `test_cuda_f16_sass`. NOTE the same
  barrier is *not* inert at a `mul`→`add` site (PTX `mul.f32`+`add.f32` instead
  of `fma.rn.f32`), but `ptxas -O1`+ re-contracts that under the default
  `-fmad=true`, so it is still not a usable contraction barrier on NVIDIA — use
  `Sarek_df64`'s `mul_rn`. The AMDGPU `"+v"` barrier, which IS load-bearing, is
  unchanged. Removing a no-op that read as protection; no behaviour change.

### Fixed

- A shared-memory array of a record type behaved four different ways across
  nine devices, none of them right. `let%shared (s : tri) = 4l` followed by
  `s.(i).a <- e`, measured on this host: the Interpreter ×2 raised
  `assignment target of .a (got unit)`; Native accepted the store and wrote
  EVERY slot (`7 7 7 7`, want `7 0 0 0`); CUDA/PTX ×2 raised
  `unsupported construct: btype of custom type`, naming neither the array nor
  the type; and OpenCL ×2 and Vulkan ×2 failed inside the DEVICE compiler,
  OpenCL with `unknown type name 'Test_..._tri'` and Vulkan with a glslang
  syntax error at the shared declaration. Three separate causes. (1) Native's
  `alloc_shared_with_key` filled the array with `Array.make size default`, so
  every slot of a boxed element type was the same allocation — it now takes a
  per-slot thunk and calls `Array.init`, and the identical `Array.make` in the
  `create_array n Local` path is fixed with it. (2) The interpreter mapped a
  record element type to `VUnit`, so there was nothing to store into — it now
  builds a zeroed `VRecord` — or, for a variant, a `VVariant` carrying the
  first NULLARY constructor, the same one Native's `default_value_for_type`
  picks, falling back to the first constructor with zeroed payloads where
  there is no nullary one (a case in which Native has no default at all and
  raises, so the interpreter is more defined there rather than agreeing) — per slot in
  `Sarek_ir_interp_value.alloc_kernel_array`. The tag is
  `Hashtbl.hash ctor mod 256`, the encoding `EVariant` and the two matchers
  already use and which is now the single `variant_tag_of_ctor`; a positional
  index there produced a `VVariant` no arm could match, so reading a default
  variant slot raised `Pattern match failure in SMatch` on the Interpreter
  while Native answered the nullary constructor. That replaces three identical
  copies of the old init table in `Sarek_ir_interp_eval` plus a fourth, NARROWER
  one on `Sarek_ir_interp`'s `DShared` kernel-parameter path, which mapped only
  `TInt32` and `TFloat32` and everything else to `VUnit`. Unifying them
  therefore also widens that path: `TInt64`, `TFloat64`, `TBool`, `TFloat16` and
  `TUint8` go from `VUnit` to their typed zeros there. That is a fix, but it is
  a behaviour change outside backlog-206's shape and it is stated rather than
  folded into the word "copies". (3) OpenCL and Vulkan had no shared-memory gap at all:
  `register_types_from_typ` ran over PARAMETER types only, so a record named
  nowhere but the shared declaration was never emitted as a `struct`; it is now
  top-level and runs at both kernel-array declaration sites. The tell was that
  the same kernel written with a whole-slot store (`s.(tid) <- {...}`) compiled
  and ran correctly on all seven non-PTX devices throughout, because the record
  literal registered the type — so a whole-slot-store test would have been green
  the whole time, and `test_shared_record_slots.ml` exercises both shapes.
  CUDA/PTX still refuses, and the test asserts the refusal rather than
  tolerating it: PTX has no struct type and
  `Sarek_ir_ptx_mem.emit_agg_elem_addr` refuses state-space aggregates on
  purpose. What changed there is that the message names the array, the element
  type and the state space. Metal, HIP and WGSL were not measured — no such
  device on this machine — and nothing is claimed about them.

- A `[@@sarek.type]` record with a record-typed field had its inner struct
  emitted AFTER the struct that referenced it, so no kernel touching a nested
  record compiled on any shader backend. Both declaration-emission loops walked
  `kern_types` in list order, and that is not a dependency order: the PPX
  prepends the types reachable through the registry to the ones the kernel
  payload declares. Measured on an RX 7900 XTX host, a read-only nested access
  (`dst.(tid) <- src.(tid).mid.b`) failed on OpenCL ×2 with
  `unknown type name 'Test_..._triple'` and on Vulkan ×2 with a glslang parse
  error at the field line; a three-level chain and two independent nested types
  failed the same way. `Sarek_ir_codegen.sort_record_types_by_dependency` now
  orders the declarations, stably — ties break on the incoming list position,
  so an already-correct order is returned unchanged and no committed golden
  moved. A cycle between distinct record declarations raises `Record_type_cycle`
  instead of being emitted in some wrong order (a self-referencing field is
  dropped, not reported); that path is unreachable through the PPX, which
  refuses a self- or mutually-referencing record field while resolving its
  alignment (measured: both spellings stop at *unknown alignment for field
  type*), and is kept as a backstop for hand-built IR. Interpreter and Native
  were never affected — they carry values, not struct declarations. This orders
  records against records only; the two cross-kind halves it left live are the
  entry below, which also replaced the two identifiers named here — see the
  BREAKING note under Changed for what they became.
- A dependency edge CROSSING between a record declaration and a variant
  declaration was ordered by nothing, because each backend family sorted one
  kind inside its own emission loop and the families disagreed on which loop ran
  first. Both halves are reachable from ordinary `[@@sarek.type]` source and both
  were measured on an RX 7900 XTX / Ryzen 7950X host, two OpenCL devices and two
  Vulkan devices. A variant with a RECORD payload is red on both OpenCL devices
  (`error: unknown type name 'Test_record_variant_decl_order_probe_pt'` at the
  union member) and green on both Vulkan devices; a record with a VARIANT-typed
  field is the exact inverse — red on both Vulkan devices
  (`syntax error, unexpected IDENTIFIER` at the field line) and green on both
  OpenCL devices. Each family being green on the shape the other fails on is why
  a reproducer built on either half alone reports the ordering fixed. Reaching
  either half needs a RUNTIME-SELECTED constructor: static tag erasure (L14
  S1/S2) reduces a variant-typed local or record field written by a literal
  constructor application, and an erased variant never reaches `kern_variants`,
  so a literal-constructor reproducer is green on every device while covering
  nothing. All five generators that declare struct types (CUDA — also HIP —
  OpenCL, Metal, GLSL, WGSL) now emit records and variants from ONE interleaved
  topological pass over both lists together, `Sarek_ir_codegen.gen_type_decls`,
  instead of two per-kind loops. The tie-break is still the incoming index and
  each backend passes a `~tie_break` naming the order its own two loops used to
  run, so an edge-free kernel's emitted source is byte-identical and no committed
  golden moved. Node identity in the sort is the POSITION in the list, not the
  type name: a name-keyed self-edge drop discards a record's genuine edge to a
  SAME-NAMED variant, which `mangle_name` and fusion can both produce, and that
  is precisely the edge class this entry is about. The cyclic-value guard stays
  keyed on PHYSICAL node identity, so a cyclic `elttype` value still terminates
  whatever closes the cycle — including `let rec t = TVec t`, which has neither a
  record nor a variant on it. What this does NOT fix: a type referenced but never
  declared. The PPX registers a variant in `kern_variants` without registering a
  record that appears only in its payload, and no ordering pass can supply a
  declaration that is absent — it still surfaces as the backend's own *unknown
  type name*. Interpreter, Native and PTX were never affected: the first two
  carry values, and the PTX emitter declares no struct types.

- `v.(i).f <- e` — an in-place record-field store on a vector element — did
  something different on each CPU backend, while working on CUDA/PTX. The
  Interpreter **refused** it (`Unsupported_operation "record field assignment"
  / "not fully supported"`). Native had two failure modes split by mutability:
  on a **mutable** field it accepted the store and **silently dropped** it — the
  generated OCaml was a setfield on the fresh record `Vector.get` had just
  marshalled out of storage, so the write hit a temporary, nothing raised, and
  the vector kept its old values — while on an **immutable** field the same
  setfield did not compile (`The record field b is not mutable`), loud but
  misdiagnosed, which is what pushed users to add `mutable` and so into the
  silent half. Both halves are pinned.

  The nested form `v.(i).f.g <- e` needed a second fix on top of the depth-1
  one, on both CPU backends: with depth 1 fixed the Interpreter still read the
  intermediate record through the registry's copying `get_field` and Native still
  matched only the depth-1 shape, so the chained store was silently dropped by
  both. (Before either fix the Interpreter refused every field store, nested or
  not; the silent nested drop is the state the depth-1 fix alone would have
  shipped.) Asserted on every enumerated device, reading the values back and
  checking the neighbouring fields at both levels, because a dropped store is
  otherwise indistinguishable from a kernel that never ran.

  Two limits, stated rather than implied. On Native the store is now a
  whole-element read-modify-write, which makes two threads writing *different*
  fields of the *same* element a lost-update **race** there — an interleaving
  that reads both elements before either writes back loses one store, a
  serialised one loses neither — where the C-family, PTX and interpreter paths
  touch disjoint locations and cannot interfere. Read off the expansion, not
  observed, and covered by no test (backlog-207). And a record type declared
  *inside* the kernel payload is read through a generated getter that has no
  setter, so a store to one is now **refused with a message** instead of silently
  emitting a setfield.

- A record bound to a `let` inside a kernel was a live window onto vector
  storage in the Interpreter rather than a copy, so mutating the local mutated
  the vector element it came from. Detaching it copies through nested records and
  variant payloads, and refuses a chain deeper than 64 links. What that guard
  replaces is not a hang: the recursion is not tail-recursive, so an unguarded
  cyclic value dies with `Stack_overflow` — an untyped crash naming neither the
  value nor the binding — after tens of seconds of allocation and GC thrash
  (measured on one host; the seconds are a property of that `ulimit -s`, so the
  direction is the claim, not the figure). The depth-65 refusal itself is read
  off the marshaller and `bind_var`, not observed, and no committed test covers a
  65-link chain.

- `Sarek_df64` silently ran at plain float32 precision on real NVIDIA
  hardware (CUDA/PTX and NVIDIA OpenCL): `ptxas` contracted the multiply in
  `two_prod` into the `add`/`sub` of the `quick_two_sum` closing `df64_mul`,
  rebuilding the exact product and cancelling the TwoProd error term.
  `mul`/`div`/`sqrt` degraded from ~2^-47 to ~2^-24 with no error reported.
  `two_prod` now forms its product with `fma a b 0.0`, which cannot be fused
  again. Measured on a GTX 1070 (sm_61, CUDA 12.9): mul 5.92e-08 → 9.07e-15,
  div 5.64e-08 → 5.08e-15, throughput unchanged. A CPU-only regression guard
  (`test_df64_no_contraction`) asserts the emitted PTX contains no
  contractable `mul.f32`. The df64 per-backend precision table now names the
  device and toolchain behind every figure — the previous table generalised
  AMD-only measurements to "CUDA/PTX", which is why this went unseen.
- df64 precision gates could not distinguish a working df64 from a collapsed
  one. `test_df64`, `test_real64` and `test_real64_single_source` all widened
  their tolerance to `0x1p-22` (2.38e-07 — four times the float32 unit
  roundoff) on the backends with a documented deviation, so a df64 that had
  degraded all the way to plain float32 (measured 5.84e-08 on RADV) and one
  meeting its contract (9.07e-15) both read PASS. The widening also keyed on
  the bare `Vulkan` framework tag, sweeping in NVIDIA Vulkan, which meets the
  full contract. All three tests now hold every device to the derived contract
  bound (2^-47 add/sub, 2^-46 mul/div/sqrt) and express the documented
  deviations as an explicit expected-failure band — upper end 2^-23, twice the
  float32 unit roundoff — keyed on driver identity (Mesa RADV, Mesa ANV) in
  `Test_helpers.df64_known_deviation`. Degrading past plain float32 now FAILs
  even on an allowlisted device, and so does an allowlisted device that starts
  MEETING the contract (strict XPASS, as in pytest's `xfail(strict=True)`) —
  the run goes red naming the match arm to delete, so the allowlist cannot rot
  behind a green exit code. Red-proved twice: replacing `df64_mul` with a plain
  float32 multiply (old gate PASS on both RADV devices and Native at 1.43e-07
  vs tol 2.38e-07, new gate FAIL), and adding a bogus allowlist entry for an op
  that already meets the contract (all three tests exit 1 with the stale-entry
  message).
- PTX backend emits `sqrt.rn.f32` rather than `sqrt.approx.f32` for the float32
  `sqrt` intrinsic — the same correctness fix already applied to division
  (`div.approx.f32` → `div.rn.f32`, audit finding M2), which had missed `sqrt`.
  Dumping the generated PTX showed `sqrt.approx.f32` (~1 ulp) was the only
  non-correctly-rounded instruction in the whole `df64_sqrt` body, where it
  serves as the Newton seed. This is a global change: every f32 sqrt in every
  PTX kernel is now correctly rounded. Measured worst-case relative error over
  each test's own input set, on a GTX 1070 Max-Q (sm_61, CUDA 12.9, driver
  580.119.02): `df64_sqrt` 1.42e-14 (failing) → 8.53e-15 in `test_df64`, and
  1.68e-14 (failing) → 8.87e-15 in `test_real64`'s df64 fallback. Both post-fix
  figures coincide with the interpreter's for the same inputs — agreement
  between summary statistics, not element-wise identity — so the seed accounted
  for the whole gap these tests can see. Sampled maxima on one device and
  toolchain, not bounds. Costs ~12% kernel time on a sqrt-dominated benchmark
  (`bench_nbody` n=4096: 1.535 ms → 1.722 ms); `rsqrt` is unchanged for code
  that wants the fast form. The same bug class remains open in the OpenCL
  backend (no `-cl-fp32-correctly-rounded-divide-sqrt`, sqrt 1.81e-14) and the
  Vulkan backend (1.68e-14 on NVIDIA); see the `KNOWN RESIDUAL` block in
  `sarek/Sarek_df64/Sarek_df64.ml`.
- Indexed kernel-argument container with strict launch validation, honored
  across all six backends (out-of-order/sparse `set_arg` now correct)
- Unambiguous, collision-resistant compile-cache keys (kernel name included,
  digest-per-field); CUDA kernel-cache eviction on context destroy
- e2e test alias actually executes tests; vacuous tests now assert real
  results; negative suite runs in CI
- Vulkan push-constant scalar binding by logical index; 64-bit write
  alignment; buffer-index validation
- Benchmark dashboard HTML escaping at all sinks (stored-XSS hardening)
- CPU data kept authoritative after Interpreter runs

## 2026-06

### Added

- CUDA backend split into CUDA/PTX and CUDA/C devices (PTX is the default)
- `.shared` (DShared) emission in the PTX backend with formal spec
  (`specs/ptx-dshared-formal.md`)
- T3-SEMANTIC theorem work for the convergence-safety and type-safety
  formal projects
- WGSL/WebGPU codegen backend, in-browser playground and Learn course

### Changed

- Breaking: removed first-party `Obj` escape hatches from Sarek/SPOC execution
  paths.
  - Kernel vector arguments now cross framework/plugin boundaries through typed
    existential accessors and runtime type witnesses.
  - Custom value conversion now uses typed helper lookup instead of raw runtime
    representation casts.
  - Native and interpreter plugin buffer copies use typed Bigarray operations.
  - Custom shared-memory arrays are keyed by typed witnesses to reject
    mismatched reuse.
  - Legacy native direct execution now accepts typed `Framework_sig.exec_arg`
    arrays.

## 2026-01

### Changed

- Documentation cleanup and modernization
  - Comprehensive README rewrite with clear SPOC/Sarek distinction
  - Added sarek/ directory navigation guide
  - Created CONTRIBUTING.md with project guidelines
  - Removed pretentious language ("Grade A", "100%", etc.)
  - Added AI assistance acknowledgment
- Code quality improvements
  - Eliminated 49 failwith calls from Native and Interpreter plugins
  - Added structured error handling using Backend_error pattern
  - Created test suites for Native and Interpreter plugins
  - Added READMEs for Native, Interpreter, ppx_intrinsic, Sarek_float64
- CI/CD modernization
  - Added unit test execution (dune runtest)
  - Created fast benchmark suite (~20s) for CI
  - Integrated coverage measurement with bisect_ppx
  - Simplified workflow to single build job
- Repository cleanup
  - Removed AI artifact files (AGENTS.md, etc.) from history
  - Removed tracked build artifacts (*.exe, *.log, etc.)
  - Updated .gitignore for better hygiene

## 2025-12

### Added

- Full PPX rewrite replacing Camlp4
  - PPX-based kernel syntax ([%kernel ...], [%ktype ...])
  - Type registry for cross-module GPU types
  - Intrinsic registration system (Sarek_stdlib)
  - Float32 stdlib module with math intrinsics
  - Pragma support for loop unrolling hints
- Kernel fusion system (Sarek_fusion.ml)
  - Vertical fusion for producer-consumer patterns
  - Reduction fusion (map + reduce)
  - Stencil fusion with radius tracking
  - Auto-fusion heuristics
- Clean intermediate representation (Sarek_ir.ml)
- BSP superstep syntax with barrier synchronization
- Warp convergence tracking (Sarek_convergence.ml)
- Core primitives with convergence information
- Comprehensive test suite (unit, negative, e2e tests)

### Changed

- Removed Camlp4 dependency entirely
- Improved type inference for kernel parameters
- Better compile-time error messages

## 2024-2025

### Changed

- GPU Backend Overhaul (all 4 backends: CUDA, OpenCL, Vulkan, Metal)
  - Eliminated all failwith calls (94 total removed)
  - Implemented Backend_error.Make functor pattern for structured errors
  - Refactored code generators for maintainability
  - Added comprehensive unit test suites (19-20 tests per backend)
  - Created professional documentation (9-26KB per backend)
  - Improved error messages with context
- OCaml 5.x migration
  - Updated to OCaml 5.4.0
  - Migrated from Domains to Effect handlers where appropriate
  - Maintained backward compatibility

## 20210823

- Add PPX extension to declare external GPGPU kernels
- Update Samples to use PPX instead of Camlp4 extension
- Update dune/opam files for opam release

## 20210816

### Added

- Build with dune
- Compatible with OCaml 4.12
- Switch to github actions for CI
