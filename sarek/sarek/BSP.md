# BSP Model and Supersteps

Sarek provides Bulk Synchronous Parallel (BSP) constructs for structured GPU programming with explicit synchronization.

## Overview

GPU kernels often require synchronization between phases:
1. Load data into shared memory
2. **Barrier** - ensure all threads have loaded
3. Compute using shared data
4. **Barrier** - ensure all threads have computed
5. Write results

The `superstep` construct enforces this pattern with compile-time safety.

## Syntax

### Superstep Blocks

```ocaml
let kernel = [%kernel fun shared input output ->
  (* Phase 1: Load *)
  superstep {
    shared.(thread_idx_x) <- input.(global_idx)
  };

  (* Phase 2: Compute - all threads see loaded data *)
  superstep {
    let sum = shared.(thread_idx_x) + shared.(thread_idx_x + 1) in
    output.(global_idx) <- sum
  }
]
```

Each `superstep { }` block:
- Executes its body
- Inserts an implicit `block_barrier` at the end
- Ensures all threads complete before the next superstep begins

### Barriers

Explicit barriers are also available:

```ocaml
Gpu.block_barrier  (* Synchronize all threads in block *)
```

`block_barrier` is the only synchronisation primitive the front end declares.
`warp_barrier` and `memory_fence` exist as IR statements (`SWarpBarrier`,
`SMemFence`) with a lowering in every backend, but no PPX syntax constructs
them, so they are **not callable from a kernel** today. Earlier revisions of
this section and of the backend READMEs presented `Gpu.warp_barrier` as
available; it never was.

## Convergence Safety

Sarek tracks warp convergence to prevent deadlocks.

### The Problem

```ocaml
(* DEADLOCK: Some threads wait at barrier, others skip it *)
if thread_idx_x < 16 then
  Gpu.block_barrier  (* Only half the threads reach this! *)
```

### Compile-Time Detection

Sarek detects barriers in diverged control flow:

```
Error: Barrier in potentially diverged control flow
  This barrier may not be reached by all threads, causing deadlock.
  Consider using 'superstep ~divergent { }' if intentional.
```

### Convergence States

| State | Meaning | Barrier Allowed |
|-------|---------|-----------------|
| Converged | All threads executing | Yes |
| Uniform | Threads may diverge but will reconverge | Yes (at reconvergence) |
| Diverged | Threads have diverged | No |

### Allowing Divergent Barriers

For advanced cases where divergence is intentional:

```ocaml
superstep ~divergent {
  if thread_idx_x = 0 then
    (* Only thread 0 does cleanup *)
    cleanup ()
}
(* Barrier still inserted, but divergence warning suppressed *)
```

## Warp-Level Operations

### Shuffles, `warp_size`, `lane_id`

A warp is the hardware subgroup (a wave on AMD, a SIMD-group on Apple, a
subgroup in Vulkan and OpenCL). Four shuffles move a value between the lanes of
the warp, on `int32`, `float32` and `float64` (the compiler picks the instance
from the type of the value):

| Call | Value read by lane `i` | When the source is out of range |
|---|---|---|
| `warp_shuffle v x` | lane `x land (W-1)` | cannot occur |
| `warp_shuffle_down v d` | lane `i + d` | `d < 0` or `i + d >= W`: own `v` |
| `warp_shuffle_up v d` | lane `i - d` | `d < 0` or `i - d < 0`: own `v` |
| `warp_shuffle_xor v m` | lane `i lxor m` | `m < 0` or `i lxor m >= W`: own `v` |

`W` is `warp_size` (an `int32` constant that depends on the device: 32 on NVIDIA,
RDNA and Apple GPUs, 32 or 64 on AMD, 8 to 64 on Vulkan and OpenCL devices) and
`i` is `lane_id`, the index of the thread in its warp. The lane operand is read
as a two's-complement `int32`. This table is the contract of every backend; it
does not depend on what a vendor leaves undefined (for example PTX reads only
the low five bits of a delta, and Metal leaves an out-of-range source
undefined): the emitters add the guards that make it hold.

```ocaml
(* Sum of 8 consecutive lanes, in every lane of the group: masks 4, 2, 1 stay
   inside aligned groups of 8, whatever the warp width. *)
let s = x +. warp_shuffle_xor x 4l in
let s = s +. warp_shuffle_xor s 2l in
s +. warp_shuffle_xor s 1l
```

Rules:

- **Every lane of the warp takes part.** There is no mask and no `width`
  argument. A shuffle in thread-varying control flow (under a varying `if`, in a
  loop with a varying bound, after a varying early exit, in a helper called from
  such flow) is a compile-time error: `Warp collective 'warp_shuffle_xor' called
  in diverged control flow`. Sub-warp groups are expressed with `xor` masks
  below a power of two, as above. `down` and `up` inside a segment of the warp
  need a `width` argument, which does not exist yet.
- **The block is a whole number of warps.** A kernel that uses a shuffle is
  refused at launch (`Invalid_dimensions`) when the number of threads of the
  block, `bx * by * bz`, is not a multiple of the warp width of the device. The
  lane of a thread is its linear index `tx + bx * (ty + by * tz)` modulo `W`.
- **Devices.** CUDA (C and PTX), HIP, Metal (generated, not run in CI), Vulkan
  (needs the `BASIC` and `SHUFFLE` subgroup operations in the compute stage) and
  OpenCL (needs `cl_khr_subgroup_shuffle`, and the lane mapping is measured once
  per device) run the shuffles natively. WebGPU/WGSL needs the `subgroups`
  feature (generated, not run in CI). A device without the feature refuses the
  kernel with a message naming the feature (`warp-shuffle unavailable`);
  emulation through shared memory on such devices is planned, not available.
- **Native and the Interpreter** simulate a warp of `Sarek_warp.simulated_width
  ()` lanes (32 by default; `Sarek_warp.with_width` is the test hook): each
  shuffle writes the value to a per-block array, passes a block barrier, reads
  its source and passes a second barrier. This is slow and exact: it is the
  oracle the device results are compared with, bit for bit.
- A shuffle is not `Pure` for the IR passes: kernels that use a warp operation
  are never fused (horizontal or epilogue), windowed together, rewritten to a
  matrix unit, or sliced for host evaluation.

`warp_vote_all`, `warp_vote_any`, `warp_ballot` and `warp_active_mask` are not
available yet; they are refused at PPX time. They arrive with warp-scope
supersteps (roadmap S1).

## Best Practices

### Do

```ocaml
(* Structured phases with supersteps *)
superstep { load_phase () };
superstep { compute_phase () };
superstep { store_phase () }

(* Uniform conditions are safe *)
if block_idx_x = 0 then
  superstep { special_block_work () }
```

### Don't

```ocaml
(* Barrier depends on thread-varying condition *)
if thread_idx_x mod 2 = 0 then
  Gpu.block_barrier  (* ERROR: diverged barrier *)

(* Barrier in loop with thread-varying bound *)
for i = 0 to thread_idx_x do
  Gpu.block_barrier  (* ERROR: different iteration counts *)
done
```

## Implementation

The convergence checker (`Sarek_convergence.ml`) tracks:
- Control flow entry points
- Conditions involving thread indices
- Barrier locations
- Reconvergence points

Errors are reported at compile time, before GPU code generation.

## Example: Parallel Reduction

```ocaml
let reduce = [%kernel fun shared input output n ->
  (* Load into shared memory *)
  superstep {
    shared.(thread_idx_x) <-
      if global_idx < n then input.(global_idx) else 0l
  };

  (* Tree reduction *)
  let stride = ref (block_dim_x / 2) in
  while !stride > 0 do
    superstep {
      if thread_idx_x < !stride then
        shared.(thread_idx_x) <-
          shared.(thread_idx_x) + shared.(thread_idx_x + !stride)
    };
    stride := !stride / 2
  done;

  (* Write result *)
  if thread_idx_x = 0 then
    output.(block_idx_x) <- shared.(0)
]
```
