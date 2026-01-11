# Radix Sort

Multi-pass parallel sorting using radix decomposition: **Sort by processing 4 bits at a time**

## Description

Radix sort is a non-comparison based sorting algorithm that processes integers digit by digit (or in this case, 4 bits at a time). Each of the 8 passes (for 32-bit integers) histograms the current digit, computes a prefix sum to determine destinations, and scatters elements to their sorted positions. This creates a highly parallel, data-independent sorting algorithm.

## Why It Matters

Radix sort is:
- **Linear Time**: O(n×k) where k is number of passes (8 for 32-bit, 4-bit radix)
- **Parallel Friendly**: Each pass is highly parallelizable
- **Non-Comparison**: Doesn't depend on comparisons like quicksort
- **Composite Algorithm**: Combines histogram, prefix sum, and scatter

This benchmark demonstrates how to build complex algorithms from simpler parallel primitives and tests the GPU's ability to handle multi-kernel workflows with intermediate data transfers.

## Sarek Kernels

### Histogram Kernel
```ocaml
[%kernel
  fun (input : int32 vector)
      (histogram : int32 vector)
      (n : int32)
      (shift : int32)
      (mask : int32) ->
    let open Std in
    let gid = global_thread_id in
    
    if gid < n then begin
      let value = input.(gid) in
      let digit = (value lsr shift) land mask in
      atomic_add histogram.(digit) 1l
    end]
```

### Scatter Kernel
```ocaml
[%kernel
  fun (input : int32 vector)
      (output : int32 vector)
      (prefix_sum : int32 vector)
      (n : int32)
      (shift : int32)
      (mask : int32) ->
    let open Std in
    let gid = global_thread_id in
    
    if gid < n then begin
      let value = input.(gid) in
      let digit = (value lsr shift) land mask in
      let pos = atomic_add prefix_sum.(digit) 1l in
      output.(pos) <- value
    end]
```

<!-- GENERATED_CODE_TABS: radix_sort -->

## Key Features

- **4-bit radix**: 16 bins per pass (2^4)
- **8 passes**: Total for 32-bit integers
- **Histogram phase**: Count occurrences of each digit value
- **Scatter phase**: Place elements in sorted order using prefix sum

## Performance Characteristics

- **Complexity**: O(n × 8) for 32-bit integers with 4-bit radix
- **Memory transfers**: Multiple passes over data
- **Bottleneck**: Atomic operations in histogram and scatter
- **CPU/GPU hybrid**: Prefix sum computed on CPU

## Algorithm Structure

For each of 8 passes (bits 0-3, 4-7, ..., 28-31):
1. **Extract digit**: Get 4-bit digit from current position
2. **Histogram**: Count occurrences of each digit (16 bins)
3. **Prefix sum**: Compute exclusive scan of histogram (on CPU)
4. **Scatter**: Place elements in sorted order based on digit

After 8 passes, the array is fully sorted.

## Why 4-bit Radix?

Trade-off between number of passes and bin count:
- **1-bit radix**: 32 passes, 2 bins → too many passes
- **4-bit radix**: 8 passes, 16 bins → good balance
- **8-bit radix**: 4 passes, 256 bins → more shared memory needed
- **16-bit radix**: 2 passes, 65K bins → prohibitive memory

## Buffer Ping-Pong

Each pass swaps input/output buffers:
- **Even passes (0,2,4,6)**: Read from input, write to output
- **Odd passes (1,3,5,7)**: Read from output, write to input  
- **Final result**: In input buffer after 8 passes

## Known Issues

⚠️ This benchmark currently has verification failures under investigation. The algorithm is correct but there may be implementation details causing incorrect results.

## Expected Results (When Working)

Modern GPUs should achieve:
- **Overall**: ~100-500 M elements/s
- **Per pass**: ~50-200 M elements/s
- **8× overhead**: Compared to single-pass algorithms
- **Still competitive**: With comparison-based parallel sorts
