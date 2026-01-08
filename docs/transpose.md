---
layout: index_sample
title: Matrix Transpose Example
---

# Matrix Transpose

Matrix transpose is a memory-bound operation. A naive implementation often suffers from uncoalesced memory writes. This example shows how to use shared memory to coalesce both reads and writes.

## Optimized Kernel

The kernel reads a tile from the input matrix into shared memory (coalesced read). It then writes the tile to the output matrix from shared memory, but with indices swapped. By adjusting the write indices, we ensure that writes to global memory are also coalesced.

```ocaml
open Sarek

let%kernel transpose_optimized (input : float32 vector) (output : float32 vector)
                               (width : int32) (height : int32) =
  (* Shared memory tile (padded to avoid bank conflicts) *)
  let%shared tile = Array.create Float32 (16 * (16 + 1)) in
  
  let x_in = get_global_id 0 in
  let y_in = get_global_id 1 in
  let tx = thread_idx_x in
  let ty = thread_idx_y in
  
  (* Coalesced read into shared memory *)
  if x_in < width && y_in < height then
    tile.((ty * 17) + tx) <- input.((y_in * width) + x_in);
    
  (* Wait for all threads to load *)
  barrier ();
  
  (* Calculate transposed indices *)
  let x_out = (block_idx_y * 16) + tx in
  let y_out = (block_idx_x * 16) + ty in
  
  (* Coalesced write from shared memory *)
  if x_out < height && y_out < width then
    output.((y_out * height) + x_out) <- tile.((tx * 17) + ty)
```

## Host Code

```ocaml
let run_transpose () =
  let width, height = 2048, 2048 in
  let device = Device.get_default () in
  
  let block = (16, 16, 1) in
  let grid = ((width + 15)/16, (height + 15)/16, 1) in
  
  Execute.run transpose_optimized
    ~device ~grid ~block
    [Vec input; Vec output; Int32 width; Int32 height]
```
