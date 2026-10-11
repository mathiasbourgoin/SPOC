---
name: roster-spec
type: spec
status: live
feature: Asynchronous, batched kernel launches on Vulkan
brief: /mnt/ssd-external-2to/spoc-pipeline-logs/pkgH/H5a-vulkan-async-launches.md
date: 2026-10-12
version: 1.0.0
---

# Spec — Asynchronous, batched launches on Vulkan

Port of the Metal change (CHANGES.md, "Metal kernel launches are asynchronous";
`specs/metal-ranged-copies.md`) to `sarek-vulkan`. Before, `Kernel.launch`
recorded one dispatch into the stream's single command buffer, submitted it and
waited for its fence: every launch blocked the host for the full
submit-and-complete round trip. Now a launch appends its dispatch to an open
command buffer and returns. The change is semantics-preserving (results and
their order are unchanged, only the point where the host blocks moves), so it
is on by default.

## Clarifications

| Q | A |
|---|---|
| When is a batch submitted? | When it holds `SPOC_VULKAN_BATCH` dispatches (default 32); when another stream of the same device launches (so all streams reach the single queue in launch order); and at every synchronisation point below. Submitting does not wait. |
| When does the host block? | At a synchronisation point (below), and when a launch needs the stream's other command buffer and its previous submission has not completed. |
| How many command buffers? | Two per stream. One records while the other runs, so recording overlaps execution. |
| Why per-launch descriptor sets? | A batch may hold the same kernel several times with different arguments. Each dispatch takes a set from a descriptor pool owned by its command buffer slot; the pools are reset when the slot's fence has been waited for. A pool holds 64 sets and 512 descriptors; a slot grows another pool when one is full. The kernel object no longer owns a pool or a set. |
| How is memory ordered between dispatches? | A global compute-to-compute memory barrier (`SHADER_WRITE` to `SHADER_READ\|SHADER_WRITE`) is recorded before every dispatch. Its first scope includes every earlier command submitted to the queue, so it orders dispatches inside one command buffer and across command buffers and streams. Without it Vulkan orders execution start only (measured below: removing it fails the dependent-chain tests). |
| Several streams? | All streams of a device share one queue. A launch first submits the open batches of the device's other streams (no wait), so submission order is launch order and the barrier makes it a memory order. This costs a submission when a program alternates streams, never a wait. |
| What do errors look like? | A failed `vkQueueSubmit` or `vkWaitForFences` raises `Kernel_launch_failed` naming the kernels of the batch (a repeated kernel as `name xN`), reason from the Vulkan error. A submission that fails while launching (batch full, other stream, event) is kept and raised by the next wait of that stream; one that fails at a synchronisation point is raised there. The failed batch is dropped and the stream stays usable. Host-side refusals (buffer index validation, push-constant layout) still raise at the launch, before the batch is touched. |
| Threads? | One mutex per device (`batch_lock`) guards every stream's batch state; two domains never record into one command buffer. A domain that waits on a fence (a synchronisation point, or reuse of an in-flight command buffer) holds it, so other domains' launches on that device queue behind the wait. `queue_lock` (the queue's external synchronisation) is only ever taken inside `batch_lock`, never the other way round; the FFI calls release the OCaml runtime lock (`~release_runtime_lock:true`) as before. |
| `SPOC_VULKAN_BATCH=1`? | Submit and wait at every launch; errors surface at the launch as before. The barrier and the per-launch descriptor set remain. |

## Requirements

- **FR-001.** `Kernel.launch` returns after recording; with the default batch,
  N back-to-back launches cause about N/32 submissions and no fence wait except
  at the end and when the second command buffer is reused.
- **FR-002.** Every operation that reads or writes device memory from the host,
  or ends the life of an object a pending dispatch uses, first completes the
  pending launches of the device (`Vulkan_api_device.drain_pending`, installed
  by `Vulkan_api_stream`). The inventory:

  | Operation | Behaviour |
  |---|---|
  | `Memory.host_to_device`, `device_to_host`, `host_ptr_to_device`, `device_to_host_ptr` (mapped and staged paths) | drain first |
  | `Memory.run_single_time_command` (every staging copy) | drain first |
  | `Memory.free` | drain first (a pending dispatch may bind the buffer) |
  | `Stream.synchronize` | submit and wait for that stream |
  | `Stream.destroy` | `synchronize`, then destroy slots, pools, command pool; leaves the registry |
  | `Device.synchronize` (`vkDeviceWaitIdle`) | drain first |
  | `Device.destroy` | drain first; a drain failure is raised after the teardown |
  | kernel cache eviction (`clear_cache`, per-device eviction) | drain the kernel's device before destroying its pipeline |
  | `Event.record` | submit the device's open batches (no wait), then the event's empty submission; the event fence therefore covers the batches |
  | `Event.synchronize` | waits for the event fence only |
  | mapped host pointers | no path hands one out except through the `Memory` functions above |

- **FR-003.** Dispatches are ordered in memory (barrier before each dispatch).
- **FR-004.** A failed submission or wait raises `Kernel_launch_failed` naming
  the batch's kernels at the first wait; the stream remains usable.
- **FR-005.** `SPOC_VULKAN_BATCH` (positive integer, read at start; anything
  else gives 32) sets the batch size; `1` restores submit-and-wait per launch.
- **FR-006.** The launch window's results are unchanged
  (`sarek/tests/e2e/test_launch_window.ml` passes on Vulkan).

## Not done

- Metal commits an open buffer as soon as the previous one completed; Vulkan
  submits only at the batch size or a synchronisation point. A decode chain
  shorter than the batch therefore starts on the GPU at the first sync point.
  *(agent decision: avoids a `vkGetFenceStatus` call per launch; the chain
  benchmark below is host-bound, so the GPU is not the limit. Revisit if a GPU-bound
  short chain shows it.)*
- Cross-stream hazards are not tracked per buffer: another stream's open batch is
  submitted at every launch, whether or not it shares a buffer with the
  launch. Metal tracks the buffers.
- `Event.elapsed` stays unsupported (no timestamp queries).

## Backend coverage (package H, 2026-10-12)

| Backend | Before | After | Evidence |
|---|---|---|---|
| Vulkan | submit + fence wait per launch | batched (32), double-buffered, barrier per dispatch | executed on RADV NAVI31 (RX 7900 XTX, gfx1100), Mesa 26.1.4 (vulkan-radeon), Linux 7.1.2-3-cachyos, glslangValidator/shaderc compile; `test_vulkan_async_launches`, `test_launch_window`, benchmarks in CHANGES.md |
| Metal | batched (4c32450b) | unchanged | by-construction (not touched, not run) |
| HIP | asynchronous stream launches | unchanged | by-construction (not touched, not run) |
| CUDA | asynchronous stream launches | unchanged | by-construction (not touched, not run) |
| OpenCL | asynchronous enqueue | unchanged | by-construction (not touched, not run) |
| Native, Interpreter | synchronous CPU execution | unchanged; nothing to batch | by-construction |

## Evidence

Logs under `/mnt/ssd-external-2to/spoc-pipeline-logs/pkgH/`:

- `H5a-before-async-launches.txt`: the new tests on the unchanged code; the
  batching test and the deferred-failure test fail, the ordering tests pass.
- `H5a-after-async-launches.txt`: all pass.
- `H5a-mutation-no-barrier.txt`: with the barrier removed, the dependent-chain,
  mid-batch-read, free-after-launch and batching tests go red (the barrier is
  load-bearing).
- `H5a-bench-launch-overhead.txt`, `H5a-bench-launch-window-vk.txt` and its
  summary: interleaved `SPOC_VULKAN_BATCH=1` / default, 5 pairs.
