---
name: roster-spec
type: spec
status: live
feature: Ranged copies, device-only vectors, pinned and registered host memory, events and stream launches on Vulkan
brief: /mnt/ssd-external-2to/spoc-pipeline-logs/pkgH/H7-vulkan-async-api.md
date: 2026-10-12
version: 1.0.0
---

# Spec — the asynchronous / memory API on Vulkan (slice 1)

Vulkan used `Framework_sig.Async_refused`: ranged copies, pinned memory,
`register_host`, device-only vectors, `copy_bytes_to_host`, `stage_to_device`,
`stream_wait_event`, `event_query` and launches on a stream all raised
`Feature_not_supported`. HIP, CUDA and (except async copies) Metal provide them
(`specs/device-buffers-async-copy.md`, `specs/metal-ranged-copies.md`). Vulkan
is the path for Intel GPUs and AMD iGPUs without ROCm, so this was a parity gap
with users. This slice implements everything Vulkan can honour exactly
**except asynchronous copies** (`async_copy = false`, below).

Decisions the brief left open were **decided autonomously by the agent** and
are marked *(agent decision)*.

## Clarifications

| Q | A |
|---|---|
| Which `async_support` flags? | `ranged_copy`, `pinned_host`, `host_register`, `stream_launch`, `stream_events` true; `async_copy` false, like Metal. `support` is per backend: `host_register` additionally depends on the device (next row), so `Transfer.register_host` on a device without the extension (or not opted in, or AMD) raises `Feature_not_supported` from the call itself and never degrade; `alloc_pinned` needs neither and can instead fail with an out-of-memory error from `vkAllocateMemory`. Consumers that must know up front call the operation; the record alone does not say. |
| Pinned and registered memory? | Pinned: `alloc_pinned` allocates through the driver (`vkAllocateMemory`, HOST_VISIBLE | HOST_COHERENT and not DEVICE_LOCAL, HOST_CACHED preferred), maps it once and uses the mapped pointer as the pinned address, wrapped in a transfer-only `VkBuffer`; no host page reaches the driver (no userptr object, not the #503 path), no extension and no opt-in needed, so it works on AMD. The block belongs to its device and is unmapped with it. Registered: host ranges imported into the driver with `VK_EXT_external_memory_host` (`vkGetMemoryHostPointerPropertiesEXT`, `VkImportMemoryHostPointerInfoEXT`) and wrapped in a `VkBuffer` (`Vulkan_api_host`). A copy between such a range and a device buffer is one `vkCmdCopyBuffer`: no staging buffer, no host `memcpy`. The import covers whole pages (the range rounded outwards to `minImportedHostPointerAlignment`, 4096 on RADV). `register_host` imports the pages around an existing Bigarray. The driver, not `mlock`, keeps the pages resident. The extension is requested at device creation when advertised, the instance is Vulkan 1.1+ and `vkGetPhysicalDeviceExternalBufferProperties` reports host allocations importable as transfer buffers; `Vulkan_api_device.t.host_import_alignment` is `None` otherwise. Each import checks the buffer's own size requirement against the rounded range, and `register_host` is refused when the granule is larger than a page (rounding would reach pages the process may not own). *(agent decision: a driver-allocated mapped block, not an import, for pinned memory: it removes the staging copy like an import (22 GB/s against 5.5 GB/s staged, RX 7900 XTX) without handing host pages to the driver (#503). `pread` with `O_DIRECT` into the block works on RADV, checked by `test_vulkan_host_lifecycle`.)* |
| Is a copy asynchronous? | No. Every copy completes before it returns (`async_copy = false`); the `~stream` argument is accepted and changes nothing. Each copy first completes the device's pending launches (`Vulkan_api_device.drain_pending`, H5a), so it is ordered against launches on every stream: a read-back sees what earlier launches wrote, a write cannot land under a launch still reading the buffer, and a launch recorded after a copy sees it. Real asynchronous copies (copy commands recorded into the stream's batch, completing at sync points, with the staging released after the batch) are slice 2; they change observable behaviour and so are opt-in per the owner rule. |
| Which path does a copy take? | (1) Host range inside an import and the host cannot access the device buffer at memory speed (unmapped, or a mapping of uncached memory for reads): one DMA `vkCmdCopyBuffer` between the import and the device buffer. (2) Host-mapped device buffer: `memcpy` at the byte offset; a pinned write of at least 16 MiB (`Vulkan_api_transfer.dma_min_bytes`) still uses DMA because it is faster. (3) Otherwise a staging buffer of exactly the range, one `memcpy`, one `vkCmdCopyBuffer`; read-back staging is HOST_CACHED when the device has such a type. |
| Why not read a mapped device buffer with `memcpy`? | Measured on the RX 7900 XTX: VRAM mapped through the BAR is write-combined and uncached; reading it is 0.03 GB/s (64 MiB). `Vulkan_api_memory.buffer.mapped_cached` records whether the mapping is HOST_CACHED; reads from an uncached mapping go through a cached staging buffer (2.6 GB/s here, per-call allocation included) or, from a pinned range, DMA (22 GB/s). The whole-buffer `Memory.device_to_host`/`device_to_host_ptr` follow the same rule: 0.03 GB/s before, 2.6 GB/s after at 64 MiB (`H7-before-bench-whole-buffer.txt`, `H7-after-bench-whole-buffer.txt`). |
| Device-only vectors? | `Transfer.create_device_only` needs only `ranged_copy`, so it works with no change in core: a buffer with no host mirror. |
| `copy_bytes_to_host`, `stage_to_device`? | Same contract as Metal: `copy_bytes_to_host` is a ranged read-back; `stage_to_device` falls back to the synchronous ranged copy because `async_copy` is false. |
| `stream_wait_event`? | The stream remembers the event (`Vulkan_api_slot.stream.awaited`); `Stream.synchronize` also waits for the event's fence, because core's `Transfer.synchronize_stream` releases the waited-for work transitively once the waiting stream has synchronised. Ordering of later work needs nothing more: all streams of a device share one queue and a launch first submits the open batches of the device's other streams, so everything an event covers was submitted before any later launch, and the barrier before each dispatch (`Vulkan_api_barrier`) orders it in memory. |
| `event_query`? | `vkGetFenceStatus` on the event's fence, without blocking; `true` if never recorded, `false` while the recorded work is pending. While another domain holds the event's lock (recording or waiting) it answers `false` rather than wait. A signalled fence is marked not pending, so the next `record` need not wait. |
| Launches on a stream? | `run_source_on_stream` compiles for the stream's own device and records into the stream's batch (H5a); previously refused. |
| What still refuses? | `device_to_device`, `memset`, `Event.elapsed`, raw-pointer kernel arguments, and registered memory on a device without `VK_EXT_external_memory_host` (or not opted in, or AMD). |

## Requirements

- **FR-001.** A ranged host-to-device or device-to-host copy moves exactly the
  requested bytes at the requested byte offset; bytes outside the range are
  untouched; the range is bounds-checked before any Vulkan call
  (`Invalid_argument`).
- **FR-002.** A copy completes the device's pending launches first and is
  complete on return.
- **FR-003.** A host range inside an import is copied without a staging buffer
  when the device buffer is unmapped or not host-readable.
- **FR-004.** `free_pinned` and `unregister_host` refuse an address that is not
  a live allocation/registration of that kind (`Invalid_argument`);
  registering an address twice is refused. Imports are destroyed before their
  `VkDevice`; a pinned allocation is driver memory and is unmapped with its
  device, after which any access is undefined behaviour and `free_pinned` still
  succeeds and touches nothing. Free blocks before destroying the device: the
  table is keyed by address, so a late free after the address was reused is
  indistinguishable from a free of the new block.
- **FR-005.** `event_query` never blocks.
- **FR-006.** `Transfer.register_host` on a device without the extension (or
  not opted in, or AMD), and ranged copies and device-only vectors on a device
  without ranged copies, raise `Feature_not_supported` naming the backend; no
  fallback. `Transfer.alloc_pinned` needs neither.

## Backend coverage (package H, H7 slice 1, 2026-10-12)

| Backend | Before | After | Evidence |
|---|---|---|---|
| Vulkan | all of the above refused | ranged copies, device-only vectors, `copy_bytes_to_host`, `stage_to_device` (synchronous), pinned and registered memory (imported), `event_query`, `stream_wait_event`, launches on a stream; `async_copy` refused | executed on RADV NAVI31 (RX 7900 XTX, gfx1100), Mesa 26.1.4 (vulkan-radeon), Linux 7.1.2-3-cachyos: `test_device_buffers` (full suite), `test_vulkan_ranged_copies` |
| HIP, CUDA | full async support | unchanged | by-construction (not touched, not run) |
| Metal | all but `async_copy` | unchanged | by-construction (not touched, not run) |
| OpenCL | refused | unchanged (still refused; the suite's refusal check runs on rusticl, RX 7900 XTX) | executed |
| Native, Interpreter | synchronous | unchanged | executed (`test_device_buffers`) |

## Not done (slice 2 and findings)

- **Asynchronous copies** (`async_copy`, and with it asynchronous
  `stage_to_device` and copy/compute overlap): needs per-slot completion actions
  (release the staging buffer, copy a read-back into the host pointer after the
  batch's fence) in `Vulkan_api_batch`/`Vulkan_api_slot`. Pinned memory is
  already importable, so pinned copies need no staging at all.
- **Staging pool.** Every staged copy allocates, maps and frees a staging
  buffer; at 1 MiB this dominates (3.5 GB/s staged against 19 GB/s mapped
  `memcpy`, and 0.84 GB/s read-back).
- `test_vulkan_copy_domains` is a regression guard (it caught the
  use-after-collect of the copy submission structures), not evidence for the
  private command pool per copy, which holds by construction; a shared-pool
  mutant is not detected by it.
- Lazy FFI bindings: the first use of a binding from two domains at once
  raises `Lazy.Undefined` (pre-existing, backend-wide; the multi-domain test
  warms up first).
- `register_host` of a range whose pages the process does not own is the
  caller's error (the driver rejects the import); not specially detected.

## Evidence

Logs under `/mnt/ssd-external-2to/spoc-pipeline-logs/pkgH/`:

- `H7-before-device-buffers.txt`: `test_device_buffers` with Vulkan switched
  from the refusal check to the full suite, on the unchanged backend. Fails on
  the first operation (`Feature not supported: device-only vectors`).
- `H7-after-device-buffers.txt`: the same, passing (exit 0; the whole suite on
  the RX 7900 XTX).
- `H7-mutation-no-drain.txt`: `Vulkan_api_transfer` without the pending-launch
  drain; the ordering test goes red (copy over a launch's input).
- `H7-mutation-no-import.txt`: import lookup disabled; the byte comparison still
  passes but the staging-buffer count assertion goes red (a silent fallback to
  staging is detected).
- `H7-before-event-wait.txt`, `H7-mutation-query-true.txt`: the event tests on
  the first slice (the waiting-stream test fails) and with `query` always true
  (the pending test fails).
- `H7-after-vulkan-tests-x3.txt`: all `sarek-vulkan` tests, three consecutive
  forced runs (includes the H5a async executables).
- `H7-bench-pinned-copy.txt`: `bench_vulkan_pinned_copy`, interleaved
  configurations, 7 rounds each at 64 MiB, 16 MiB, 1 MiB and 256 KiB, with
  `uptime` and top consumers. Load average was about 4.7, so the numbers are
  provisional.
