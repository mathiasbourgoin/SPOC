---
name: roster-spec
type: spec
status: live
feature: Deferred launch window with horizontal fusion of independent launches
brief: /mnt/ssd-external-2to/spoc-pipeline-logs/dynamic-fusion/intake.md
date: 2026-10-11
version: 1.0.0
---

# Spec — Launch window and horizontal fusion

Driver: spoc_llm decodes one token of Qwen3.5-0.8B with 497
`Execute.run_vectors` launches and one of the 35B with 1 145. Each launch costs
4.5-6 µs of host time after #424, and the host is a bound on the 0.8B.
Many consecutive launches are independent. For example, the gate and up
matvecs read the same activation and write different outputs, and so do the
q/k/v projections and the four GDN input projections. This spec is the first
part of automatic dynamic fusion. It covers:

- launches deferred up to the next synchronisation point;
- a dependency graph built from IR read/write sets mapped onto the device
  buffers actually bound;
- horizontal fusion of independent launches;
- fused kernels cached by the sequence of member kernels and replayed.

Epilogue and element-wise vertical fusion are the next spec.

## Clarifications

| Q | A |
|---|---|
| Which launches are deferred? | `Execute.run`'s JIT path with no `~stream`, on a backend whose window is enabled. `run_vectors` goes through it. Stream launches, `run_source`, `Soa_launch`, raw `Kernel` launches and Direct/Custom backends (Native, Interpreter) are never deferred: they flush the window first, then run as before. |
| What is a synchronisation point? | Any operation that moves data between host and device, orders or observes device work, or launches outside the window. See FR-002 for the list. |
| What do errors look like? | Argument checks, capability checks and source generation still run in the `run`/`run_vectors` call and raise there. Device compilation and the driver launch happen when the launch is issued. An error there is raised by the call that triggered the flush (documented on `run_vectors`). |
| What does "bit-identical by construction" mean for horizontal fusion? | No value flows between members. Work-group `b` of the fused kernel runs, unchanged, the code that work-group `b - off_k` of member `k` ran. The only changes are the work-group index and grid size it observes, which are rewritten back to the member's own values. No arithmetic is added, removed, reordered or brought next to another member's arithmetic, so no backend can contract or reassociate across members. It is therefore default-on wherever the window is. |
| Which backends get the window by default? | HIP, OpenCL and Vulkan, the three that can be exercised on this machine. CUDA, Metal and WebGPU stay immediate: CUDA has no NVIDIA GPU here, ladon is off limits for Metal, and WebGPU codegen is not wired. They can opt in with `Launch_window.set_backend_enabled`. This follows the owner rule: untested means risky, and risky means opt-in. |
| How are dependencies decided? | Each vector parameter is classified as read and/or written from the IR. Element reads and writes count, and so do atomics and any use the analysis cannot classify, which count as both. Each bound argument is mapped to its device byte range `[ptr, ptr + bytes)`. Two launches conflict if a range of one overlaps a range of the other and at least one of the two accesses writes. Vector identity is not used, so aliasing through the same buffer is still caught. |
| Can launches be reordered? | Yes. A launch may join an earlier pending group only if it conflicts with no launch in that group or in any later group. Independent launches commute. |
| How long can the GPU be kept waiting? | At most `max_groups` pending groups (default 8). A new group beyond that issues the oldest one, so deferral is bounded by a few launches' worth of host time. |
| How is the fused kernel cached? | By the list of member IRs (physical identity, in order). Dimensions are parameters of the fused kernel, not constants, so one fused IR serves every grid size. It is a stable physical value, so #424's codegen memo and the backend compile cache hit on every replay. |

## User Stories

### US-1: Independent launches become one launch (Priority: P0)
As a host program issuing many small independent kernels, I want SPOC to
merge them so that each costs one host launch instead of several.
**Scope**: launches with equal block shape and equal grid y/z, no dynamic
shared memory, on a window-enabled backend.
**Independent Test**: launch two kernels with disjoint outputs, then read both
back. One backend launch is issued, and the results equal the unfused ones
bit for bit.
**Acceptance Scenarios**:
1. **Given** kernels A (writes `y1`) and B (writes `y2`) that both read `x`,
   with the same block and different grid x, **When** both are launched and
   `y1`, `y2` are read back, **Then** the stats report 2 submitted and 1
   issued, and `y1` and `y2` are bit-identical to unfused execution and to the
   interpreter.
2. **Given** A writes `t` and B reads `t`, **When** both are launched,
   **Then** they are issued as two launches in order.
3. **Given** two launches with different block sizes, **When** launched,
   **Then** they are not fused.
4. **Given** kernels that use barriers and shared memory (row reductions),
   **When** fused, **Then** results are bit-identical (barriers sit inside a
   work-group-uniform branch).

### US-2: Deferral is unobservable (Priority: P0)
As a SPOC user, I want every operation that observes or orders device work to
see all launches made before it.
**Scope**: every public Transfer, Device, Profiling and Kernel operation that
synchronises.
**Independent Test**: for each sync point, launch a writer, perform the
operation, and check that it observes the write.
**Acceptance Scenarios**:
1. **Given** a deferred launch writing `v`, **When** `Vector.get v 0` is
   called (auto-sync), **Then** it returns the kernel's value.
2. **Given** a deferred launch reading `v`, **When** the host then writes `v`
   and launches another reader, **Then** the first launch saw the old value.
3. **Given** a deferred launch, **When** `Transfer.flush`, a copy, a stream or
   event operation, a profiling timer or a raw `Kernel` launch follows,
   **Then** the launch is issued before that operation.
4. **Given** a deferred launch whose device compilation fails, **When** the
   window is flushed by `Transfer.flush`, **Then** `Transfer.flush` raises the
   compilation error, and launches deferred after it are still issued.

### US-3: Control and visibility (Priority: P1)
As a caller or benchmark, I want to turn the window off, flush it explicitly,
and count launches.
**Acceptance Scenarios**:
1. **Given** `SAREK_LAUNCH_WINDOW=0` in the environment or
   `set_enabled false`, **When** a kernel is launched, **Then** it is issued
   immediately (pre-change behaviour).
2. **Given** a run, **When** `stats ()` is read, **Then** it reports the
   launches submitted, the backend launches issued, the fused groups and the
   fused kernels built.

## Challenges

| ID | Story | Challenge | Resolution |
|---|---|---|---|
| C-1 | US-2 | Native/Interpreter read host memory directly; `Vector.get` does not call back into Transfer there | Direct/Custom backends are never deferred (FR-001). |
| C-2 | US-2 | `to_device` runs on every launch for every argument; flushing there would defeat deferral | Flush only on the arms that actually move data (migration, upload). The already-resident arms do not flush. |
| C-3 | US-2 | Created streams are blocking streams that order after default-stream work | Every stream and event operation and every stream launch flushes first. |
| C-4 | US-2 | Re-entrancy: issuing a launch must not recurse into a flush; concurrent domains | One global mutex around submit and flush. A domain-local "flushing" flag makes the hook a no-op inside its own flush. The pending count is atomic, so an empty window costs one atomic read per hook. |
| C-5 | US-1 | Barriers inside the dispatch branch | The branch condition depends only on `block_idx_x`, so it is uniform per work-group. That is legal on CUDA/HIP, OpenCL, GLSL and MSL. |
| C-6 | US-1 | Shared arrays declared inside a branch are illegal in OpenCL/MSL | Only kernels whose shared arrays are bound by a top-of-body `let` prefix are fusable. Those bindings are hoisted, renamed, above the dispatch. |
| C-7 | US-1 | Name clashes between members (parameters, `sarek_<p>_length`, locals) | Every variable and array name of member k is prefixed `m<k>_`, and var ids are offset. |
| C-8 | US-1 | Intrinsics that read the work-group position | `block_idx_x`, `grid_dim_x`, `global_thread_id`, `global_idx(_x)` and `global_size(_x)` are rewritten. Intrinsics outside `Sarek_stdlib`, `SNative`, helper functions using any of those, and coopmat statements make the kernel non-fusable. |
| C-9 | US-2 | GC freeing a buffer a pending launch uses | The pending record holds the argument list (vectors stay rooted). An explicit `free_buffer` flushes first. |
| C-10 | US-1 | A fused kernel fails to compile or launch | Its members are issued individually, and that member sequence is never fused again in this process. |
| C-11 | all | jsoo smoke compiles Execute.ml standalone | New modules use only Stdlib (`Mutex`, `Atomic`, `Domain.DLS`, `Hashtbl`) and are copied into the jsoo smoke build. The hook module is added to the jsoo core shim. |
| C-12 | US-1 | Occupancy: the fused kernel's register count is the members' maximum | Measured (bench). Groups are capped at `max_members` (default 8). |
| C-13 | US-2 | Device destroy or cache clear with pending launches | `Cache_hooks.on_device_destroy` and `on_clear_all` flush first. `at_exit` flushes too. |

## Functional Requirements

#### Window
- **FR-001** [US-2]: `Execute.run` MUST defer a launch only on the JIT path,
  with no `~stream`, on a backend whose window is enabled, with no SoA-dispatched
  argument. Every other launch MUST flush the window and then behave as before.
- **FR-002** [US-2]: The window MUST be flushed before each of these operations
  acts:
  - `Transfer.to_device` when it uploads or migrates;
  - `to_cpu` / `sync` when they read back;
  - `free_buffer`, `free_all_buffers`, `flush`;
  - stream synchronise and destroy;
  - event record, wait, query and synchronise;
  - every ranged copy (`copy_*_to_device` / `copy_*_to_host`, `copy_ptr_to_device`)
    and `stage_to_device`;
  - `free_pinned`, `register_host`, `unregister_host`;
  - `Device.synchronize`;
  - `Profiling` timer start, stop and elapsed;
  - `Kernel.launch` in spoc_core;
  - `Execute.run_source` and stream launches;
  - `Soa_launch.run_soa`;
  - device destroy and cache clear.
- **FR-003** [US-1, US-2]: Issue order MUST respect every conflict
  (overlapping device byte ranges with at least one write) between launches in
  submission order.
- **FR-004** [US-1]: At most `max_groups` groups MAY be pending (default 8). Each
  group holds at most `max_members` launches (default 8).
- **FR-005** [US-2]: Argument checks, capability checks and source generation
  MUST run in the submitting call.
- **FR-006** [US-2]: If issuing a group raises, the flush MUST still issue every
  other pending group, then re-raise the first exception.
- **FR-007** [US-2]: The window MUST be safe under concurrent submission and
  flushing from several domains.
#### Horizontal fusion
- **FR-010** [US-1]: Launches L1..Ln MAY be fused only if:
  - all are pending on the same device;
  - they have equal block dimensions and equal grid y and z;
  - none has dynamic shared memory;
  - they are pairwise conflict-free;
  - every member IR is fusable (C-6, C-8).
- **FR-011** [US-1]: In the fused kernel, work-group `b` with
  `off_k <= b < off_k + gx_k` MUST execute member k's body with `block_idx_x`
  read as `b - off_k` and `grid_dim_x` as `gx_k` (and the derived global index
  and size intrinsics accordingly), with every other statement and expression
  unchanged.
- **FR-012** [US-1]: Fused IRs MUST be cached by the member IR sequence
  (physical identity) and reused physically on every later identical sequence.
- **FR-013** [US-1]: A fused group whose code generation, compilation or launch
  raises MUST be issued member by member, and its member sequence MUST NOT be
  fused again.
#### Control
- **FR-020** [US-3]: The window MUST be enabled by default on HIP, OpenCL and
  Vulkan only. `SAREK_LAUNCH_WINDOW=0` MUST disable it globally.
  `Launch_window.set_enabled` and `set_backend_enabled` MUST control it at run
  time (each first flushes).
- **FR-021** [US-3]: `Launch_window.stats` MUST report submitted launches, issued
  backend launches, fused groups and fused kernels built. `reset_stats` MUST zero
  them.
- **FR-022** [all]: New code MUST NOT use `Obj`.

## Acceptance Criteria

- AC-1 [US-1]: Two independent kernels on HIP → 1 issued launch, bit-identical outputs (CHECK-2).
- AC-2 [US-1]: A dependent pair → 2 launches, in order, correct (CHECK-1, CHECK-2).
- AC-3 [US-1]: Reductions with barriers and shared memory fused → bit-identical to unfused and to the interpreter (CHECK-1 runs the fused IR on the Interpreter; CHECK-2 runs it on HIP/OpenCL/Vulkan).
- AC-4 [US-2]: Each sync point in FR-002 observes prior launches (CHECK-1 with a probe backend that records call order; CHECK-2 on HIP).
- AC-5 [US-2]: A compile failure at issue is raised by the flushing call, and the later group still runs (CHECK-1).
- AC-6 [US-3]: Disabled → immediate issue, identical to pre-change behaviour (CHECK-1).
- AC-7 [US-1]: Bench on the RX 7900 XTX reports issued launches and host µs per submitted launch, with and without the window (CHECK-4).
- AC-8 [mutation]: Breaking the conflict test, the work-group offset rewrite, or one flush hook makes a check go red (recorded in the pipeline log).

## Edge Cases

- EC-1: the same vector bound to two parameters of one launch → handled by range overlap.
- EC-2: a launch whose IR is not fusable → its own group, still deferred.
- EC-3: grid x = 0 for a member → never fused (issued alone).
- EC-4: window full → the oldest group is issued.
- EC-5: program exits with pending launches → `at_exit` flushes.
- EC-6: profiling timers around every launch → each launch is flushed (no fusion under profiling; documented).

## Runnable Checks

- CHECK-1 [AC-2..AC-6]: `dune build @sarek/execute/test/runtest` (probe-backend window tests plus fused-IR interpreter differential tests) → exit 0.
- CHECK-2 [AC-1..AC-4]: `dune exec sarek/tests/e2e/test_launch_window.exe` (all enumerated devices) and `@sarek-hip/test/e2e-hip` → PASS.
- CHECK-3 [all]: `dune build` (with jsoo smoke), `dune runtest`, `dune build @fmt` → exit 0.
- CHECK-4 [AC-7]: `dune exec sarek-hip/test/bench_launch_window.exe` → numbers in the PR.

## Claims Metadata

```claims
{"record":"claims-header","schema_version":1,"namespace":"launch-window","spec_lifecycle":"draft"}
{"record":"requirement","id":"FR-001","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-002","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-003","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-004","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-005","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-006","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-007","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-010","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-011","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-012","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-013","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-020","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-021","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-022","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"acceptance-criterion","id":"AC-1","for":["FR-010","FR-011"]}
{"record":"acceptance-criterion","id":"AC-2","for":["FR-003"]}
{"record":"acceptance-criterion","id":"AC-3","for":["FR-011"]}
{"record":"acceptance-criterion","id":"AC-4","for":["FR-002","FR-001"]}
{"record":"acceptance-criterion","id":"AC-5","for":["FR-006","FR-013"]}
{"record":"acceptance-criterion","id":"AC-6","for":["FR-020"]}
{"record":"acceptance-criterion","id":"AC-7","for":["FR-021","FR-012"]}
{"record":"acceptance-criterion","id":"AC-8","for":["FR-003","FR-011","FR-002"]}
{"record":"check","id":"CHECK-1","for":["AC-2","AC-3","AC-4","AC-5","AC-6"]}
{"record":"check","id":"CHECK-2","for":["AC-1","AC-2","AC-3","AC-4"]}
{"record":"check","id":"CHECK-3","for":["AC-1"]}
{"record":"check","id":"CHECK-4","for":["AC-7"]}
```

## Entities

- `launch window`: per-device list of pending groups of deferred launches.
- `group`: pending launches that will be issued as one backend launch (one member → the original kernel).
- `access`: (device byte range, read, write) for one bound vector argument.
- `fused kernel cache`: member-IR sequence → fused IR, or refused.
