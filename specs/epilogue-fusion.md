---
name: roster-spec
type: spec
status: live
feature: Epilogue fusion of element-wise consumers in the launch window
brief: /mnt/ssd-external-2to/spoc-pipeline-logs/dynamic-fusion/intake.md
date: 2026-10-11
version: 1.0.0
---

# Spec — Epilogue fusion

This is the second part of automatic dynamic fusion; the first is
`specs/launch-window.md`. A launch-window producer P is often followed by an
element-wise consumer C that reads P's output at its own index. Examples:

- a matvec followed by the residual add `x[i] <- x[i] + h[i]`;
- the attention output followed by its sigmoid gate;
- the `moe_combine` output followed by the residual add.

C is merged into P: C's body runs right after each of P's stores, in the
thread that made the store. This saves one launch and one pass over the
intermediate. In spoc_llm decode it removes 54 launches per token on the 0.8B
model and 84 on the 35B.

## Clarifications

| Q | A |
|---|---|
| What exactly is fused? | P's store `I[e] <- v` becomes `let t = e in I[t] <- v; if t < tc then (let i = t in C_body)`, where `tc` is the number of threads C was launched with. C then reads `I[t]` from memory, written just before by the same thread. |
| Why is it equal to running P then C? | Three conditions make it equal, and each is checked before merging. (1) For these dimensions and scalar values, P stores each element of I at most once, and every element C writes is one that P stores. This is evaluated on the host by `Index_slice`, which runs only the integer computations the stores depend on, for every thread of both launches. (2) C reads nothing else that P writes, and reads I only at its index. (3) C writes nothing P accesses except I itself, and reads its own outputs only at its index. Under these conditions every value C reads is the value it read when run after P. The same operations run in the same order. |
| Is it bit-identical by construction? | Not on every backend. P's store and C's load are now in one function, so a device compiler that may contract `a*b+c` could fuse P's last multiply with C's first add. HIP compiles with `-ffp-contract=off` forced last (`Hip_rtc.base_options`, verified in `docs/fp-contraction-policy.md`), so epilogue fusion is on by default on HIP only. It is off elsewhere (OpenCL contracts by default, CUDA has `-fmad=true`, Vulkan and Metal are not verified for this shape) and can be turned on with `Execute.Window.set_epilogue_backend`. That is the owner rule: bit-identical by construction may be on by default, anything else is opt-in. |
| What is a consumer? | A kernel of the form `let i = global_thread_id in body`. `body` has no barrier, fence, return, `match`, atomic, shared or local array, other position intrinsic or inline native code. It stores only at index `i`, and does not use a vector whole. It must also pass the horizontal-fusion verdict: stdlib intrinsics only, no free variables, no shared memory. C must be 1-D with no dynamic shared memory. |
| Which producers? | Any pending launch, already fused or not, that binds I to exactly one parameter and never reads I. Its stores to I must be plain element stores (`LArrayElem`). |
| When is the coverage evaluated, and what does it cost? | When C arrives, once per (P IR, C IR, both launch shapes, values of the scalars the evaluation read). Scalars the stores do not depend on, such as the attention position, are not part of the key. If a launch has more than 2^24 threads, or an index or guard depends on data, there is no merge. |
| Interaction with horizontal fusion | The merged launch replaces P in its group. It can host a further epilogue, which gives chains, and it can join or stay in a horizontal group. C must not conflict with P's group mates, nor with any launch of a later group. |
| Failure | If the merged kernel fails to build or launch, P and C are issued as submitted and the pair is never merged again. |

## Functional Requirements

- **FR-101**: A launch C MAY be merged into pending launch P only if all the conditions in the Clarifications hold, on a backend where epilogue fusion is enabled.
- **FR-102**: The merged kernel MUST compute C's element `t` after P's store of `I[t]`, in the same thread, reading `I[t]` from memory, with C's body otherwise unchanged and P's body unchanged apart from binding the store index to a variable.
- **FR-103**: The coverage verdict MUST be computed from these launches' dimensions and scalar values. An unknown result MUST mean no merge.
- **FR-104**: Epilogue fusion MUST be on by default only on HIP. `SAREK_LAUNCH_EPILOGUE=0` MUST turn it off, and `Execute.Window.set_epilogue_backend` MUST turn it on or off per backend.
- **FR-105**: If the merged kernel fails, P and C MUST be issued as submitted, and the pair MUST NOT be merged again.

## Acceptance Criteria and Checks

- AC-1 [FR-101, FR-102]: A covered pair is merged into one launch. On HIP the results are bit-identical to the unfused run (`test_launch_window`, "dependent"). On the Interpreter, the merged IR gives bit-identical results to the pair (`test_launch_window`, "epilogue IR = pair").
- AC-2 [FR-103]: An uncovered consumer (domain larger than the producer's stores) is not merged, and its results stay correct ("uncovered consumer" on every device; the probe test "not merged when uncovered"). Mutation check: making the coverage check always pass turns both red.
- AC-3 [FR-104]: Default backends are checked by a probe test.
- AC-4: spoc_llm decode on the RX 7900 XTX: dispatch counts, and logits bit-identical with and without epilogue fusion.

Checks:
- CHECK-1: `dune build @sarek/execute/test/runtest`.
- CHECK-2: `dune exec sarek/tests/e2e/test_launch_window.exe`, which runs under `dune runtest` on every enumerated device.
- CHECK-3: `@sarek-hip/test/e2e-hip`, `dune build`, `dune build @fmt`.

## Claims Metadata

```claims
{"record":"claims-header","schema_version":1,"namespace":"epilogue-fusion","spec_lifecycle":"draft"}
{"record":"requirement","id":"FR-101","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-102","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-103","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-104","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-105","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"acceptance-criterion","id":"AC-1","for":["FR-101","FR-102"]}
{"record":"acceptance-criterion","id":"AC-2","for":["FR-103"]}
{"record":"acceptance-criterion","id":"AC-3","for":["FR-104"]}
{"record":"acceptance-criterion","id":"AC-4","for":["FR-102"]}
{"record":"check","id":"CHECK-1","for":["AC-2","AC-3"]}
{"record":"check","id":"CHECK-2","for":["AC-1","AC-2"]}
{"record":"check","id":"CHECK-3","for":["AC-1"]}
```
