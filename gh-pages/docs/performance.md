---
layout: page
title: Launch Window and Fusion
---

# Launch Window and Fusion

When a program issues many small kernels, the host time spent on each launch can exceed the time the GPU spends running it. On HIP, OpenCL, Vulkan and CUDA, Sarek therefore holds JIT launches back until the next point where your program observes the device, merges independent ones into a single kernel launch, and on HIP also merges element-wise consumers into the kernel that produces their input. This page says what is on by default, what flushes the held launches, how to see the effect, how to turn it off, and one change you may notice: device-compile and launch errors can surface in a later call than the launch.

## Defaults per backend

| Feature | HIP | OpenCL | Vulkan | CUDA | Metal | Native, Interpreter, CPU devices |
|---|---|---|---|---|---|---|
| Launch window (deferral, horizontal fusion) | on | on | on | on | off (opt-in; also WebGPU) | never |
| Epilogue fusion | on | off | off | off | off | never |

- The window is on by default for the backends in the first row, and a device is excluded from deferral if it is a CPU device (host-pointer buffers from `Host_ptr_buffer` alias host memory there). Native and Interpreter execute directly and are never deferred.
- When Metal is opted in, a fused kernel is limited to 31 argument slots (each vector takes two, each scalar one, and every member adds an offset and a size). A group that would pass it is split, so a long chain of launches with many arguments is fused in several smaller groups.
- On Vulkan every scalar and vector length of a kernel travels in one 128-byte push-constant block. A fused kernel carries all its members' (vector lengths, then scalars) and an int32 offset and size per member, so a group whose block would pass 128 bytes is split, and so is an epilogue merge that would pass it. Six launches of one vector and three int32 scalars, for example, are issued as five fused plus one, not six single launches.
- Metal is opt-in (`Execute.Window.set_backend_enabled ~backend:"Metal" true`) until the window tests have been executed on Apple hardware; only the generated fused source has been checked so far.
- CUDA/PTX was tested through ZLUDA on an AMD GPU, and CUDA/C shares HIP's CUDA C generator; neither has run on NVIDIA hardware. Switch it off with `Execute.Window.set_backend_enabled ~backend:"CUDA/PTX" false` if you see a difference.
- Epilogue fusion is on only on HIP. HIP compiles every kernel with floating-point contraction forced off, so merging two kernels cannot change a result. On the other backends the device compiler may contract, in the merged kernel, a multiply of the first kernel with an add of the second, or share a product that both compute (which the unfused consumer would contract), changing the result by an ulp: observed on Vulkan (RADV), and rusticl and nvrtc contract inside kernels as well. So it stays opt-in there (`Execute.Window.set_epilogue_backend`), a precision grant; Metal is exact by construction (`fp contract(off)`) but not yet executed on Apple hardware. Details: [docs/fp-contraction-policy.md](https://github.com/mathiasbourgoin/Sarek/blob/main/docs/fp-contraction-policy.md), section 14.
- Horizontal fusion produces bit-identical results by construction: each member runs unchanged and nothing is added, removed or reordered between members.

## What is deferred, and what flushes

A launch is deferred when all of these hold:

- it goes through `Execute.run` or `Execute.run_vectors` on a JIT backend;
- it has no `~stream`;
- the backend's window is on and the device is not a CPU device.

Every other launch (stream launches, `Execute.run_source`, `Soa_launch`, raw `Kernel` launches) first issues everything that is held, then runs as before.

Held launches are issued, in an order that respects every dependency, before any of these runs:

- any read-back from the device (`Vector.get`, `Transfer.to_cpu`, `Transfer.sync`), and an upload (`Transfer.to_device`) of a vector that a held launch uses;
- `Transfer.flush`, `Device.synchronize`, and freeing buffers;
- stream and event operations, ranged copies (`copy_*`), `stage_to_device`, `register_host`, `free_pinned`;
- `Profiling` timer start, stop and elapsed;
- destroying a device, clearing the backend caches, and program exit.

You normally do not need to do anything: the observable result is the same as issuing each launch immediately. To make the device start earlier, call `Execute.Window.flush ()`.

At most 8 groups of launches are held; a ninth issues the oldest first. A group fuses at most 8 launches.

## Asynchronous launches on Vulkan

Independently of the window, a Vulkan launch no longer waits for the GPU. It is appended to the stream's open command buffer and returns; the buffer is submitted after `SPOC_VULKAN_BATCH` dispatches (default 32), when another stream of the device launches, and at any point where your program observes the device: reading or writing a vector, `Transfer.flush`, `Device.synchronize`, `Stream.synchronize`, destroying a device. Freeing a buffer does not submit or wait: the release is queued until the pending launches have completed. Results and their order do not change. Dispatches that depend on each other are ordered in memory by a barrier the backend records before each one.

On an RX 7900 XTX with RADV, the host cost of a launch dropped from about 59 us to 6.6 us (`SPOC_VULKAN_BATCH=1` against the default; see [CHANGES.md](https://github.com/mathiasbourgoin/Sarek/blob/main/CHANGES.md) for the method and the chain numbers).

A failed submission or wait raises `kernel_launch_failed` naming the kernels of the batch at the first wait that observes it (a read, a synchronisation), not at the launch. `SPOC_VULKAN_BATCH=1` submits and waits at every launch again. Details: `specs/vulkan-async-launches.md` in the repository.

## Error timing

Argument checks, capability checks, source generation and argument expansion run inside the `run` or `run_vectors` call, so those errors are raised where the launch is made. Device compilation and the driver launch happen when the launch is issued. A failure there is raised by whichever call flushed the window (for example `Transfer.flush` or `Vector.get`), not by the launch that caused it. The other held groups are still issued and the first exception is re-raised afterwards. One known exception: if a merged epilogue kernel fails and falls back, the producer is issued first and, if it raises, the consumer is not issued ([issue #459](https://github.com/mathiasbourgoin/Sarek/issues/459)). If the window overflows, the oldest group is issued before the new launch is accepted; if that raises, the new launch is not made. A failed deferred launch at program exit is printed on stderr.

If you need an error at the launch site while debugging, turn the window off (below).

A fused kernel that fails to build or launch does not fail your program: its members are issued one by one, and that combination is not fused again until `Execute.Window.clear_fused_cache ()` is called.

## See what happened

`Execute.Window.stats ()` counts launches since the last `reset_stats`:

| Field | Meaning |
|---|---|
| `submitted` | JIT launches made through `run` and `run_vectors` |
| `issued` | backend launches actually made |
| `fused_groups` | groups of several launches issued as one kernel |
| `fused_built` | distinct fused kernels built; `reset_stats` does not reset it |
| `fused_refused` | launch sequences whose fusion failed |

If `issued` is lower than `submitted`, launches were merged. Horizontal merges also raise `fused_groups`; epilogue merges do not, and show only as `issued` lower than `submitted`. The example launches two kernels that read the same input and write different outputs; they can run as one launch.

```ocaml
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let double =
  [%kernel
    fun (x : float32 vector) (y : float32 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let i = global_thread_id in
      if i < n then y.(i) <- x.(i) * 2.0]

let negate =
  [%kernel
    fun (x : float32 vector) (z : float32 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let i = global_thread_id in
      if i < n then z.(i) <- 0.0 - x.(i)]

let ir_of (_, kirc) =
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

let launch dev ir args n =
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args
    ~block:(Execute.dims1d 256)
    ~grid:(Execute.dims1d ((n + 255) / 256))
    ()

let run_two dev =
  let n = 1024 in
  let x = Vector.create Vector.float32 n in
  let y = Vector.create Vector.float32 n in
  let z = Vector.create Vector.float32 n in
  Vector.fill x 1.0 ;
  let double_ir = ir_of double and negate_ir = ir_of negate in
  Execute.Window.reset_stats () ;
  launch dev double_ir [Vec x; Vec y; Int n] n ;
  launch dev negate_ir [Vec x; Vec z; Int n] n ;
  Transfer.flush dev ;
  let s = Execute.Window.stats () in
  Printf.printf
    "submitted=%d issued=%d fused_groups=%d y[0]=%f z[0]=%f\n"
    s.Execute.Window.submitted
    s.Execute.Window.issued
    s.Execute.Window.fused_groups
    (Vector.get y 0)
    (Vector.get z 0)
```

On HIP this should print `submitted=2 issued=1 fused_groups=1`; with the window off, `issued=2`. A test with a fake backend checks the mechanism (`sarek/execute/test/test_launch_window.ml:378-379`). Two launches are fused only if they are independent: if the second reads what the first writes, they are issued in order as two launches.

## Turn it off or on

| Scope | How |
|---|---|
| Whole process, at start | `SAREK_LAUNCH_WINDOW=0` (also `false`, `off`, `no`) |
| Whole process, at run time | `Execute.Window.set_enabled false` |
| One backend | `Execute.Window.set_backend_enabled ~backend:"HIP" false` |
| Epilogue fusion, at start | `SAREK_LAUNCH_EPILOGUE=0` (also `false`, `off`, `no`) |
| Epilogue fusion, one backend | `Execute.Window.set_epilogue_backend ~backend:"HIP" false` |

The setters flush held launches first. The backend argument is the framework name shown by `sarek-device-info` (`HIP`, `OpenCL`, `Vulkan`, `Metal`, `CUDA/PTX`, `CUDA/C`). Turning the window on for a backend outside the default set (Metal, until executed on Apple hardware; WebGPU, whose codegen is not wired) is an opt-in. Turning epilogue fusion on for OpenCL or Vulkan accepts the possible ulp difference described above.

```ocaml
let configure () =
  (* Keep launches immediate while debugging a device-compile error. *)
  Execute.Window.set_enabled false ;
  (* Or: opt Metal in, keep everything else at the defaults. *)
  Execute.Window.set_enabled true ;
  Execute.Window.set_backend_enabled ~backend:"Metal" true ;
  Printf.printf
    "window on HIP: %b, epilogue on HIP: %b\n"
    (Execute.Window.defers "HIP")
    (Execute.Window.epilogue_on "HIP")
```

`Execute.Window.clear_fused_cache ()` forgets every fused kernel built so far.

## When two launches fuse

Horizontal fusion merges launches that all hold:

- same device, same block dimensions, same grid y and z, no dynamic shared memory;
- no dependency between them: they do not bind the same vector with at least one of them writing it;
- a kernel that uses only Sarek stdlib intrinsics, has no free variables, and uses shared arrays of constant size declared at the top of the body; a kernel that reads `grid_dim_x`, uses the `block_id_x` alias (write `block_idx_x`), reads a position intrinsic in a helper function, or contains a cooperative-matrix statement or inline native code is not fused;
- the summed grid x and shared memory stay within the device's limits.

Launches that do not qualify stay in the window and are issued in order. A launch may join an earlier held group only if it conflicts with nothing in that group or in any later one, so independent launches can be issued ahead of an unrelated launch.

Epilogue fusion merges a launch C into a held launch P when C is a one-dimensional element-wise kernel (`let i = global_thread_id in ...`, no barrier, atomic, shared or local array) that reads P's output at its own index and stores only at its own index. C's body then runs right after P's store, in the same thread. Before merging, Sarek evaluates on the host, for these dimensions and scalar values, that P stores each element at most once and that every element C writes is one P stores. If it cannot prove that (for example an index depends on data, or a launch has more than 2^24 threads), there is no merge and nothing changes.

Explicit, caller-driven fusion of kernel IRs is a different mechanism and can delete intermediate arrays; see [Kernel Fusion](https://github.com/mathiasbourgoin/Sarek/blob/main/sarek/sarek/FUSION.md). The window never deletes a store.

## Repeated launches

The IR record is compared by physical identity. Take it once from `[%kernel ...]` (as `double_ir` above) and reuse it: source generation, the capability check and the backend's compile-cache key are then skipped on later launches. After you change an intrinsic registration at run time, call `Execute.clear_launch_cache ()`.

## First launch in a new process

OpenCL compiles a kernel from source the first time a process uses it, and used to do so in every process. The OpenCL backend now keeps the driver's program binary on disk (Vulkan already kept SPIR-V), so a second process loads it instead. Measured on tartarus (RX 7900 XTX through rusticl, and the Intel OpenCL CPU runtime on a Ryzen 9 7950X; the machine was loaded, so treat the numbers as provisional), the build of a matvec, an RMS-norm and a tiled GEMM kernel took, median of 7 separate processes each:

| Device | From source | From the disk cache |
|---|---|---|
| Intel OpenCL CPU | 91 / 33 / 42 ms | 1.6 / 1.1 / 1.2 ms |
| rusticl, Mesa shader cache off | 7.7 / 3.8 / 5.3 ms | 2.2 / 1.9 / 4.2 ms |
| rusticl, Mesa shader cache on (default) | 1.8 / 0.8 / 0.9 ms | 0.7 / 0.5 / 0.6 ms |

Mesa keeps its own cache, so on rusticl the gain is small by default. The results are the same bits as a build from source. See [Compile cache](environment.html#compile-cache) for the key, the fallback and `SAREK_DISK_CACHE=0`.

## See also

- [Memory and transfers](memory.html): streams, pinned memory and the other operations that flush.
- [Device & Backend Selection](device_selection.html): `sarek-device-info` and `SPOC_DISABLE_*`.
- [HIP backend](https://github.com/mathiasbourgoin/Sarek/blob/main/sarek-hip/README.md): the backend where both fusions are on by default.
- [Matrix units](matrix.html): the automatic matrix-product rewrite that goes through the same launch path.
- [Environment variables](environment.html): every `SAREK_*` and `SPOC_*` variable with its accepted values.
