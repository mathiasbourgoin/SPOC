(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Cold-process OpenCL kernel build latency, with and without the persistent
 * disk cache (H5c).
 *
 * One process = one measurement: the point is what a FRESH process pays, so
 * the in-process caches never help. Prints one line per kernel:
 *
 *   kernel=<name> device=<idx> build_ms=<f> launch_ms=<f> total_ms=<f>
 *
 * [build_ms] spans clCreateProgramWithSource, Program.build (disk lookup or
 * clBuildProgram, plus the store on a miss) and clCreateKernel. Run it under
 * sarek-opencl/bench/run_ab.sh, which alternates SAREK_DISK_CACHE=0 (always
 * compiles from source: the behaviour before the cache) with a warm cache, in
 * separate processes.
 *
 * Usage: bench_opencl_disk_cache.exe <device-index>
 ******************************************************************************)

open Sarek_opencl

(* Shapes of the decode-path kernels: a row-per-workgroup matvec with a local
   reduction, an RMS norm, and a tiled GEMM. *)
let kernels =
  [
    ( "matvec_f32",
      {|
__kernel void matvec_f32(__global const float *w, __global const float *x,
                         __global float *y, int cols, __local float *scratch) {
  int row = get_group_id(0);
  int lid = get_local_id(0);
  int lsz = get_local_size(0);
  float acc = 0.0f;
  for (int c = lid; c < cols; c += lsz)
    acc += w[row * cols + c] * x[c];
  scratch[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsz / 2; s > 0; s >>= 1) {
    if (lid < s) scratch[lid] += scratch[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) y[row] = scratch[0];
}
|}
    );
    ( "rmsnorm",
      {|
__kernel void rmsnorm(__global const float *x, __global const float *g,
                      __global float *y, int n, float eps,
                      __local float *scratch) {
  int lid = get_local_id(0);
  int lsz = get_local_size(0);
  float ss = 0.0f;
  for (int i = lid; i < n; i += lsz) ss += x[i] * x[i];
  scratch[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsz / 2; s > 0; s >>= 1) {
    if (lid < s) scratch[lid] += scratch[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float scale = rsqrt(scratch[0] / (float)n + eps);
  for (int i = lid; i < n; i += lsz) y[i] = x[i] * scale * g[i];
}
|}
    );
    ( "gemm_tiled",
      {|
#define TILE 16
__kernel void gemm_tiled(__global const float *a, __global const float *b,
                         __global float *c, int m, int n, int k) {
  __local float ta[TILE][TILE];
  __local float tb[TILE][TILE];
  int row = get_global_id(1);
  int col = get_global_id(0);
  int ly = get_local_id(1);
  int lx = get_local_id(0);
  float acc = 0.0f;
  for (int t = 0; t < k; t += TILE) {
    ta[ly][lx] = (row < m && t + lx < k) ? a[row * k + t + lx] : 0.0f;
    tb[ly][lx] = (col < n && t + ly < k) ? b[(t + ly) * n + col] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int i = 0; i < TILE; i++) acc = fma(ta[ly][i], tb[i][lx], acc);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (row < m && col < n) c[row * n + col] = acc;
}
|}
    );
  ]

(* Argument binding and launch geometry per kernel. Sizes are small and fixed:
   the launch exists to force the driver's lazy device-code compilation, not to
   measure throughput. *)
let bind_and_geometry name kernel (a, b, c) =
  let open Opencl_api.Kernel in
  set_arg_buffer kernel 0 a ;
  set_arg_buffer kernel 1 b ;
  set_arg_buffer kernel 2 c ;
  match name with
  | "matvec_f32" ->
      set_arg_int32 kernel 3 256l ;
      set_arg_local kernel 4 (64 * 4) ;
      ((4096, 1, 1), (64, 1, 1))
  | "rmsnorm" ->
      set_arg_int32 kernel 3 256l ;
      set_arg_float32 kernel 4 1e-5 ;
      set_arg_local kernel 5 (64 * 4) ;
      ((64, 1, 1), (64, 1, 1))
  | _ ->
      set_arg_int32 kernel 3 64l ;
      set_arg_int32 kernel 4 64l ;
      set_arg_int32 kernel 5 64l ;
      ((64, 64, 1), (16, 16, 1))

let seconds_since started = (Unix.gettimeofday () -. started) *. 1000.0

(* [build_ms]: program creation, build (disk lookup or clBuildProgram, plus the
   store on a miss) and clCreateKernel. [launch_ms]: first enqueue plus finish,
   which is where a lazily compiling driver pays for device code. *)
let measure queue buffers (name, source) =
  let context = queue.Opencl_api.CommandQueue.context in
  let started = Unix.gettimeofday () in
  let program = Opencl_api.Program.create_from_source context source in
  Opencl_api.Program.build program () ;
  let kernel = Opencl_api.Kernel.create program name in
  let build_ms = seconds_since started in
  let global, local = bind_and_geometry name kernel buffers in
  let launched = Unix.gettimeofday () in
  Opencl_api.Kernel.launch queue kernel ~global ~local ;
  Opencl_api.CommandQueue.finish queue ;
  let launch_ms = seconds_since launched in
  Opencl_api.Kernel.release kernel ;
  Opencl_api.Program.release program ;
  (build_ms, launch_ms)

let () =
  let index =
    if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 0
  in
  let device = Opencl_api.Device.get index in
  let context = Opencl_api.Context.create device in
  let queue = Opencl_api.CommandQueue.create context () in
  let buffer () = Opencl_api.Memory.alloc context 65536 Bigarray.float32 in
  let buffers = (buffer (), buffer (), buffer ()) in
  List.iter
    (fun ((name, _) as kernel) ->
      let build_ms, launch_ms = measure queue buffers kernel in
      Printf.printf
        "kernel=%s device=%d build_ms=%.3f launch_ms=%.3f total_ms=%.3f\n%!"
        name
        index
        build_ms
        launch_ms
        (build_ms +. launch_ms))
    kernels
