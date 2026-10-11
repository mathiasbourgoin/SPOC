// Plain-HIP reproducer for SPOC #427 (no SPOC): 300 iterations of
// host write -> hipMemcpyHtoD (null stream) -> kernel copying into its own slice.
// Build: hipcc -O1 --offload-arch=gfx1036 --offload-arch=gfx1100 repro427.hip.cpp -o repro427 ; run: ./repro427 DEV RUNS [MODE]
// MODE: 0 plain (default), 1 hipDeviceSynchronize after each launch,
// 2 initialise device 0 first (two HIP contexts alive),
// 4 hipMalloc(4 KiB) and hipFree of the previous one every iteration (allocator
//   churn, like finalisers running mid-loop),
// 5 usleep(20 ms) before ~5% of the iterations (queue goes idle, then a launch),
// 3 sync + read the slice back after each launch; if lost, re-read after 50 ms
//   (late visibility?) and relaunch the same kernel once (flaky dispatch?).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <unistd.h>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(2);} } while (0)
__global__ void snap(int *out, const int *x, int n, int base) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) out[base + tid] = x[tid];
}
int main(int argc, char **argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 1, runs = argc > 2 ? atoi(argv[2]) : 20;
  int mode = argc > 3 ? atoi(argv[3]) : 0;
  const int n = 4096, iters = 300;
  if (mode == 2) { CK(hipSetDevice(0)); CK(hipFree(0)); }
  CK(hipSetDevice(dev));
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, dev));
  int badruns = 0;
  for (int r = 0; r < runs; r++) {
    std::vector<int> fill((size_t)n * iters, -5), src(n), all((size_t)n * iters);
    int *out, *x;
    CK(hipMalloc(&out, sizeof(int) * n * iters));
    CK(hipMalloc(&x, sizeof(int) * n));
    CK(hipMemcpy(out, fill.data(), sizeof(int) * n * iters, hipMemcpyHostToDevice));
    void *churn = nullptr;
    for (int it = 0; it < iters; it++) {
      if (mode == 5 && rand() % 20 == 0) usleep(20000);
      if (mode == 4) { if (churn) CK(hipFree(churn)); CK(hipMalloc(&churn, 4096)); }
      for (int i = 0; i < n; i++) ((volatile int *)src.data())[i] = it + 1000;
      CK(hipMemcpy(x, src.data(), sizeof(int) * n, hipMemcpyHostToDevice));
      hipLaunchKernelGGL(snap, dim3(n / 64), dim3(64), 0, 0, out, x, n, it * n);
      CK(hipGetLastError());
      if (mode == 1) CK(hipDeviceSynchronize());
      if (mode == 3) {
        CK(hipDeviceSynchronize());
        int v; CK(hipMemcpy(&v, out + it * n, 4, hipMemcpyDeviceToHost));
        if (v != it + 1000) {
          int v2, v3; usleep(50000);
          CK(hipMemcpy(&v2, out + it * n, 4, hipMemcpyDeviceToHost));
          hipLaunchKernelGGL(snap, dim3(n / 64), dim3(64), 0, 0, out, x, n, it * n);
          CK(hipDeviceSynchronize());
          CK(hipMemcpy(&v3, out + it * n, 4, hipMemcpyDeviceToHost));
          printf("  LOST it=%d first=%d after50ms=%d after_relaunch=%d\n", it, v, v2, v3);
        }
      }
    }
    CK(hipMemcpy(all.data(), out, sizeof(int) * n * iters, hipMemcpyDeviceToHost));
    int bad = 0;
    for (int it = 0; it < iters; it++) {
      bool ok = true;
      for (int i = 0; i < n; i++) if (all[(size_t)it * n + i] != it + 1000) ok = false;
      if (!ok) { bad++; printf("  iteration %d: slice starts with %d\n", it, all[(size_t)it * n]); }
    }
    if (bad) badruns++;
    CK(hipFree(out)); CK(hipFree(x));
  }
  printf("%s: %d of %d runs with a bad slice (mode %d)\n", p.name, badruns, runs, mode);
  return badruns ? 1 : 0;
}
