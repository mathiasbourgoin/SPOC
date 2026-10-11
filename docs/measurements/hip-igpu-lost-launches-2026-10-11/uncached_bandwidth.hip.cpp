// Cost of hipDeviceMallocUncached vs hipMalloc on the iGPU (SPOC #427): a
// bandwidth-bound copy (read N bytes, write N bytes) and a read-only reduction-like sweep.
// Build: hipcc -O1 --offload-arch=gfx1036 --offload-arch=gfx1100 uncached_bandwidth.hip.cpp -o uncached_bandwidth
// Run:   ./uncached_bandwidth DEV MiB REPS
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__); exit(2);} } while (0)
__global__ void copy(const float4 *a, float4 *b, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s) b[i] = a[i];
}
__global__ void sum(const float4 *a, float *out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, s = (size_t)gridDim.x * blockDim.x;
  float acc = 0;
  for (; i < n; i += s) { float4 v = a[i]; acc += v.x + v.y + v.z + v.w; }
  if (acc == 12345.f) out[0] = acc;
}
static void *alloc(size_t bytes, bool uncached) {
  void *p;
  if (uncached) CK(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached));
  else CK(hipMalloc(&p, bytes));
  return p;
}
int main(int argc, char **argv) {
  int dev = atoi(argv[1]); size_t mib = atoi(argv[2]); int reps = atoi(argv[3]);
  CK(hipSetDevice(dev));
  hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, dev));
  size_t bytes = mib << 20, n = bytes / sizeof(float4);
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  for (int unc = 0; unc < 2; unc++) {
    float4 *a = (float4 *)alloc(bytes, unc), *b = (float4 *)alloc(bytes, unc);
    float *o = (float *)alloc(64, unc);
    CK(hipMemset(a, 0, bytes)); CK(hipMemset(b, 0, bytes));
    for (int k = 0; k < 2; k++) {
      for (int r = -1; r < reps; r++) {  // r = -1 is a warm-up
        CK(hipEventRecord(e0));
        if (k == 0) hipLaunchKernelGGL(copy, dim3(1024), dim3(256), 0, 0, a, b, n);
        else hipLaunchKernelGGL(sum, dim3(1024), dim3(256), 0, 0, a, o, n);
        CK(hipGetLastError());
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        if (r >= 0) printf("%s %s %s %zu MiB: %.3f ms, %.1f GB/s\n", pr.name, unc ? "uncached" : "hipMalloc", k == 0 ? "copy(r+w)" : "read-only",
                           mib, ms, (k == 0 ? 2.0 : 1.0) * bytes / (ms * 1e6));
      }
    }
    CK(hipFree(a)); CK(hipFree(b)); CK(hipFree(o));
  }
  return 0;
}
