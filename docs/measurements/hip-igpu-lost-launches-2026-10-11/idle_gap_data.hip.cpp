// SPOC #427, plain HIP, no SPOC: is DATA written to device memory lost across an
// idle gap on the gfx1036 iGPU? Each trial writes a tag, sleeps, reads it back.
// Build: hipcc -O1 --offload-arch=gfx1036 --offload-arch=gfx1100 idle_gap_data.hip.cpp -o idle_gap_data
// Run:   ./idle_gap_data DEV TRIALS GAP_US WRITER ALLOC [WAKE]
//   WRITER: kernel  a kernel stores the tag (hipGetLastError checked after the launch)
//           htod    hipMemcpy host-to-device stores the tag (no kernel at all)
//           batch-kernel / batch-htod: TRIALS batches of 300 writes to 300 distinct words,
//                   a GAP_US sleep before ~5% of them, one read-back per batch; reports
//                   lost words (the pattern of the SPOC loop; GAP_US is the sleep)
//   ALLOC:  default hipMalloc ; uncached hipExtMallocWithFlags(hipDeviceMallocUncached)
// Optional 6th argument WAKE=1: after the gap, launch one unrelated kernel (touching
// another word) before reading back, as in the SPOC loop where work follows the gap.
// Output: trials where the tag read back after the gap differs from the tag written.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__); exit(2);} } while (0)
__global__ void store(int *out, int v) { if (threadIdx.x == 0 && blockIdx.x == 0) *out = v; }
int main(int argc, char **argv) {
  if (argc < 6) { printf("usage: %s DEV TRIALS GAP_US kernel|htod default|uncached\n", argv[0]); return 2; }
  int dev = atoi(argv[1]), trials = atoi(argv[2]), gap = atoi(argv[3]);
  bool kernel = !strcmp(argv[4], "kernel"), uncached = !strcmp(argv[5], "uncached");
  bool wake = argc > 6 && atoi(argv[6]) == 1;
  CK(hipSetDevice(dev));
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, dev));
  if (!strncmp(argv[4], "batch", 5)) {
    bool bk = !strcmp(argv[4], "batch-kernel");
    const int W = 300;
    int *db;
    if (uncached) CK(hipExtMallocWithFlags((void **)&db, 4 * W, hipDeviceMallocUncached));
    else CK(hipMalloc((void **)&db, 4 * W));
    int lostw = 0, badb = 0;
    srand(4242);
    for (int b = 1; b <= trials; b++) {
      int fill[W], got[W];
      for (int i = 0; i < W; i++) fill[i] = -5;
      CK(hipMemcpy(db, fill, 4 * W, hipMemcpyHostToDevice));
      CK(hipDeviceSynchronize());
      for (int i = 0; i < W; i++) {
        if (rand() % 20 == 0) usleep(gap);
        int tag = b * 1000 + i;
        if (bk) { hipLaunchKernelGGL(store, dim3(1), dim3(64), 0, 0, db + i, tag); CK(hipGetLastError()); }
        else CK(hipMemcpy(db + i, &tag, 4, hipMemcpyHostToDevice));
      }
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(got, db, 4 * W, hipMemcpyDeviceToHost));
      int l = 0;
      for (int i = 0; i < W; i++) if (got[i] != b * 1000 + i) l++;
      lostw += l; badb += l > 0;
    }
    printf("%s writer=%s alloc=%s gap=%dus: %d of %d batches lost words (%d words of %d)\n", p.name, argv[4], argv[5], gap, badb, trials, lostw, trials * W);
    return 0;
  }
  int *d;
  if (uncached) CK(hipExtMallocWithFlags((void **)&d, 8, hipDeviceMallocUncached));
  else CK(hipMalloc((void **)&d, 8));
  int lost = 0;
  for (int t = 1; t <= trials; t++) {
    int old = -5; CK(hipMemcpy(d, &old, 4, hipMemcpyHostToDevice));
    CK(hipDeviceSynchronize());
    if (kernel) {
      hipLaunchKernelGGL(store, dim3(1), dim3(64), 0, 0, d, t);
      CK(hipGetLastError());
    } else {
      CK(hipMemcpy(d, &t, 4, hipMemcpyHostToDevice));
    }
    CK(hipDeviceSynchronize());
    usleep(gap);
    if (wake) {
      hipLaunchKernelGGL(store, dim3(1), dim3(64), 0, 0, d + 1, t);
      CK(hipGetLastError());
      CK(hipDeviceSynchronize());
    }
    int v; CK(hipMemcpy(&v, d, 4, hipMemcpyDeviceToHost));
    if (v != t) lost++;
  }
  printf("%s writer=%s alloc=%s gap=%dus wake=%d: %d of %d tags lost\n", p.name, argv[4], argv[5], gap, (int)wake, lost, trials);
  return 0;
}
