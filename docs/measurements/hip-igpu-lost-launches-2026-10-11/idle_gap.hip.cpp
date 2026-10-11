// Plain-HIP, no SPOC, no copies in the loop: is the first kernel launched after
// an idle gap lost? Each trial: usleep(gap), launch a kernel that stores the
// trial number into a pinned-free device word, hipDeviceSynchronize, read back.
// Build: hipcc -O1 --offload-arch=gfx1036 --offload-arch=gfx1100 idle_gap.hip.cpp -o idlegap ; run: ./idlegap DEV TRIALS GAP_US [MODE]
// MODE 0: plain launch (default); 1: a hipMemcpyHtoD (4 B) just before the launch.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(2);} } while (0)
__global__ void store(int *out, int v) { if (threadIdx.x == 0 && blockIdx.x == 0) *out = v; }
int main(int argc, char **argv) {
  int dev = atoi(argv[1]), trials = atoi(argv[2]), gap = atoi(argv[3]);
  int mode = argc > 4 ? atoi(argv[4]) : 0;
  CK(hipSetDevice(dev));
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, dev));
  int *d; CK(hipMalloc(&d, 4));
  int lost = 0;
  for (int t = 1; t <= trials; t++) {
    int sentinel = -5; CK(hipMemcpy(d, &sentinel, 4, hipMemcpyHostToDevice));
    CK(hipDeviceSynchronize());
    if (gap) usleep(gap);
    if (mode == 1) CK(hipMemcpy(d + 0, &sentinel, 4, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(store, dim3(1), dim3(64), 0, 0, d, t);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    int v; CK(hipMemcpy(&v, d, 4, hipMemcpyDeviceToHost));
    if (v != t) lost++;
  }
  printf("%s gap=%dus mode=%d: %d of %d launches lost\n", p.name, gap, mode, lost, trials);
  return 0;
}
