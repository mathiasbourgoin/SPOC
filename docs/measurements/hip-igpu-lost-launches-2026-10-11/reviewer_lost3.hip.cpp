// Reviewer repro v3: is the LAUNCH lost or its DATA lost? Each launch (b,i) writes the tag into
// device memory (hipMalloc) and into fine-grained host memory (hipHostMalloc, coherent), and bumps a
// device counter and a host counter (atomicAdd_system). Host sleeps at random positions are logged.
// usage: lost3 DEV NBATCH BATCH SLEEP_PCT SLEEP_US
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP error %s at line %d\n", hipGetErrorString(e_), __LINE__); exit(2);} } while (0)
__global__ void mark(int *dout, int *dcnt, volatile int *hout, int *hcnt, int b, int i) {
  if (threadIdx.x == 0) {
    int tag = b * 100000 + i;
    dout[i] = tag; atomicAdd(dcnt + b, 1);
    hout[i] = tag; atomicAdd_system(hcnt + b, 1);
  }
}
int main(int argc, char **argv) {
  if (argc < 6) { fprintf(stderr, "usage\n"); return 3; }
  int dev = atoi(argv[1]), nb = atoi(argv[2]), batch = atoi(argv[3]), pct = atoi(argv[4]), us = atoi(argv[5]);
  srand(4242);
  CK(hipSetDevice(dev));
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, dev));
  int *dout, *dcnt, *hout, *hcnt;
  CK(hipMalloc(&dout, sizeof(int) * (batch + 1)));
  CK(hipMalloc(&dcnt, sizeof(int) * nb));
  CK(hipHostMalloc(&hout, sizeof(int) * (batch + 1), hipHostMallocCoherent));
  CK(hipHostMalloc(&hcnt, sizeof(int) * nb, hipHostMallocCoherent));
  CK(hipMemset(dout, 0, sizeof(int) * (batch + 1))); CK(hipMemset(dcnt, 0, sizeof(int) * nb));
  for (int i = 0; i <= batch; i++) hout[i] = 0;
  for (int b = 0; b < nb; b++) hcnt[b] = 0;
  CK(hipDeviceSynchronize());
  std::vector<int> h(batch + 1);
  long dlost = 0, hlost = 0, dshort = 0, hshort = 0, prefix_ends_at_last_sleep = 0, bad = 0;
  for (int b = 1; b < nb; b++) {
    int last_sleep = 0;
    for (int i = 1; i <= batch; i++) {
      if (pct && rand() % 100 < pct) { usleep(us); last_sleep = i; }
      hipLaunchKernelGGL(mark, dim3(1), dim3(64), 0, 0, dout, dcnt, hout, hcnt, b, i);
      CK(hipGetLastError());
    }
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h.data(), dout, sizeof(int) * (batch + 1), hipMemcpyDeviceToHost));
    int dc; CK(hipMemcpy(&dc, dcnt + b, 4, hipMemcpyDeviceToHost));
    int bl = 0, hi = 0, hl = 0;
    for (int i = 1; i <= batch; i++) {
      if (h[i] != b * 100000 + i) { bl++; hi = i; }
      if (hout[i] != b * 100000 + i) hl++;
    }
    dlost += bl; hlost += hl; dshort += batch - dc; hshort += batch - hcnt[b];
    if (bl || hl) {
      bad++; if (bl && hi == last_sleep - 1) prefix_ends_at_last_sleep++;
      if (bad <= 10) printf("  batch %d: dev_lost=%d (last lost=%d) host_lost=%d dcnt=%d hcnt=%d last_sleep_before=%d\n", b, bl, hi, hl, dc, hcnt[b], last_sleep);
    }
  }
  printf("RESULT %s dev=%d launches=%ld dev_out_lost=%ld host_out_lost=%ld dev_cnt_missing=%ld host_cnt_missing=%ld bad_batches=%ld lost_prefix_ends_right_before_last_sleep=%ld | pct=%d us=%d\n",
         p.name, dev, (long)(nb - 1) * batch, dlost, hlost, dshort, hshort, bad, prefix_ends_at_last_sleep, pct, us);
  return 0;
}
