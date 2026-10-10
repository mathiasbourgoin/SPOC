// SPDX-License-Identifier: CECILL-B
// SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
//
// rocBLAS reference for sarek-hip/test/bench_hip_coopmat_gemm.ml
// (specs/coopmat-hip-wmma.md): D = A x B + C, row-major, A m x k f16,
// B k x n f16, C/D m x n f32, f32 compute, via rocblas_gemm_ex. Same method
// as the Sarek benchmark: hipEvents around a batch of back-to-back calls on
// the default stream, `samples` batches per round, `rounds` rounds; median
// and min-max of per-call time.
//
// Build: hipcc -O2 --offload-arch=gfx1100 rocblas_gemm_f16.cpp -lrocblas \
//          -o rocblas_gemm_f16
// Run:   ./rocblas_gemm_f16 [rounds] [samples] [batch] [f32|f16]
// The last argument is the C/D type (default f32, what Sarek's kernel
// produces). f16 output is measured too because rocBLAS 5.2 / ROCm 7.2.4 on
// gfx1100 is several times faster with it: the f32-output path does not reach
// the tuned kernels (specs/coopmat-hip-wmma.md, Benchmark).
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK_HIP(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); std::exit(1); } } while (0)
#define CHECK_RB(x) do { rocblas_status s = (x); if (s != rocblas_status_success) { \
  std::fprintf(stderr, "%s: %s\n", #x, rocblas_status_to_string(s)); std::exit(1); } } while (0)

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

int main(int argc, char **argv) {
  int rounds = argc > 1 ? std::atoi(argv[1]) : 3;
  int samples = argc > 2 ? std::atoi(argv[2]) : 10;
  int batch = argc > 3 ? std::atoi(argv[3]) : 10;
  bool f16_out = argc > 4 && std::string(argv[4]) == "f16";
  rocblas_datatype cd = f16_out ? rocblas_datatype_f16_r : rocblas_datatype_f32_r;
  rocblas_handle h;
  CHECK_RB(rocblas_create_handle(&h));
  hipEvent_t e0, e1;
  CHECK_HIP(hipEventCreate(&e0));
  CHECK_HIP(hipEventCreate(&e1));
  const int m = 256, k = 3072;
  for (int n : {1024, 3072, 12288}) {
    std::vector<_Float16> ha((size_t)m * k), hb((size_t)k * n);
    std::vector<float> hc((size_t)m * n);
    for (size_t i = 0; i < ha.size(); ++i) ha[i] = (_Float16)((float)((i * 7919) % 17) / 16.0f - 0.5f);
    for (size_t i = 0; i < hb.size(); ++i) hb[i] = (_Float16)((float)((i * 7919) % 17) / 16.0f - 0.5f);
    for (size_t i = 0; i < hc.size(); ++i) hc[i] = (float)((i * 104729) % 13) / 16.0f;
    void *da, *db, *dc, *dd;
    CHECK_HIP(hipMalloc(&da, ha.size() * 2));
    CHECK_HIP(hipMalloc(&db, hb.size() * 2));
    CHECK_HIP(hipMalloc(&dc, hc.size() * 4));
    CHECK_HIP(hipMalloc(&dd, hc.size() * 4));
    // For f16 output only the first half of the C bytes is read, as f16; the
    // values do not matter for timing.
    CHECK_HIP(hipMemcpy(da, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(db, hb.data(), hb.size() * 2, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dc, hc.data(), hc.size() * 4, hipMemcpyHostToDevice));
    float alpha = 1.0f, beta = 1.0f;
    // Row-major D = A B + C is column-major D^T = B^T A^T + C^T.
    auto call = [&]() {
      CHECK_RB(rocblas_gemm_ex(h, rocblas_operation_none, rocblas_operation_none,
                               n, m, k, &alpha, db, rocblas_datatype_f16_r, n,
                               da, rocblas_datatype_f16_r, k, &beta,
                               dc, cd, n,
                               dd, cd, n,
                               rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
    };
    call();
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<double> all;
    std::printf("rocblas_gemm_ex f16->%s    M=%d N=%-5d K=%d  round medians",
                f16_out ? "f16" : "f32", m, n, k);
    for (int r = 0; r < rounds; ++r) {
      std::vector<double> round;
      for (int s = 0; s < samples; ++s) {
        CHECK_HIP(hipEventRecord(e0, 0));
        for (int b = 0; b < batch; ++b) call();
        CHECK_HIP(hipEventRecord(e1, 0));
        CHECK_HIP(hipEventSynchronize(e1));
        float ms;
        CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
        round.push_back(ms * 1000.0 / batch);
      }
      std::printf(" %.1f", median(round));
      all.insert(all.end(), round.begin(), round.end());
    }
    double med = median(all);
    double flops = 2.0 * m * n * (double)k;
    std::printf("\n  median %9.1f us (%6.2f TFLOP/s)  min %9.1f max %9.1f us\n", med,
                flops / (med * 1e-6) / 1e12, *std::min_element(all.begin(), all.end()),
                *std::max_element(all.begin(), all.end()));
    hipFree(da); hipFree(db); hipFree(dc); hipFree(dd);
  }
  rocblas_destroy_handle(h);
  return 0;
}
