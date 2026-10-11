# OpenCL host-pointer buffer RSS, 2026-10-11

Machine: Ryzen 9 7950X, 125 GB RAM, Intel CPU OpenCL runtime 2026.21.3.0.31
(OpenCL 3.0), Linux 7.1.2-3-cachyos. Shared with other workloads; the 1-min
load average is at the head of each file. Each file is one run of
`sarek/tests/e2e/bench_opencl_host_ptr_rss.exe MODE GIB`, built from the
branch that adds `Spoc_core.Host_ptr_buffer`.

| File | What |
|---|---|
| `bench_opencl_host_ptr_rss-host_ptr-3gib.txt` | 3 GiB array, `Host_ptr_buffer.of_bigarray` |
| `bench_opencl_host_ptr_rss-copy-3gib.txt` | 3 GiB array, plain `Vector.of_bigarray` (default copy path) |
| `bench_opencl_host_ptr_rss-host_ptr-4gib.txt` | 4 GiB array, `Host_ptr_buffer.of_bigarray` |

VmRSS growth from "array filled" to "after read-back": host-pointer 3 GiB
+116.6 MiB, copy 3 GiB +3188.7 MiB, host-pointer 4 GiB +116.7 MiB. The
host-pointer growth is the same at 3 and 4 GiB, so it does not scale with the
array (kernel compilation and runtime threads); the copy path adds one copy of
the array.
