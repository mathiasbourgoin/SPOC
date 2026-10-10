# transfer-pitfalls measurements, 2026-10-10

Machine: tartarus (Ryzen 9 7950X, 125 GB RAM, RX 7900 XTX, ROCm 7.2, Linux
7.1.2-3-cachyos). The machine was shared with other workloads, with 59 GB in
zram swap and a 17 GB tmpfs; free RAM is printed at the head of each file.
Spec: `specs/transfer-pitfalls.md`.

| File | What |
|---|---|
| `bench_hip_small_writes.txt` | `sarek-hip/test/bench_hip_small_writes.exe`, 3 runs (final code) |
| `bench_hip_register_host.txt` | `bench_hip_register_host.exe`, 3 runs, with a `populated` scenario (the `~populate` option, since withdrawn) |
| `bench_hip_register_host_5way.txt` | 4 runs with the `reg+write` and `pop+write` scenarios added |
| `stress_hip_staged_launches.txt` | the C-9 reproducer, default mode (`stage_to_device`) |
| `earlier-runs.txt` | output of the earlier runs and probes this PR cites, copied from the session |
