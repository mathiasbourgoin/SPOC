# Launch cost gate

`sarek/execute/test/test_launch_cost.ml` runs in `dune test` (CI included) and fails when the
launch path of `Execute.run_vectors` gets slower for a reason that has nothing to do with the
kernel. It exists because #522 added a whole-IR fold to every launch, took 42 to 53 % off a
consumer decode (281 to 497 launches per token), and was found only by that consumer.
`test_launch_check_cost.ml` pins the two checks that regressed then; this gate covers the whole
path, whichever check the next regression hides in.

## What it measures

N = 2000 launches of one kernel through `Execute.run_vectors` on a probe JIT backend that does
nothing at the device (no GPU, no driver, no clock-dependent code), after a warming launch (which
pays code generation). The figure kept is the minimum over 7 rounds: noise only ever adds time.
Two kernels per case: 1 statement and 20000 statements, with and without a warp shuffle (the
shuffle turns the launch rule on).

## Two assertions

1. **IR-size independence**: a launch of the 20000-statement kernel costs at most 25 times a
   launch of the 1-statement kernel. A launch may depend on the IR only through a cache lookup.
   Anything that folds the IR at each launch is thousands of times slower on the big kernel.
2. **Fixed cost**: a launch costs at most 30 units, where a unit is a reference loop timed in the
   same process (8 hash-table lookups and one formatted string). This catches a constant added to
   every launch, which assertion 1 cannot see. The ratio makes it independent of the machine.

Both are ratios taken inside one process, so they hold on a slow or loaded runner. The test also
prints the measured figures (`launch-cost ...` lines) so a CI log shows the trend.

## Calibration

Apple M4 (10 cores), OCaml 5.4.0, dune build in the default (dev) profile, 2026-10-18, commit
`f12ac093` plus the test.

| Condition | Runs | Big/small ratio | Small launch | Small launch in units |
|---|---|---|---|---|
| Idle machine | 30 | 0.97 to 1.03 | 0.31 to 0.32 us | at most 0.6 |
| 14 busy loops on 10 cores | 15 | 0.99 to 2.23 | at most 0.35 us | at most 0.6 |
| Whole-IR fold in each launch (the #522 shape) | 1 | 1234 to 1965 | 0.34 to 0.54 us | 0.6 to 0.9 |

The limits (25 and 30) sit an order of magnitude above the worst observed healthy value and two to
three orders of magnitude below the regression, so the gate does not flake and cannot be loosened
unnoticed: `test_gate_sees_a_whole_ir_fold` fails if a fold per launch ever stops exceeding the
size limit.

## When it fails

Read the `launch-cost` line. A big/small ratio far above 1 means something walks the IR at each
launch: record its verdict once per IR (a slot of `Ir_memo`). A small launch above 30 units means
a fixed cost was added to the path (allocation, a lock, a hash of the arguments); measure it before
accepting a limit change, and update the calibration table above.
