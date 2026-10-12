#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# gpu-kernel-log-gate.sh — fail a GPU test run when the kernel log shows a GPU
# fault since the run started.
#
# A GPU test can report success while the driver logged a page fault or a ring
# reset (Sarek #504: test_vulkan_async passed while amdgpu logged write faults
# and two gfx ring resets), and a driver bug can oops the kernel (#503). The
# test's own verdict is not enough, so the run is wrapped:
#
#   scripts/gpu-kernel-log-gate.sh -- <command> [args...]
#
# 1. Scans `journalctl -k` for the last GPU_LOG_GATE_PRESCAN (default 10 min).
#    A machine that is already faulting does not start another GPU run (exit 3).
# 2. Runs the command.
# 3. Scans `journalctl -k --since <run start>`. Any match fails the gate
#    (exit 1) whatever the command returned.
#
# Exit: the command's own status when the log is clean; 1 = fault in the log
# after the run; 3 = fault in the log before the run (command not started);
# 4 = the kernel log cannot be read (fails closed; GPU_LOG_GATE_ALLOW_NO_LOG=1
# runs the command unguarded and says so).
#
# Patterns (extended regex, case-insensitive): GPU_LOG_GATE_PATTERN, default
# page fault, ring timeout/reset, GPU reset, Oops, BUG:, WARNING: CPU:,
# 'blocked for more than', *ERROR*. GPU_LOG_GATE_SETTLE (default 3 s) is waited
# before the post-run scan.
# GPU_LOG_GATE_LOG_CMD overrides the log reader (it receives --since's value as
# $1 and prints log lines); used by the red-path harness.
set -uo pipefail

PATTERN="${GPU_LOG_GATE_PATTERN:-page fault|ring .*(timeout|reset)|GPU reset|\\bOops|\\bBUG:|WARNING: CPU:|blocked for more than|\*ERROR\*}"
SETTLE="${GPU_LOG_GATE_SETTLE:-3}"
PRESCAN="${GPU_LOG_GATE_PRESCAN:-10 minutes ago}"

if [ "${1:-}" = "--" ]; then shift; fi
if [ $# -eq 0 ]; then
  echo "usage: $0 -- <command> [args...]" >&2
  exit 2
fi

read_log() { # $1 = value of --since
  if [ -n "${GPU_LOG_GATE_LOG_CMD:-}" ]; then
    "$GPU_LOG_GATE_LOG_CMD" "$1"
  else
    journalctl -k --no-pager --since "$1"
  fi
}

# Prints matching lines; returns 0 if there are none, 1 if there are, 4 if the
# log cannot be read.
scan() { # $1 = since
  local out
  if ! out=$(read_log "$1" 2>&1); then
    echo "gpu-kernel-log-gate: cannot read the kernel log: $out" >&2
    return 4
  fi
  local hits
  local hits rc
  hits=$(printf '%s\n' "$out" | grep -E -i -- "$PATTERN")
  rc=$?
  if [ "$rc" -gt 1 ]; then
    echo "gpu-kernel-log-gate: grep failed on the pattern '$PATTERN' (exit $rc)" >&2
    return 4
  fi
  if [ -n "$hits" ]; then
    printf '%s\n' "$hits" >&2
    return 1
  fi
  return 0
}

# Positive control: a user who cannot read the system journal gets an empty,
# successful answer, which would read as a clean log. Real reader only.
if [ -z "${GPU_LOG_GATE_LOG_CMD:-}" ] && [ -z "$(journalctl -k --no-pager -q -n 1 2>/dev/null)" ]; then
  echo "gpu-kernel-log-gate: the kernel log is empty or not readable by this user (journal group?)" >&2
  if [ "${GPU_LOG_GATE_ALLOW_NO_LOG:-0}" != "1" ]; then exit 4; fi
fi

guarded=1
# Taken before the pre-scan so that the two scans overlap.
started=$(date '+%Y-%m-%d %H:%M:%S')
scan "$PRESCAN"
case $? in
  0) ;;
  1)
    echo "gpu-kernel-log-gate: GPU faults already in the kernel log (last $PRESCAN); not starting a GPU run" >&2
    exit 3
    ;;
  *)
    if [ "${GPU_LOG_GATE_ALLOW_NO_LOG:-0}" = "1" ]; then
      echo "gpu-kernel-log-gate: running UNGUARDED (no readable kernel log)" >&2
      guarded=0
    else
      exit 4
    fi
    ;;
esac

"$@"
status=$?

if [ "$guarded" = "1" ]; then
  # journald ingests kmsg asynchronously and a ring timeout is logged when the
  # job times out, possibly after a short-lived process has exited.
  sleep "$SETTLE"
  scan "$started"
  case $? in
    0) ;;
    1)
      echo "gpu-kernel-log-gate: FAIL, the kernel log shows GPU faults since $started (command exit status $status)" >&2
      exit 1
      ;;
    *) exit 4 ;;
  esac
fi
exit "$status"
