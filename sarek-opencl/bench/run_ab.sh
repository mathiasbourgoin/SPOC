#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Interleaved A/B of cold-process OpenCL kernel build latency (H5c).
#   A = SAREK_DISK_CACHE=0   (compile from source every process: the old behaviour)
#   B = warm disk cache      (one untimed priming run, then timed runs)
# Each sample is a separate process. Load average and the top CPU consumers are
# printed with every pair, because the machine is shared.
#
# Usage: run_ab.sh <bench-exe> <device-index> [pairs=5]
set -euo pipefail
exe=$1
device=$2
pairs=${3:-5}
cache_home=$(mktemp -d)
trap 'rm -rf "$cache_home"' EXIT
mkdir -p "$cache_home/spoc"

echo "# device index $device, $pairs pairs, cache dir $cache_home"
XDG_CACHE_HOME=$cache_home "$exe" "$device" >/dev/null # prime B
for i in $(seq 1 "$pairs"); do
  echo "# pair $i load: $(cut -d' ' -f1-3 /proc/loadavg)"
  top -b -n1 | sed -n '8,10p' | sed 's/^/# top: /'
  SAREK_DISK_CACHE=0 XDG_CACHE_HOME=$cache_home "$exe" "$device" | sed 's/^/A /'
  XDG_CACHE_HOME=$cache_home "$exe" "$device" | sed 's/^/B /'
done
