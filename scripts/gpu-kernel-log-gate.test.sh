#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Red-path harness for gpu-kernel-log-gate.sh. No GPU, no journal: the log
# reader is replaced by a fake that prints a chosen text. Every case asserts the
# exit status; a gate that cannot fail would read like a passing one.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
GATE=scripts/gpu-kernel-log-gate.sh
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
fail=0
export GPU_LOG_GATE_SETTLE=0

# fake log reader: prints $work/log-before for the pre-scan window (10 minutes
# ago) and $work/log-after for any other window.
cat >"$work/reader.sh" <<EOF
#!/usr/bin/env bash
case "\$1" in
  *minutes*) cat "$work/log-before" ;;
  *) cat "$work/log-after" ;;
esac
EOF
chmod +x "$work/reader.sh"

expect() { # name want_status -- command...
  local name="$1" want="$2"
  shift 3
  GPU_LOG_GATE_LOG_CMD="$work/reader.sh" "$GATE" -- "$@" >/dev/null 2>&1
  local got=$?
  if [ "$got" = "$want" ]; then echo "PASS: $name"; else echo "FAIL: $name (want $want, got $got)"; fail=1; fi
}

clean='amdgpu 0000:03:00.0: [drm] something harmless'
: >"$work/log-before"; echo "$clean" >"$work/log-after"
expect "clean log, passing command passes" 0 -- true
expect "clean log, failing command keeps its status" 7 -- bash -c 'exit 7'

echo 'amdgpu 0000:03:00.0: [gfxhub] page fault (src_id:0 ring:40 vmid:7)' >"$work/log-after"
expect "page fault after a passing run fails" 1 -- true

echo 'amdgpu 0000:03:00.0: ring gfx_0.0.0 timeout, signaled seq=1, emitted seq=2' >"$work/log-after"
expect "ring timeout after a passing run fails" 1 -- true

echo 'blocked for more than 120 seconds' >"$work/log-after"
expect "hung task fails" 1 -- true

echo 'Oops: Oops: 0000 [#1] SMP NOPTI' >"$work/log-after"
expect "oops fails" 1 -- true

echo 'BUG: kernel NULL pointer dereference, address: 0000000000000000' >"$work/log-after"
expect "BUG fails" 1 -- true

echo "$clean" >"$work/log-after"
echo 'amdgpu: page fault from an earlier run' >"$work/log-before"
expect "fault already present: command not started, exit 3" 3 -- true
if GPU_LOG_GATE_LOG_CMD="$work/reader.sh" "$GATE" -- touch "$work/ran" >/dev/null 2>&1; [ -e "$work/ran" ]; then
  echo "FAIL: the command ran despite a faulting log"; fail=1
else
  echo "PASS: the command did not run despite a faulting log"
fi

: >"$work/log-before"
printf '#!/usr/bin/env bash\nexit 1\n' >"$work/broken.sh"; chmod +x "$work/broken.sh"
GPU_LOG_GATE_LOG_CMD="$work/broken.sh" "$GATE" -- true >/dev/null 2>&1
if [ $? = 4 ]; then echo "PASS: unreadable log fails closed"; else echo "FAIL: unreadable log did not fail closed"; fail=1; fi
GPU_LOG_GATE_ALLOW_NO_LOG=1 GPU_LOG_GATE_LOG_CMD="$work/broken.sh" "$GATE" -- true >/dev/null 2>&1
if [ $? = 0 ]; then echo "PASS: explicit opt-out runs unguarded"; else echo "FAIL: opt-out"; fail=1; fi

"$GATE" >/dev/null 2>&1
if [ $? = 2 ]; then echo "PASS: no command is a usage error"; else echo "FAIL: usage"; fail=1; fi

exit "$fail"
