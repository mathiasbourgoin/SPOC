#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Run unit tests with code coverage measurement

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Clean previous coverage data
rm -f _coverage/*.coverage 2>/dev/null || true
mkdir -p _coverage

# bisect_ppx must already be available in the current build environment. This
# script never installs anything and never selects an opam switch: run it
# inside the switch you want (CI image, or `opam exec --switch=<path> --`, with
# OPAMSWITCH set for tools that read it). `ocamlfind query` is read-only and
# prints nothing useful when the library is absent, so test its exit code
# (docs review D-48).
echo "Checking for bisect_ppx..."
if ! ocamlfind query bisect_ppx >/dev/null 2>&1; then
  echo "ERROR: bisect_ppx is not available in the current environment." >&2
  echo "Install it in the switch you build with (the CI image must provide it)," >&2
  echo "then re-run. This script does not install packages." >&2
  exit 1
fi
if ! command -v bisect-ppx-report >/dev/null 2>&1; then
  echo "ERROR: bisect-ppx-report is not on PATH in the current environment." >&2
  exit 1
fi

echo "Building with coverage instrumentation..."
LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib:$LIBRARY_PATH \
  dune build --force --instrument-with bisect_ppx \
              sarek/tests/unit/ \
              sarek/core/test/ \
              spoc/framework/test/ \
              spoc/ir/test/ \
              spoc/registry/test/

echo ""
echo "Running unit tests..."
export BISECT_FILE="$PROJECT_ROOT/_coverage/unit"

# Run all sarek unit tests
for test in _build/default/sarek/tests/unit/test_*.exe; do
  echo "Running $(basename $test)..."
  $test
done

# Run core tests
for test in _build/default/sarek/core/test/test_*.exe; do
  echo "Running $(basename $test)..."
  $test
done

# Run spoc tests
for test in _build/default/spoc/*/test/test_*.exe; do
  echo "Running $(basename $test)..."
  $test
done

# The run must have produced coverage data; an empty _coverage means the
# instrumentation did not take effect and a green report would be a lie.
if ! ls _coverage/*.coverage >/dev/null 2>&1; then
  echo "ERROR: no _coverage/*.coverage files were produced by the unit tests" >&2
  exit 1
fi

echo ""
echo "Generating coverage report..."
bisect-ppx-report html \
  --coverage-path _coverage \
  -o _coverage/unit-report

bisect-ppx-report summary \
  --coverage-path _coverage

if [ ! -s _coverage/unit-report/index.html ]; then
  echo "ERROR: coverage report _coverage/unit-report/index.html was not generated" >&2
  exit 1
fi

echo ""
echo "Coverage report generated at: _coverage/unit-report/index.html"
