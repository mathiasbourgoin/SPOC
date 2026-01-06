#!/bin/bash
# Verification script for Priority 14 (Sarek_tailrec split)
# Run this in a terminal with the local opam switch active

set -e

echo "=== Verifying Sarek_tailrec module split ==="
echo ""

echo "Module breakdown:"
echo "  Sarek_tailrec_analysis.ml: 233 lines (recursion analysis)"
echo "  Sarek_tailrec_elim.ml:     365 lines (tail recursion elimination)"
echo "  Sarek_tailrec_bounded.ml:  100 lines (bounded recursion inlining)"
echo "  Sarek_tailrec_pragma.ml:   373 lines (pragma-based inlining)"
echo "  Sarek_tailrec.ml:          145 lines (kernel-level pass, public API)"
echo "  Total:                    1216 lines (was 1175 lines)"
echo ""

echo "1. Clean build..."
dune clean

echo "2. Building project..."
dune build

echo "3. Running test-all..."
make test-all

echo "4. Running benchmarks..."
make benchmarks

echo ""
echo "=== All tests passed! ==="
echo ""
echo "Ready to commit:"
echo "  git add -A"
echo "  git commit -m 'Split Sarek_tailrec into 5 focused modules'"
