#!/bin/bash
# Run all benchmarks and update web data
# Usage: ./run_all_benchmarks.sh [output_dir] [--generate-backend-code]

set -e

# Show help
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  cat <<EOF
SAREK Benchmark Suite Runner

Usage: $0 [output_dir] [--generate-backend-code]

Runs all benchmarks and updates web viewer data.

Arguments:
  output_dir              Base directory for results (default: benchmarks/results)
                          Results saved to output_dir/run_TIMESTAMP/
  --generate-backend-code Also regenerate backend code for all benchmarks

Examples:
  $0                              # Save to benchmarks/results/run_TIMESTAMP/
  $0 my_results                   # Save to my_results/run_TIMESTAMP/
  $0 --generate-backend-code      # Also regenerate CUDA/OpenCL/Vulkan/Metal code
  make benchmarks                 # Same as running script directly

After running:
  1. Review results in the timestamped directory
  2. Optionally run 'make bench-deduplicate' to check for duplicates
  3. Commit updated benchmarks/results/*.json files in PR
  4. Reviewer will deduplicate before merging
EOF
  exit 0
fi

# Parse arguments
OUTPUT_DIR="benchmarks/results"
GENERATE_CODE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --generate-backend-code)
      GENERATE_CODE=true
      shift
      ;;
    *)
      if [[ ! "$1" == --* ]]; then
        OUTPUT_DIR="$1"
        shift
      fi
      ;;
  esac
done
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/run_${TIMESTAMP}"

echo "================================================"
echo "  SAREK Benchmark Suite Runner"
echo "================================================"
echo "Output directory: ${RUN_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Create output directory
mkdir -p "${RUN_DIR}"

echo "Building all benchmarks..."
dune build \
  benchmarks/bench_matrix_mul.exe \
  benchmarks/bench_matrix_mul_tiled.exe \
  benchmarks/bench_vector_add.exe \
  benchmarks/bench_vector_copy.exe \
  benchmarks/bench_stream_triad.exe \
  benchmarks/bench_reduction.exe \
  benchmarks/bench_transpose.exe \
  benchmarks/bench_transpose_tiled.exe \
  benchmarks/bench_mandelbrot.exe \
  benchmarks/to_web.exe

echo ""
echo "Running benchmarks..."
echo "This may take several minutes depending on your hardware."
echo ""

# Matrix Multiplication
echo "▶ Matrix Multiplication (naive)..."
dune exec benchmarks/bench_matrix_mul.exe -- --output "${RUN_DIR}"
echo "  ✓ Complete"
echo ""

# Matrix Multiplication (Tiled)
echo "▶ Matrix Multiplication (tiled)..."
dune exec benchmarks/bench_matrix_mul_tiled.exe -- --output "${RUN_DIR}"
echo "  ✓ Complete"
echo ""

# Vector Addition
echo "▶ Vector Addition..."
dune exec benchmarks/bench_vector_add.exe -- --output "${RUN_DIR}"
echo "  ✓ Complete"
echo ""

# Vector Copy
echo "▶ Vector Copy..."
dune exec benchmarks/bench_vector_copy.exe -- --output "${RUN_DIR}"
echo "  ✓ Complete"
echo ""

# STREAM Triad
echo "▶ STREAM Triad..."
dune exec benchmarks/bench_stream_triad.exe -- --output "${RUN_DIR}"
echo "  ✓ Complete"
echo ""

# Reduction
echo "▶ Parallel Reduction (sum)..."
dune exec benchmarks/bench_reduction.exe -- --output "${RUN_DIR}"
echo "  ✓ Complete"
echo ""

# Transpose (naive)
echo "▶ Matrix Transpose (naive)..."
dune exec benchmarks/bench_transpose.exe -- --output "${RUN_DIR}"
echo "  ✓ Complete"
echo ""

# Transpose (tiled)
echo "▶ Matrix Transpose (tiled)..."
dune exec benchmarks/bench_transpose_tiled.exe -- --output "${RUN_DIR}"
echo "  ✓ Complete"
echo ""

# Mandelbrot
echo "▶ Mandelbrot Set (with image generation)..."
dune exec benchmarks/bench_mandelbrot.exe -- --output "${RUN_DIR}"
echo "  ✓ Complete"
echo ""

# Count results
RESULT_COUNT=$(ls -1 "${RUN_DIR}"/*.json 2>/dev/null | wc -l)
echo "================================================"
echo "All benchmarks complete!"
echo "Generated ${RESULT_COUNT} result files in ${RUN_DIR}"
echo ""

# Move results to benchmarks/results/ for git tracking and CI
echo "Moving results to benchmarks/results/ for git tracking..."
if [ ${RESULT_COUNT} -gt 0 ]; then
  mv "${RUN_DIR}"/*.json "benchmarks/results/"
  echo "  ✓ Moved ${RESULT_COUNT} files to benchmarks/results/"
  
  # Remove empty run directory
  rmdir "${RUN_DIR}" 2>/dev/null || rm -rf "${RUN_DIR}"
  echo "  ✓ Cleaned up ${RUN_DIR}"
else
  echo "  ⚠ No results found to move"
fi
echo ""

# Update web data
echo "Updating web viewer data..."
WEB_OUTPUT="gh-pages/benchmarks/data/latest.json"
dune exec benchmarks/to_web.exe -- "${WEB_OUTPUT}" benchmarks/results/*.json

echo "  ✓ Updated ${WEB_OUTPUT}"
echo ""

# Generate backend code if requested
if [ "$GENERATE_CODE" = true ]; then
  echo "================================================"
  echo "Regenerating backend code for all benchmarks..."
  dune build benchmarks/generate_backend_code.exe
  dune exec benchmarks/generate_backend_code.exe
  
  # Copy to gh-pages
  echo "Copying generated code to gh-pages..."
  mkdir -p gh-pages/benchmarks/descriptions/generated
  cp benchmarks/descriptions/generated/*.md gh-pages/benchmarks/descriptions/generated/
  
  echo "  ✓ Backend code regenerated"
  echo ""
fi

echo "================================================"
echo "✅ Benchmark run complete!"
echo ""
echo "Results are ready in benchmarks/results/"
echo "Web viewer data updated: ${WEB_OUTPUT}"
echo ""
echo "Next steps:"
echo "  1. (Optional) Check for duplicates:"
echo "     make bench-deduplicate"
echo ""
echo "  2. Add results to git:"
echo "     git add benchmarks/results/*.json"
echo "     git add ${WEB_OUTPUT}"
if [ "$GENERATE_CODE" = true ]; then
  echo "     git add benchmarks/descriptions/generated/"
  echo "     git add gh-pages/benchmarks/descriptions/generated/"
fi
echo ""
echo "  3. Commit and push:"
echo "     git commit -m \"Add benchmark results from $(hostname) ($(date +%Y-%m-%d))\""
echo "     git push origin <your-branch>"
echo ""
echo "  4. Create PR - CI will generate preview at:"
echo "     https://mathiasbourgoin.github.io/Sarek/preview/pr-<number>/benchmarks/"
echo ""
echo "Reviewers can deduplicate before merging with:"
echo "  dune exec benchmarks/deduplicate_results.exe"
echo "================================================"
